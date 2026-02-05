from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CLIPVisionModel, CLIPImageProcessor
import torch
from PIL import Image

# 建议使用 bf16 或 fp32，Qwen2.5 在 fp16 下有时不稳定
target_dtype = torch.float32 

class VLMModel(torch.nn.Module):
    def __init__(self, llm_name="Qwen/Qwen2.5-0.5B-Instruct", vision_name="openai/clip-vit-base-patch16", projector_params=None):
        super().__init__()
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        # todo 临时用cpu
        # device = "cpu"
        self.device = torch.device(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        # 【关键修复】显式添加特殊 Token
        self.tokenizer.add_tokens(["<image>"], special_tokens=True)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")

        self.language_model = AutoModelForCausalLM.from_pretrained(
            llm_name,
            dtype=target_dtype,
            device_map=self.device,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True
        )
        # 【关键修复】调整 LLM 的 Embedding 层大小以匹配新增的 <image> token
        self.language_model.resize_token_embeddings(len(self.tokenizer))

        self.vision_encoder = CLIPVisionModel.from_pretrained(
            vision_name,
            dtype=target_dtype, 
            device_map=self.device
        )
        self.vision_processor = CLIPImageProcessor.from_pretrained(vision_name)

        self.llm_hidden_dim = self.language_model.config.hidden_size
        
        # Projector 结构保持不变
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(768, 2048),
            torch.nn.LayerNorm(2048),
            torch.nn.GELU(),
            torch.nn.Linear(2048, self.llm_hidden_dim),
            torch.nn.LayerNorm(self.llm_hidden_dim)
        ).to(dtype=target_dtype, device=self.device)

        # 初始化 Projector
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        self.projector.apply(init_weights)

        if projector_params is not None:
            self.projector.load_state_dict(projector_params)
        
        # 冻结
        for param in self.language_model.parameters():
            param.requires_grad = False
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        # 确保 Projector 可训练
        for param in self.projector.parameters():
            param.requires_grad = True

    def forward(self, input_ids, pixel_values, labels=None):
        visual_outputs = self.vision_encoder(pixel_values)
        # [B, 576, 768] -> [B, 576, LLM_DIM]
        image_features = self.projector(visual_outputs.last_hidden_state[:, 1:, :])
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        batch_size = input_ids.size(0)
        new_embeds = []
        new_labels = []

        for i in range(batch_size):
            # 【关键修复】直接通过 ID 寻找 <image> 占位符的位置
            image_token_mask = (input_ids[i] == self.image_token_id)
            indices = torch.where(image_token_mask)[0]

            # 如果image标签没有找到，要打印出来看看，明显一点的错误
            if len(indices) == 0:
                #先打印Image token的id
                print(f"Image token id: {self.image_token_id}")
                # 再打印当前的input_ids
                print(f"Current input_ids: {input_ids[i]}")
                # 再打印当前的input_ids中image token的位置
                print(f"Image token indices: {indices}")
                # 再打印当前的input_ids中image token的位置
                print(f"Warning: No <image> token found in input_ids ❌[{i}]")

            if len(indices) == 0:
                # 如果没图，退化为普通文本处理
                new_embeds.append(inputs_embeds[i])
                if labels is not None:
                    new_labels.append(labels[i])
            else:
                # 假设每条数据只有一个 <image> 占位符
                idx = indices[0]

                # 拼接：前文本 + 图像特征 + 后文本
                # 注意：这里完全跳过了原有的 <image> token embedding
                cur_embeds = torch.cat([
                    inputs_embeds[i, :idx, :],
                    image_features[i],
                    inputs_embeds[i, idx + 1:, :]
                ], dim=0)
                new_embeds.append(cur_embeds)

                if labels is not None:
                    # 标签同步偏移：前标签 + 图像占位(-100) + 后标签
                    img_labels = torch.full((image_features.size(1),), -100,
                                          dtype=labels.dtype, device=labels.device)
                    cur_labels = torch.cat([
                        labels[i, :idx],
                        img_labels,
                        labels[i, idx + 1:]
                    ], dim=0)
                    new_labels.append(cur_labels)

        # 动态 Padding
        max_len = max(e.size(0) for e in new_embeds)
        final_embeds = torch.zeros(batch_size, max_len, self.llm_hidden_dim,
                                 dtype=target_dtype, device=self.device)
        # 使用 tokenizer 的 pad_token_id 填充
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        final_labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=self.device)

        for i in range(batch_size):
            cur_len = new_embeds[i].size(0)
            final_embeds[i, :cur_len, :] = new_embeds[i]
            if labels is not None:
                final_labels[i, :cur_len] = new_labels[i]

        # --- 临时调试代码修正版 ---
        if labels is not None and torch.rand(1).item() < 0.05:
            # 1. 正常的 Forward 已经拿到了 logits
            # 这里的 logits 维度是 [B, N, Vocabulary_Size]
            output = self.language_model(inputs_embeds=final_embeds, labels=final_labels if labels is not None else None,return_dict=True)
            logits = output.logits

            # 2. 执行 Shift 操作，对齐预测与标签
            # 预测序列：   [P0, P1, P2, P3] (logits 里的 0 到 n-1)
            # 对应目标：   [L1, L2, L3, L4] (labels 里的 1 到 n)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = final_labels[:, 1:].contiguous()

            # 3. 构造 Mask，只解码我们需要关注的（Assistant 回答且非 Padding 部分）
            valid_mask = (shift_labels != -100)

            # 4. 随机采样打印，避免刷屏
            print(f"\n" + "="*50)
            # 找到当前 batch 中第一个有有效标签的样本
            for i in range(batch_size):
                m = valid_mask[i]
                if m.any():
                    # 提取预测和真实标签
                    p_tokens = torch.argmax(shift_logits[i][m], dim=-1)
                    t_tokens = shift_labels[i][m]

                    # 解码成文本
                    pred_str = self.tokenizer.decode(p_tokens, skip_special_tokens=False)
                    target_str = self.tokenizer.decode(t_tokens, skip_special_tokens=False)

                    print(f"【Step 调试采样】")
                    print(f"预测文本 >> {pred_str}")
                    print(f"实际文本 >> {target_str}")

                    # 额外检查：对比前 5 个 token 的 ID 是否一致（直观判断对齐）
                    # print(f"前5个Token ID对比: \nPred: {p_tokens[:5].tolist()}\nTarget: {t_tokens[:5].tolist()}")
                    break
            print("="*50 + "\n")
            return output
            # print("------------------")
            # print("image token id:", self.image_token_id)
            # print(f"预测 ID 前 10 个: {pred_tokens[:10].tolist()}")
            # print(f"实际 ID 前 10 个: {target_tokens[:10].tolist()}")
            # exit(-1)

            # 3. 检查图像特征的强度
            # print(f"Image Features Mean Abs: {image_features.abs().mean().item()}")
            # ------------------

        return self.language_model(
            inputs_embeds=final_embeds,
            labels=final_labels if labels is not None else None,
            return_dict=True
        )

    @torch.no_grad()
    def answer(self, image: Image.Image, prompt: str, max_new_tokens=128):
        self.eval()
        # 1. 严格对齐 Qwen 的 ChatML 格式
        # 必须让模型觉得 Assistant 准备开口了        
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs['input_ids'].to(self.device)
        
        # 2. 定位 <image>
        image_indices = torch.where(input_ids[0] == self.image_token_id)[0]
        if len(image_indices) == 0:
            return "Error: No <image> tag found in prompt."
        idx = image_indices[0]

        # 3. 准备图像特征
        pixel_values = self.vision_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device=self.device, dtype=target_dtype)
        visual_outputs = self.vision_encoder(pixel_values)
        image_features = self.projector(visual_outputs.last_hidden_state[:, 1:, :])

        # 4. 拼接 Embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        combined_embeds = torch.cat([
            inputs_embeds[:, :idx, :],
            image_features,
            inputs_embeds[:, idx + 1:, :]
        ], dim=1)

        bad_ids = self.tokenizer.encode("] /", add_special_tokens=False)        # 5. 生成
        # 注意：这里我们只传 inputs_embeds，不传 input_ids
        output_ids = self.language_model.generate(
            inputs_embeds=combined_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=False,            # Greedy search 比较稳
            repetition_penalty=1.2,     # 稍微加大惩罚，防止口吃
            begin_suppress_tokens=bad_ids,  # 强行禁止开场白出现这些脏数据
            eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"), # 明确结束符
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # 6. 解码
        # generate 返回的通常只有新生成的 token
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response
    
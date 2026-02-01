
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CLIPVisionModel, CLIPImageProcessor
import torch
from PIL import Image

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"



target_dtype = torch.float32

class VLMModel(torch.nn.Module):
    def __init__(self, llm_name="Qwen/Qwen2.5-0.5B-Instruct", vision_name="openai/clip-vit-base-patch16", projector_params=None):
        super().__init__()
        self.llm_name = llm_name
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.language_model = AutoModelForCausalLM.from_pretrained(
            llm_name,
            dtype=target_dtype,
            device_map=device,
            attn_implementation="sdpa", 
            # 节省显存的进阶配置
            low_cpu_mem_usage=True
        )

        self.vision_encoder = CLIPVisionModel.from_pretrained(
            vision_name,
            dtype=target_dtype, 
            device_map=device
        )
    
        self.vision_processor = CLIPImageProcessor.from_pretrained(vision_name)

        self.image_ids = self.tokenizer.encode("<image>", add_special_tokens=False)
        self.image_len = len(self.image_ids)
        
        print(f"检测到占位符序列: {self.image_ids} (长度: {self.image_len})")
        # 注册为 buffer，这样保存模型时会带上，且会自动跟随模型移动到 GPU
        self.register_buffer("target_ids", torch.tensor(self.image_ids))

        self.llm_hidden_dim = self.language_model.config.hidden_size
        print(f"{self.llm_name} LLM hidden dim: {self.llm_hidden_dim}")

        self.projector = torch.nn.Sequential(
            torch.nn.Linear(768, 2048),      # 第一层：先映射到更高维度进行特征提取
            torch.nn.LayerNorm(2048),
            torch.nn.GELU(),                # 非线性激活
            torch.nn.Linear(2048, self.llm_hidden_dim),       # 第二层：映射到 LLM 的隐藏层维度
            torch.nn.LayerNorm(self.llm_hidden_dim)
        ).to(dtype=target_dtype, device=device)


        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                # 使用更小的标准差进行初始化，比如 0.01 或 0.02
                torch.nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        # 默认加载 projector 权重
        self.projector.apply(init_weights)

        # 加载 projector 参数
        if projector_params is not None:
            self.projector.load_state_dict(projector_params)
        
        for param in self.language_model.parameters():
            param.requires_grad = False
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        # 只留 self.projector 训练  
        self.projector.to(device, dtype=target_dtype)

        # 【关键】最后强制开启 Projector 梯度
        # 这样做可以确保无论之前发生了什么，Projector 都是可训练的
        for param in self.projector.parameters():
            param.requires_grad = True


    # def forward(self, input_ids, pixel_values, labels=None):
    #     visual_outputs = self.vision_encoder(pixel_values)
    #     image_features = self.projector(visual_outputs.last_hidden_state)

    #     inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

    #     combined_embeds = torch.cat([inputs_embeds, image_features], dim=1)


    #     # 扩展 labels 以匹配拼接后的序列长度
    #     batch_size = input_ids.size(0)
    #     image_seq_len = image_features.size(1)
        
    #     # 创建图像部分的标签，全部设为 -100（忽略）
    #     if labels is not None:
    #         image_labels = torch.full((batch_size, image_seq_len), -100, dtype=labels.dtype, device=labels.device)
    #         extended_labels = torch.cat([image_labels, labels], dim=1)
    #     else:
    #         extended_labels = None

    #     # 4. 喂给 LLM
    #     outputs = self.language_model(
    #         inputs_embeds=combined_embeds,
    #         labels=extended_labels, # 如果传了 labels 就会自动计算 Loss
    #         return_dict=True
    #     )
    
    #     return outputs

    def forward(self, input_ids, pixel_values, labels=None):
        visual_outputs = self.vision_encoder(pixel_values)
        # CLIP 特征去掉 CLS token: [B, 577, 768] -> [B, 576, 768]
        image_features = self.projector(visual_outputs.last_hidden_state[:, 1:, :]) 
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        new_embeds = []
        new_labels = []

        # 遍历 Batch
        for i in range(input_ids.size(0)):
            # 1. 将当前的 input_ids 还原为文本，必须拿到 offset_mapping
            # 我们直接对这一行进行处理
            curr_ids = input_ids[i]
            # 过滤掉 padding 避免干扰 decode
            non_pad_ids = curr_ids[curr_ids != self.tokenizer.pad_token_id]
            text = self.tokenizer.decode(non_pad_ids, add_special_tokens=False)
            
            char_idx = text.find("<image>")
            
            # 重新 encode 拿到 offset_mapping
            encoding = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            offsets = encoding['offset_mapping']
            
            token_idx = -1
            token_len = 0
            
            if char_idx != -1:
                matched_tokens = []
                for t_idx, (start, end) in enumerate(offsets):
                    if (start >= char_idx and end <= char_idx + 7) or (start < char_idx + 7 and end > char_idx):
                        matched_tokens.append(t_idx)
                
                if matched_tokens:
                    token_idx = matched_tokens[0]
                    token_len = len(matched_tokens)

            # 2. 手术拼接
            if token_idx == -1:
                # 打印错误用于 Debug，但通过 dummy_filler 保持训练不崩
                # print(f"❌ Batch {i} 匹配失败") 
                dummy_filler = self.projector[0].weight.view(-1)[0] * 0
                new_embeds.append(inputs_embeds[i] + dummy_filler)
                if labels is not None: new_labels.append(labels[i])
            else:
                # 考虑到可能存在的 Padding 偏移，如果是非左填充，token_idx 是准确的
                pre_e = inputs_embeds[i, :token_idx, :]
                post_e = inputs_embeds[i, token_idx + token_len:, :]
                new_embeds.append(torch.cat([pre_e, image_features[i], post_e], dim=0))
                
                if labels is not None:
                    pre_l = labels[i, :token_idx]
                    post_l = labels[i, token_idx + token_len:]
                    img_l = torch.full((image_features.size(1),), -100, device=labels.device, dtype=labels.dtype)
                    new_labels.append(torch.cat([pre_l, img_l, post_l], dim=0))

        # 3. 动态 Padding (保持你原有的逻辑，它是对的)
        max_len = max(e.size(0) for e in new_embeds)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        pad_embed_single = self.language_model.get_input_embeddings()(
            torch.tensor([pad_id], device=inputs_embeds.device)
        ).squeeze(0)

        final_embeds = []
        final_labels = []
        for i in range(len(new_embeds)):
            curr_len = new_embeds[i].size(0)
            diff = max_len - curr_len
            if diff > 0:
                pad_f = pad_embed_single.repeat(diff, 1)
                final_embeds.append(torch.cat([new_embeds[i], pad_f], dim=0))
                if labels is not None:
                    pad_l = torch.full((diff,), -100, device=labels.device, dtype=labels.dtype)
                    final_labels.append(torch.cat([new_labels[i], pad_l], dim=0))
            else:
                final_embeds.append(new_embeds[i])
                if labels is not None: final_labels.append(new_labels[i])

        return self.language_model(
            inputs_embeds=torch.stack(final_embeds), 
            labels=torch.stack(final_labels) if labels is not None else None,
            return_dict=True
        )

    @torch.no_grad()
    def answer(self, image: Image.Image, question: str, max_new_tokens=128):
        self.eval()
        prompt = f"<|im_start|>user\n<image>\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        # 统一使用 offset_mapping 定位，彻底抛弃 ID 匹配
        inputs = self.tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs['input_ids'].to(self.language_model.device)
        offsets = inputs['offset_mapping'][0]
        
        char_idx = prompt.find("<image>")
        token_idx = -1
        token_len = 0
        for t_idx, (start, end) in enumerate(offsets):
            if (start >= char_idx and end <= char_idx + 7) or (start < char_idx + 7 and end > char_idx):
                if token_idx == -1: 
                    token_idx = t_idx
                token_len += 1

        pixel_values = self.vision_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device=self.vision_encoder.device, dtype=self.vision_encoder.dtype)
        
        visual_outputs = self.vision_encoder(pixel_values)
        image_features = self.projector(visual_outputs.last_hidden_state[:, 1:, :])

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        combined_embeds = torch.cat([
            inputs_embeds[:, :token_idx, :],
            image_features,
            inputs_embeds[:, token_idx + token_len:, :]
        ], dim=1)

        output_ids = self.language_model.generate(
            inputs_embeds=combined_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=False, 
            repetition_penalty=1.1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
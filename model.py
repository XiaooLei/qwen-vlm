
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

        self.llm_hidden_dim = self.language_model.config.hidden_size
        print(f"{self.llm_name} LLM hidden dim: {self.llm_hidden_dim}")

        self.projector = self.projector = torch.nn.Sequential(
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

        self.language_model.get_input_embeddings().requires_grad_(True)


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
        image_features = self.projector(visual_outputs.last_hidden_state[:, 1:, :]) 
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        
        # 准备容器存放处理后的 Embeds 和 Labels
        new_embeds = []
        new_labels = []

        for i in range(input_ids.size(0)):
            # 找到当前这行 <image> 的位置
            indices = (input_ids[i] == image_token_id).nonzero(as_tuple=True)[0]
            
            if len(indices) == 0:
                # 【防崩点】如果没有 <image> 标签，直接原样使用文本
                # 这在处理多轮对话的后续回复时很常见
                new_embeds.append(inputs_embeds[i])
                if labels is not None:
                    new_labels.append(labels[i])
            else:
                # 正常的替换逻辑
                idx = indices[0].item()
                pre_e = inputs_embeds[i, :idx, :]
                post_e = inputs_embeds[i, idx+1:, :]
                new_embeds.append(torch.cat([pre_e, image_features[i], post_e], dim=0))
                
                if labels is not None:
                    pre_l = labels[i, :idx]
                    post_l = labels[i, idx+1:]
                    img_l = torch.full((image_features.size(1),), -100, device=labels.device, dtype=labels.dtype)
                    new_labels.append(torch.cat([pre_l, img_l, post_l], dim=0))

        # 重新打包回 Tensor
        combined_embeds = torch.stack(new_embeds, dim=0)
        extended_labels = torch.stack(new_labels, dim=0) if labels is not None else None

        return self.language_model(inputs_embeds=combined_embeds, labels=extended_labels)

    @torch.no_grad()
    def answer(self, image: Image.Image, question: str, max_new_tokens=128):
        """用于单张图片的推理回复"""
        self.eval()
        # 1. 构造标准对话模板 (必须和训练一致)
        prompt = f"<|im_start|>user\n<image>\n{question}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.language_model.device)
        
        # 2. 处理图片
        pixel_values = self.vision_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device=self.vision_encoder.device, dtype=self.vision_encoder.dtype)

        # 3. 提取图像特征并接入 Projector
        visual_outputs = self.vision_encoder(pixel_values)
        image_features = self.projector(visual_outputs.last_hidden_state[:, 1:, :])

        # 4. 找到文本中的 <image> 占位符进行替换
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        
        # 查找位置并拼接
        idx = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0][0].item()
        combined_embeds = torch.cat([
            inputs_embeds[:, :idx, :],
            image_features,
            inputs_embeds[:, idx+1:, :]
        ], dim=1)

        # 5. 调用 LLM 生成
        output_ids = self.language_model.generate(
            inputs_embeds=combined_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)



def generate(self, input_ids, pixel_values, max_new_tokens=128):
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            # 1. 同样裁掉第0个Token，保持训练推理一致
            visual_outputs = self.vision_encoder(pixel_values)
            image_features = self.projector(visual_outputs.last_hidden_state[:, 1:, :])
            
            # 2. 同样的 Embedding 逻辑
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            
            # 3. 寻找 <image> 标签位置
            image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
            # 假设 generate 时 batch 为 1
            image_idx = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0].item()
            
            # 4. 同样的三段式拼接
            pre_embeds = inputs_embeds[:, :image_idx, :]
            post_embeds = inputs_embeds[:, image_idx+1:, :]
            combined_embeds = torch.cat([pre_embeds, image_features, post_embeds], dim=1)

            # 5. 生成
            generated_ids = self.language_model.generate(
                inputs_embeds=combined_embeds,
                max_new_tokens=max_new_tokens,
                do_sample=False, # 推理时建议先关掉随机
                repetition_penalty=1.2 # 防止复读机
            )
            return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    
    

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CLIPVisionModel, CLIPImageProcessor
import torch

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"



target_dtype = torch.float16

class VLMModel(torch.nn.Module):
    def __init__(self, llm_name="Qwen/Qwen2.5-0.5B", vision_name="openai/clip-vit-base-patch16", projector_params=None):
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

        self.projector = self.projector = torch.nn.Sequential(
            torch.nn.Linear(768, 2048),      # 第一层：先映射到更高维度进行特征提取
            torch.nn.LayerNorm(2048),
            torch.nn.GELU(),                # 非线性激活
            torch.nn.Linear(2048, 896)       # 第二层：映射到 LLM 的隐藏层维度
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


    def forward(self, input_ids, pixel_values, labels=None):
        visual_outputs = self.vision_encoder(pixel_values)
        image_features = self.projector(visual_outputs.last_hidden_state)

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        combined_embeds = torch.cat([image_features, inputs_embeds], dim=1)


        # 扩展 labels 以匹配拼接后的序列长度
        batch_size = input_ids.size(0)
        image_seq_len = image_features.size(1)
        
        # 创建图像部分的标签，全部设为 -100（忽略）
        if labels is not None:
            image_labels = torch.full((batch_size, image_seq_len), -100, dtype=labels.dtype, device=labels.device)
            extended_labels = torch.cat([image_labels, labels], dim=1)
        else:
            extended_labels = None

        # 4. 喂给 LLM
        outputs = self.language_model(
            inputs_embeds=combined_embeds,
            labels=extended_labels, # 如果传了 labels 就会自动计算 Loss
            return_dict=True
        )
    
        return outputs
    
    def generate(self, input_ids, pixel_values, max_new_tokens=128):
        self.eval()
        with torch.no_grad():
            visual_outputs = self.vision_encoder(pixel_values)
            image_features = self.projector(visual_outputs.last_hidden_state)
            
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            combined_embeds = torch.cat([image_features, inputs_embeds], dim=1)

            generated_ids = self.language_model.generate(
                inputs_embeds=combined_embeds,
                max_new_tokens=max_new_tokens
            )
            return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    
    
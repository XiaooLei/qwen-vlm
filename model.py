
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CLIPVisionModel, CLIPImageProcessor
import torch

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"



class VLMModel(torch.nn.Module):
    def __init__(self, llm_name="Qwen/Qwen2.5-0.5B", vision_name="openai/clip-vit-large-patch14", projector_params=None):
        super().__init__()
        self.llm_name = llm_name
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.language_model = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.float32,
            device_map=device,
        )

        self.vision_encoder = CLIPVisionModel.from_pretrained(
            vision_name,
            torch_dtype=torch.float32,
            device_map=device
        )
    
        self.vision_processor = CLIPImageProcessor.from_pretrained(vision_name)

        self.projector = torch.nn.Linear(
            self.vision_encoder.config.hidden_size,
            self.language_model.config.hidden_size
        )

        # 加载 projector 参数
        if projector_params is not None:
            self.projector.load_state_dict(projector_params)

        self.projector.to(device)


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
    
    
    
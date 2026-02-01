



from model import VLMModel
import torch
import requests
from PIL import Image
import os
import sys

from transformers import CLIPVisionModel, CLIPImageProcessor

# 自动检测设备
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps" 
    else:
        return "cpu"

device = get_device()


def generate(llm_name="Qwen/Qwen2.5-0.5B-Instruct", 
             vision_name="openai/clip-vit-base-patch16",
             projector_path="checkpoints/checkpoint_epoch_2.pt",
             image_url="http://images.cocodataset.org/val2017/000000039769.jpg",
             input_text="this is a picture of",
             max_new_tokens=100):
    
    print(f"Using device: {device}")
    
    # 检查投影器文件是否存在
    if not os.path.exists(projector_path):
        print(f"Error: Projector file {projector_path} not found!")
        return None
    
    try:
        model = VLMModel(
            llm_name=llm_name,
            vision_name=vision_name,
        )
        model.to(device)
        model.eval()
        
        # 加载投影器权重
        projector_state_dict = torch.load(projector_path, map_location=device)
        model.projector.load_state_dict(projector_state_dict)
        model.projector.eval()
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None


    processor = CLIPImageProcessor.from_pretrained(vision_name)
    
    image_path = "/Users/admin/Desktop/workspace/ai/nlp-beginer/lab6-vlm/llava_data/train2017/000000000081.jpg"
    # url = "http://images.cocodataset.org/val2017/000000000081.jpg" # 经典的猫咪图
    #image = Image.open(requests.get(url, stream=True).raw)
    image = Image.open(image_path)

    # 4. 图像预处理
    # return_tensors="pt" 得到 PyTorch 张量
    pixel_values = processor(images=image, return_tensors="pt")   
    pixel_values = pixel_values.pixel_values

    input_text = "this is a picture of"

    input_ids = model.tokenizer.encode(input_text, return_tensors="pt")
    input_ids = input_ids.to(device)
    pixel_values = pixel_values.to(device)
    
    image_features = model.vision_encoder(pixel_values)
    image_features = model.projector(image_features.last_hidden_state)

    text_embeds = model.language_model.get_input_embeddings()(input_ids)

    print(f"image_features norm: {image_features.norm()}")
    print(f"text_embeds norm: {text_embeds.norm()}")

    combined_embeds = torch.cat([text_embeds, image_features], dim=1)

    # 计算总长度
    n_image_tokens = image_features.shape[1]
    n_text_tokens = input_ids.shape[1]
    combination_len = n_image_tokens + n_text_tokens

    # 创建 mask (batch_size=1)
    attention_mask = torch.ones((1, combination_len), device=device)

    output_ids = model.language_model.generate(
        inputs_embeds=combined_embeds,
        max_new_tokens=20,
        do_sample=False,
        attention_mask=attention_mask
    )
    output_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated text: {output_text}")


if __name__ == "__main__":
    # 支持命令行参数
    import argparse
    
    parser = argparse.ArgumentParser(description='Test VLM Model')
    parser.add_argument('--llm', default="Qwen/Qwen2.5-0.5B", help='LLM model name')
    parser.add_argument('--vision', default="openai/clip-vit-base-patch16", help='Vision model name')
    parser.add_argument('--image', default="http://images.cocodataset.org/val2017/000000039769.jpg", help='Image URL')
    parser.add_argument('--text', default="The object in this image is a", help='Input text prompt')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum new tokens to generate')
    
    args = parser.parse_args()
    
    result = generate(
        llm_name=args.llm,
        vision_name=args.vision, 
        image_url=args.image,
        input_text=args.text,
        max_new_tokens=args.max_tokens
    )
    
    if result is None:
        sys.exit(1)
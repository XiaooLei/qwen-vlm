import torch


# 加载clip模型import torch
from PIL import Image
import requests
from transformers import CLIPVisionModel, CLIPImageProcessor

# 1. 选择模型（LLaVA 常用的是 clip-vit-large-patch14）
model_name = "openai/clip-vit-large-patch14"

# 2. 加载图像处理器和视觉模型
# Processor 负责 Resize, Crop, Normalize
processor = CLIPImageProcessor.from_pretrained(model_name)
# VisionModel 只包含视觉编码部分，不含文本分支
model = CLIPVisionModel.from_pretrained(model_name)

# 3. 准备一张测试图片（可以用本地图，也可以用网络图）
url = "http://images.cocodataset.org/val2017/000000039769.jpg" # 经典的猫咪图
image = Image.open(requests.get(url, stream=True).raw)

# 4. 图像预处理
# return_tensors="pt" 得到 PyTorch 张量
inputs = processor(images=image, return_tensors="pt")

# 5. 推理提取特征
with torch.no_grad():
    outputs = model(**inputs)

# 6. 观察输出维度（这是复现 LLaVA 最关键的一步）
last_hidden_state = outputs.last_hidden_state

print(f"图像被切分后的特征形状: {last_hidden_state.shape}") 
# 预期输出: [1, 257, 1024]
# 1: Batch Size
# 257: 256个图像补丁(Patch) + 1个分类标记(CLS token)
# 1024: 每个特征向量的维度 (Feature Dimension)

print("\n--- 关键参数解读 ---")
print(f"视觉特征维度: {last_hidden_state.size(-1)} (这就是你 Projector 的输入维度)")
print(f"视觉 Token 数量: {last_hidden_state.size(1)} (这就是这张图会占用多少个单词的位置)")
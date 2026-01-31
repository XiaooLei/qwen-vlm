import os
import json
import requests
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
import re
from transformers import AutoTokenizer, CLIPImageProcessor

class LLaVADataset(Dataset):
    """LLaVA 数据集加载器"""
    
    def __init__(self, data_dir: str = "./llava_data", is_train: bool = True, llm_name="Qwen/Qwen2.5-0.5B", vision_name="openai/clip-vit-base-patch32", sample_size: int = None):
        """
        初始化数据集加载器
        
        Args:
            data_dir: 数据目录路径
            is_train: 是否是训练模式
            llm_name: 语言模型名称
            vision_name: 视觉模型名称
            sample_size: 样本数量（None表示使用所有样本）
        """
        self.sample_size = sample_size
        self.data_dir = data_dir
        self.is_train = is_train
        self.annotations_file = os.path.join(data_dir, "llava_instruct_150k.json")
        self.images_dir = os.path.join(data_dir, "train2017")
        self.data = []
        
        # 初始化tokenizer和image processor
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_name)
        
        # 如果tokenizer没有pad_token，则设置为eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self) -> int:
        """返回数据集样本数量"""
        if self.sample_size is None:
            return len(self.data)
        return min(self.sample_size, len(self.data))
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        获取指定索引的样本
        
        Args:
            index: 样本索引
            
        Returns:
            样本数据，包含input_ids, pixel_values, labels
        """
        sample = self.get_sample(index)
        
        # 处理图像
        image_path = sample.get('image_path')
        if image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
                pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
                pixel_values = pixel_values.to(torch.bfloat16)
            except Exception as e:
                # 如果图像无法打开（如损坏或格式不正确），返回零张量
                print(f"警告: 无法打开图片 {image_path}: {e}")
                pixel_values = torch.zeros(3, 224, 224)  # CLIP默认尺寸
                pixel_values = pixel_values.to(torch.bfloat16)  
        else:
            # 如果图像不存在，返回零张量
            pixel_values = torch.zeros(3, 224, 224)  # CLIP默认尺寸
            pixel_values = pixel_values.to(torch.bfloat16)  
        
        # 构建文本输入
        conversations = sample['conversations']
        text = self._build_conversation_text(conversations)
        
        # 编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # 构建标签 - 对于语言模型，通常是输入向右偏移一位
        labels = input_ids.clone()
        # 将padding位置的标签设为-100（忽略）
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'pixel_values': pixel_values,
            'labels': labels,
            'attention_mask': attention_mask
        }
        
    def load(self) -> List[Dict[str, Any]]:
        """
        加载数据集
        
        Returns:
            加载的数据列表
        """
        print(f"正在加载数据集: {self.annotations_file}")
        
        # 检查标注文件是否存在
        if not os.path.exists(self.annotations_file):
            raise FileNotFoundError(f"标注文件不存在: {self.annotations_file}")
        
        # 检查图片目录是否存在
        if not os.path.exists(self.images_dir):
            print(f"警告: 图片目录不存在: {self.images_dir}")
            print("请确保已下载并解压图片数据到该目录")
        
        # 加载 JSON 文件
        with open(self.annotations_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"数据集加载成功，共 {self.__len__()} 个样本")
        return self.data
    
    def get_sample(self, index: int) -> Dict[str, Any]:
        """
        获取指定索引的样本
        
        Args:
            index: 样本索引
            
        Returns:
            样本数据
        """
        if index < 0 or index >= len(self.data):
            raise IndexError(f"样本索引超出范围: {index}")
        
        sample = self.data[index]
        # 添加完整的图片路径
        if 'image' in sample:
            image_filename = sample['image']
            sample['image_path'] = os.path.join(self.images_dir, image_filename)
            sample['image_exists'] = os.path.exists(sample['image_path'])
        
        return sample
    
    def _build_conversation_text(self, conversations: List[Dict[str, str]]) -> str:
        """
        将对话转换为单个文本字符串
        
        Args:
            conversations: 对话列表
            
        Returns:
            构建的文本字符串
        """
        texts = []
        for conv in conversations:
            role = conv['from']
            value = conv['value']
            # 根据LLaVA格式调整角色标识
            if role == 'human':
                texts.append(f'<|im_start|>user\n{value}<|im_end|>')
            elif role == 'gpt':
                texts.append(f'<|im_start|>assistant\n{value}<|im_end|>')
        return ''.join(texts)

    def get_image_path(self, image_filename: str) -> str:
        """
        获取图片的完整路径
        
        Args:
            image_filename: 图片文件名
            
        Returns:
            图片的完整路径
        """
        return os.path.join(self.images_dir, image_filename)
    
    def download_images(self, sample_size: int = 1000) -> None:
        """
        按需下载指定数量的图片
        
        Args:
            sample_size: 要下载的样本数量
        """
        # 确保数据已加载
        if not self.data:
            self.load()
        
        # 创建图片目录
        os.makedirs(self.images_dir, exist_ok=True)
        
        # 选择要下载的子集
        subset = self.data[:sample_size]
        
        # COCO 图片下载基础 URL
        base_url = "http://images.cocodataset.org/train2017/"
        
        print(f"开始按需下载 {sample_size} 张图片...")
        
        downloaded = 0
        failed = 0
        
        for item in tqdm(subset):
            img_name = item.get('image')
            if not img_name:
                continue
            
            local_path = os.path.join(self.images_dir, img_name)
            
            # 如果本地没有再下载
            if not os.path.exists(local_path):
                img_url = base_url + img_name
                try:
                    img_data = requests.get(img_url, timeout=10).content
                    with open(local_path, 'wb') as handler:
                        handler.write(img_data)
                    downloaded += 1
                except Exception as e:
                    print(f"\n下载失败 {img_name}: {e}")
                    failed += 1
        
        print(f"\n下载完成！")
        print(f"成功下载: {downloaded} 张")
        print(f"下载失败: {failed} 张")
        print(f"本地已存在: {sample_size - downloaded - failed} 张")

# 使用示例
if __name__ == "__main__":
    # 创建数据集加载器
    dataset = LLaVADataset()
    
    # 加载数据
    data = dataset.load()
    
    # 按需下载图片（例如只下载前 1000 张）
    dataset.download_images(sample_size=1000)
    
    # 查看第一个样本
    if data:
        sample = dataset.get_sample(0)
        print("\n第一个样本:")
        print(f"ID: {sample['id']}")
        print(f"图片文件名: {sample['image']}")
        print(f"图片路径: {sample.get('image_path', 'N/A')}")
        print(f"图片存在: {sample.get('image_exists', False)}")
        print(f"对话数量: {len(sample['conversations'])}")
        print(f"第一个问题: {sample['conversations'][0]['value']}")
        print(f"第一个回答: {sample['conversations'][1]['value']}")
    
    # 测试获取训练数据项
    if len(dataset) > 0:
        training_item = dataset[0]
        print(f"\n训练数据项:")
        print(f"input_ids shape: {training_item['input_ids'].shape}")
        print(f"pixel_values shape: {training_item['pixel_values'].shape}")
        print(f"labels shape: {training_item['labels'].shape}")
    
    print("\n数据集准备完成！")
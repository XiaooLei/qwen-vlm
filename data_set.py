import os
import json
import requests
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLaVADataset(Dataset):
    """LLaVA 数据集加载器"""
    
    def __init__(self, data_dir: str = "./llava_data", is_train: bool = True, 
                 llm_name: str = "Qwen/Qwen2.5-0.5B", 
                 vision_name: str = "openai/clip-vit-base-patch16", 
                 sample_size: Optional[int] = None):
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
        pixel_values = self._process_image(sample)
        input_ids, attention_mask, labels = self._process_text(sample)
        
        return {
            'input_ids': input_ids,
            'pixel_values': pixel_values,
            'labels': labels,
            'attention_mask': attention_mask
        }
    
    def _process_image(self, sample: Dict[str, Any]) -> torch.Tensor:
        """
        处理图像，确保图像存在并返回处理后的像素值
        
        Args:
            sample: 样本数据
            
        Returns:
            处理后的像素值张量
        """
        image_path = sample.get('image_path')
        if image_path:
            # 检查图片是否存在，如果不存在尝试下载
            if not os.path.exists(image_path):
                image_filename = os.path.basename(image_path)
                logger.info(f"图片不存在，尝试下载: {image_filename}")
                self._download_single_image(image_filename)
            
            # 尝试打开图片
            try:
                image = Image.open(image_path).convert('RGB')
                pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
                return pixel_values.to(torch.bfloat16)
            except Exception as e:
                # 如果图像无法打开（如损坏或格式不正确），返回零张量
                logger.warning(f"无法打开图片 {image_path}: {e}")
                return torch.zeros(3, 224, 224, dtype=torch.bfloat16)
        else:
            # 如果图像路径不存在，返回零张量
            return torch.zeros(3, 224, 224, dtype=torch.bfloat16)
    
    def _process_text(self, sample: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        处理文本，构建输入和标签
        
        Args:
            sample: 样本数据
            
        Returns:
            input_ids, attention_mask, labels
        """
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
        
        return input_ids, attention_mask, labels
        
    def load(self) -> List[Dict[str, Any]]:
        """
        加载数据集
        
        Returns:
            加载的数据列表
        """
        logger.info(f"正在加载数据集: {self.annotations_file}")
        
        # 检查标注文件是否存在
        if not os.path.exists(self.annotations_file):
            raise FileNotFoundError(f"标注文件不存在: {self.annotations_file}")
        
        # 检查图片目录是否存在
        if not os.path.exists(self.images_dir):
            logger.warning(f"图片目录不存在: {self.images_dir}")
            logger.info("将在需要时自动创建和下载")
        
        # 加载 JSON 文件
        with open(self.annotations_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        logger.info(f"数据集加载成功，共 {self.__len__()} 个样本")

        # 过滤没有<image>标签的样本
        self.data = [sample for sample in self.data if '<image>' in sample['conversations'][0]['value']]
        logger.info(f"过滤后数据集共 {self.__len__()} 个样本")

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
    
    def _download_single_image(self, image_filename: str) -> bool:
        """
        下载单个图片
        
        Args:
            image_filename: 图片文件名
            
        Returns:
            是否下载成功
        """
        local_path = os.path.join(self.images_dir, image_filename)
        
        # 如果本地已存在，直接返回成功
        if os.path.exists(local_path):
            return True
        
        # COCO 图片下载基础 URL
        base_url = "http://images.cocodataset.org/train2017/"
        img_url = base_url + image_filename
        
        try:
            # 确保图片目录存在
            os.makedirs(self.images_dir, exist_ok=True)
            # 下载图片
            img_data = requests.get(img_url, timeout=10).content
            with open(local_path, 'wb') as handler:
                handler.write(img_data)
            return True
        except Exception as e:
            logger.error(f"下载失败 {image_filename}: {e}")
            return False
    
    def ensure_sample_data_exists(self, sample_size: Optional[int] = None, max_workers: int = 10) -> Dict[str, int]:
        """
        确保当前需要的数据都存在，如果不存在就补充下载
        
        Args:
            sample_size: 要确保存在的样本数量，None表示使用当前的sample_size
            max_workers: 并行下载的最大线程数
            
        Returns:
            下载结果统计，包含成功、失败和已存在的数量
        """
        # 确保数据已加载
        if not self.data:
            self.load()
        
        # 确定要处理的样本数量
        if sample_size is None:
            sample_size = self.sample_size or len(self.data)
        
        # 创建图片目录
        os.makedirs(self.images_dir, exist_ok=True)
        
        # 收集需要检查的图片文件名
        image_filenames = self._get_sample_image_filenames(sample_size)
        logger.info(f"检查并确保 {len(image_filenames)} 张图片存在...")
        
        # 统计结果
        results = {
            'success': 0,  # 新下载成功
            'failed': 0,   # 下载失败
            'exists': 0    # 已存在
        }
        
        # 分离已存在和需要下载的图片
        existing, to_download = self._split_existing_and_missing(image_filenames)
        results['exists'] = len(existing)
        
        # 如果有需要下载的图片
        if to_download:
            logger.info(f"发现 {len(to_download)} 张图片不存在，开始下载...")
            download_results = self._download_images_parallel(to_download, max_workers)
            results['success'] = download_results['success']
            results['failed'] = download_results['failed']
        
        logger.info(f"检查完成！")
        logger.info(f"已存在: {results['exists']} 张")
        logger.info(f"成功下载: {results['success']} 张")
        logger.info(f"下载失败: {results['failed']} 张")
        
        return results
    
    def download_images(self, sample_size: int = 1000, max_workers: int = 10) -> Dict[str, int]:
        """
        按需下载指定数量的图片
        
        Args:
            sample_size: 要下载的样本数量
            max_workers: 并行下载的最大线程数
            
        Returns:
            下载结果统计，包含成功、失败和已存在的数量
        """
        # 确保数据已加载
        if not self.data:
            self.load()
        

        # 创建图片目录
        os.makedirs(self.images_dir, exist_ok=True)
        
        # 收集需要下载的图片文件名
        image_filenames = self._get_sample_image_filenames(sample_size)
        logger.info(f"开始按需下载 {len(image_filenames)} 张图片...")
        
        # 下载结果统计
        download_results = self._download_images_parallel(image_filenames, max_workers)
        existing_count = len(image_filenames) - download_results['success'] - download_results['failed']
        
        logger.info(f"下载完成！")
        logger.info(f"成功下载: {download_results['success']} 张")
        logger.info(f"下载失败: {download_results['failed']} 张")
        logger.info(f"本地已存在: {existing_count} 张")
        
        return {
            **download_results,
            'exists': existing_count
        }
    
    def _get_sample_image_filenames(self, sample_size: int) -> List[str]:
        """
        获取指定数量样本的图片文件名列表
        
        Args:
            sample_size: 样本数量
            
        Returns:
            图片文件名列表
        """
        subset = self.data[:sample_size]
        image_filenames = []
        for item in subset:
            img_name = item.get('image')
            if img_name:
                image_filenames.append(img_name)
        return image_filenames
    
    def _split_existing_and_missing(self, image_filenames: List[str]) -> tuple[List[str], List[str]]:
        """
        分离已存在和缺失的图片文件名
        
        Args:
            image_filenames: 图片文件名列表
            
        Returns:
            (已存在的图片文件名列表, 缺失的图片文件名列表)
        """
        existing = []
        missing = []
        
        for img_name in image_filenames:
            local_path = os.path.join(self.images_dir, img_name)
            if os.path.exists(local_path):
                existing.append(img_name)
            else:
                missing.append(img_name)
        
        return existing, missing
    
    def _download_images_parallel(self, image_filenames: List[str], max_workers: int = 10) -> Dict[str, int]:
        """
        并行下载多张图片
        
        Args:
            image_filenames: 要下载的图片文件名列表
            max_workers: 并行下载的最大线程数
            
        Returns:
            下载结果统计，包含成功和失败的数量
        """
        results = {
            'success': 0,
            'failed': 0
        }
        
        # 使用线程池并行下载
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有下载任务
            future_to_img = {executor.submit(self._download_single_image, img_name): img_name 
                           for img_name in image_filenames}
            
            # 处理完成的任务
            for future in tqdm(as_completed(future_to_img), total=len(image_filenames)):
                try:
                    success = future.result()
                    if success:
                        results['success'] += 1
                    else:
                        results['failed'] += 1
                except Exception as e:
                    img_name = future_to_img[future]
                    logger.error(f"处理图片 {img_name} 时出错: {e}")
                    results['failed'] += 1
        
        return results

# 使用示例
if __name__ == "__main__":
    # 创建数据集加载器
    dataset = LLaVADataset(sample_size=1000)
    
    # 加载数据
    data = dataset.load()
    
    # 确保数据存在
    result = dataset.ensure_sample_data_exists()
    logger.info(f"数据检查结果: {result}")
    
    # 查看第一个样本
    if data:
        sample = dataset.get_sample(0)
        logger.info("\n第一个样本:")
        logger.info(f"ID: {sample['id']}")
        logger.info(f"图片文件名: {sample['image']}")
        logger.info(f"图片路径: {sample.get('image_path', 'N/A')}")
        logger.info(f"图片存在: {sample.get('image_exists', False)}")
        logger.info(f"对话数量: {len(sample['conversations'])}")
        logger.info(f"第一个问题: {sample['conversations'][0]['value']}")
        logger.info(f"第一个回答: {sample['conversations'][1]['value']}")
    
    # 测试获取训练数据项
    if len(dataset) > 0:
        training_item = dataset[0]
        logger.info(f"\n训练数据项:")
        logger.info(f"input_ids shape: {training_item['input_ids'].shape}")
        logger.info(f"pixel_values shape: {training_item['pixel_values'].shape}")
        logger.info(f"labels shape: {training_item['labels'].shape}")
    
    logger.info("\n数据集准备完成！")
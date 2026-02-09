import os
import json
import requests
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import random

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
                 sample_size: Optional[int] = None,
                 chat_round: int = 5,
                 max_seq_len: int = 512):
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
        self.chat_round = chat_round
        self.max_seq_len = max_seq_len
        
        # 初始化tokenizer和image processor
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)

        # 【必须添加这行】确保 Dataset 编码出来的 ID 和模型找的 ID 一致
        self.tokenizer.add_tokens(["<image>"], special_tokens=True)

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
        # 1. 获取对话并构建完整文本
        conversations = sample['conversations']
        conversations = conversations[:self.chat_round*2]
        text = self._build_conversation_text(conversations)
        
        # 2. 编码文本为token IDs
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_seq_len,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # 3. 初始化标签（与输入相同）
        labels = input_ids.clone()
        
        # 4. 标记padding位置为-100（忽略）
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # 5. 遮挡不需要预测的标记部分
        # 编码需要查找的标记
        user_start_tokens = self.tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
        assistant_start_tokens = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        im_end_tokens = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
        
        # 转换为张量以便比较
        user_start_tensor = torch.tensor(user_start_tokens, device=input_ids.device)
        assistant_start_tensor = torch.tensor(assistant_start_tokens, device=input_ids.device)
        im_end_tensor = torch.tensor(im_end_tokens, device=input_ids.device)
        
        # 遍历输入IDs，处理所有标记部分
        idx = 0
        while idx < len(input_ids):
            # 检查是否找到用户部分的开始
            if (len(input_ids) - idx) >= len(user_start_tensor):
                if torch.equal(input_ids[idx:idx+len(user_start_tensor)], user_start_tensor):
                    # 找到用户部分开始，查找结束标记
                    end_idx = idx + len(user_start_tensor)
                    while (len(input_ids) - end_idx) >= len(im_end_tensor):
                        if torch.equal(input_ids[end_idx:end_idx+len(im_end_tensor)], im_end_tensor):
                            # 遮挡整个用户部分（包括标记）
                            labels[idx:end_idx+len(im_end_tensor)] = -100
                            idx = end_idx + len(im_end_tensor)
                            break
                        end_idx += 1
                    else:
                        idx += 1
                # 检查是否找到assistant部分的开始
                elif (len(input_ids) - idx) >= len(assistant_start_tensor):
                    if torch.equal(input_ids[idx:idx+len(assistant_start_tensor)], assistant_start_tensor):
                        # 找到assistant部分开始，遮挡开始标记
                        labels[idx:idx+len(assistant_start_tensor)] = -100
                        # 查找结束标记
                        end_idx = idx + len(assistant_start_tensor)
                        while (len(input_ids) - end_idx) >= len(im_end_tensor):
                            if torch.equal(input_ids[end_idx:end_idx+len(im_end_tensor)], im_end_tensor):
                                # 遮挡结束标记，保留中间内容
                                # labels[end_idx:end_idx+len(im_end_tensor)] = -100
                                idx = end_idx + len(im_end_tensor)
                                break
                            end_idx += 1
                        else:
                            idx += 1
                    else:
                        idx += 1
                else:
                    idx += 1
            else:
                break

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
                texts.append(f'<|im_start|>user\n{value}<|im_end|>\n')
            elif role == 'gpt':
                texts.append(f'<|im_start|>assistant\n{value}<|im_end|>\n')
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



def check_data_set():
    """
    统计输入到模型训练的上下文长度
    
    包括：token长度、图像特征后的实际长度、标签遮挡情况等
    """
    logger.info("="*60)
    logger.info("开始统计模型训练上下文长度...")
    logger.info("="*60)
    
    # 创建数据集加载器
    dataset = LLaVADataset(sample_size=10000)  # 使用全部数据
    
    # 加载数据
    data = dataset.load()
    
    # 1. 基本统计信息
    logger.info(f"\n【基本统计信息】")
    logger.info(f"总样本数: {len(data)}")
    logger.info(f"数据集大小: {len(dataset)}")
    logger.info(f"最大序列长度: {dataset.tokenizer.model_max_length}")
    
    # 2. 输入到模型的上下文长度统计
    logger.info(f"\n【模型训练上下文长度统计】")
    
    text_token_lengths = []  # 纯文本token长度（不包括padding）
    actual_context_lengths = []  # 实际上下文长度（包括图像特征）
    valid_label_lengths = []  # 有效标签长度
    masked_label_lengths = []  # 被遮挡的标签长度
    
    # 图像特征长度（CLIP ViT-Base: 576个patch tokens）
    image_feature_length = 196  # 24x24 patches
    
    sample_size = min(100, len(data))
    for i in range(sample_size):
        item = dataset[i]
        input_ids = item['input_ids']
        labels = item['labels']
        
        # 统计纯文本token长度（排除padding）
        text_length = (input_ids != dataset.tokenizer.pad_token_id).sum().item()
        text_token_lengths.append(text_length)
        
        # 计算实际上下文长度（文本token + 图像特征 - <image>占位符）
        # 查找<image> token的位置
        image_token_id = dataset.tokenizer.convert_tokens_to_ids("<image>")
        image_token_count = (input_ids == image_token_id).sum().item()
        
        # 实际上下文长度 = 文本长度 + 图像特征长度 - <image>占位符数量
        actual_context_length = text_length + (image_feature_length * image_token_count) - image_token_count
        actual_context_lengths.append(actual_context_length)
        
        # 统计有效标签和被遮挡的标签
        valid_count = (labels != -100).sum().item()
        masked_count = (labels == -100).sum().item()
        valid_label_lengths.append(valid_count)
        masked_label_lengths.append(masked_count)
    
    logger.info(f"纯文本token长度 - 平均: {sum(text_token_lengths) / len(text_token_lengths):.2f}, 最小: {min(text_token_lengths)}, 最大: {max(text_token_lengths)}")
    logger.info(f"实际上下文长度（含图像特征） - 平均: {sum(actual_context_lengths) / len(actual_context_lengths):.2f}, 最小: {min(actual_context_lengths)}, 最大: {max(actual_context_lengths)}")
    logger.info(f"有效标签长度 - 平均: {sum(valid_label_lengths) / len(valid_label_lengths):.2f}, 最小: {min(valid_label_lengths)}, 最大: {max(valid_label_lengths)}")
    logger.info(f"被遮挡标签长度 - 平均: {sum(masked_label_lengths) / len(masked_label_lengths):.2f}, 最小: {min(masked_label_lengths)}, 最大: {max(masked_label_lengths)}")
    
    # 计算遮挡比例
    total_labels = sum(valid_label_lengths) + sum(masked_label_lengths)
    if total_labels > 0:
        mask_ratio = sum(masked_label_lengths) / total_labels * 100
        logger.info(f"标签遮挡比例 - 平均: {mask_ratio:.2f}%")
    
    # 3. 长度分布统计
    logger.info(f"\n【长度分布统计】")
    from collections import Counter
    
    # 纯文本长度分布（按区间统计）
    text_length_bins = {
        '0-50': sum(1 for l in text_token_lengths if l <= 50),
        '51-100': sum(1 for l in text_token_lengths if 50 < l <= 100),
        '101-150': sum(1 for l in text_token_lengths if 100 < l <= 150),
        '151-200': sum(1 for l in text_token_lengths if 150 < l <= 200),
        '201-256': sum(1 for l in text_token_lengths if 200 < l <= 256),
        '256+': sum(1 for l in text_token_lengths if l > 256)
    }
    logger.info(f"纯文本长度分布: {text_length_bins}")
    
    # 实际上下文长度分布（按区间统计）
    context_length_bins = {
        '0-256': sum(1 for l in actual_context_lengths if l <= 256),
        '257-512': sum(1 for l in actual_context_lengths if 256 < l <= 512),
        '513-768': sum(1 for l in actual_context_lengths if 512 < l <= 768),
        '769+': sum(1 for l in actual_context_lengths if l > 768)
    }
    logger.info(f"实际上下文长度分布: {context_length_bins}")
    
    # 4. 样本示例
    logger.info(f"\n【样本示例】")
    if len(data) > 0:
        sample = dataset[0]
        input_ids = sample['input_ids']
        labels = sample['labels']
        
        # 计算第一个样本的详细信息
        text_length = (input_ids != dataset.tokenizer.pad_token_id).sum().item()
        image_token_id = dataset.tokenizer.convert_tokens_to_ids("<image>")
        image_token_count = (input_ids == image_token_id).sum().item()
        actual_context_length = text_length + (image_feature_length * image_token_count) - image_token_count
        valid_count = (labels != -100).sum().item()
        masked_count = (labels == -100).sum().item()
        
        logger.info(f"第一个样本:")
        logger.info(f"  纯文本token长度: {text_length}")
        logger.info(f"  <image> token数量: {image_token_count}")
        logger.info(f"  图像特征长度: {image_feature_length * image_token_count}")
        logger.info(f"  实际上下文长度（含图像特征）: {actual_context_length}")
        logger.info(f"  有效标签长度: {valid_count}")
        logger.info(f"  被遮挡标签长度: {masked_count}")
        logger.info(f"  遮挡比例: {masked_count / (valid_count + masked_count) * 100:.2f}%")
        
        # 解码并打印完整的labels（显示遮挡情况）
        decoded_labels = []
        for label_id in labels:
            if label_id == -100:
                decoded_labels.append("[MASKED]")
            else:
                try:
                    token = dataset.tokenizer.decode([label_id], skip_special_tokens=False)
                    decoded_labels.append(token)
                except:
                    decoded_labels.append(f"[ID:{label_id}]")
        
        logger.info(f"\n第一个样本的labels（前60个token）:")
        logger.info(" ".join(decoded_labels[:60]))
    
    logger.info("\n" + "="*60)
    logger.info("模型训练上下文长度统计完成！")
    logger.info("="*60)


def analyze_yes_no_bias(data_dir="./llava_data", sample_size=None, chat_round=2):
    """
    分析训练数据中一般疑问句的回答分布情况
    
    Args:
        data_dir: 数据目录
        sample_size: 分析样本数量，None表示全部
        chat_round: 对话轮数
    """
    import re
    
    logger.info("="*60)
    logger.info("开始分析一般疑问句回答分布...")
    logger.info("="*60)
    
    dataset = LLaVADataset(data_dir=data_dir, sample_size=sample_size, chat_round=chat_round)
    data = dataset.load()[:sample_size]
    
    total_samples = len(data)
    yes_no_questions = []
    
    yes_no_patterns = [
        r'\b(is|are|was|were|do|does|did|can|could|will|would|should|may|might|must|has|have|had)\b.*\?',
        r'\b(Is|Are|Was|Were|Do|Does|Did|Can|Could|Will|Would|Should|May|Might|Must|Has|Have|Had)\b.*\?',
        r'\b(yes|no)\b',
    ]
    
    yes_keywords = [r'\byes\b', r'\bYes\b', r'\bYES\b', r'\byeah\b', r'\bYeah\b', r'\byup\b', r'\bYup\b', r'\bcorrect\b', r'\bCorrect\b', r'\bright\b', r'\bRight\b', r'\btrue\b', r'\bTrue\b']
    no_keywords = [r'\bno\b', r'\bNo\b', r'\bNO\b', r'\bnah\b', r'\bNah\b', r'\bnope\b', r'\bNope\b', r'\bincorrect\b', r'\bIncorrect\b', r'\bwrong\b', r'\bWrong\b', r'\bfalse\b', r'\bFalse\b']
    
    for idx, sample in enumerate(data):
        conversations = sample.get('conversations', [])
        
        for i in range(0, len(conversations) - 1, 2):
            if i + 1 >= len(conversations):
                break
                
            user_msg = conversations[i].get('value', '')
            assistant_msg = conversations[i + 1].get('value', '')
            
            if conversations[i].get('from') != 'human':
                continue
            
            is_question = any(re.search(pattern, user_msg) for pattern in yes_no_patterns)
            
            if is_question:
                answer_lower = assistant_msg.lower().strip()
                
                if any(re.search(kw, answer_lower) for kw in yes_keywords):
                    answer_type = 'yes'
                elif any(re.search(kw, answer_lower) for kw in no_keywords):
                    answer_type = 'no'
                elif answer_lower.startswith(('yes', 'no', 'yeah', 'nah', 'yup', 'nope', 'correct', 'incorrect', 'right', 'wrong', 'true', 'false')):
                    answer_type = 'yes' if answer_lower.startswith(('yes', 'yeah', 'yup', 'correct', 'right', 'true')) else 'no'
                else:
                    answer_type = 'other'
                
                yes_no_questions.append({
                    'sample_idx': idx,
                    'question': user_msg,
                    'answer': assistant_msg,
                    'answer_type': answer_type,
                    'question_length': len(user_msg),
                    'answer_length': len(assistant_msg)
                })
    
    total_yes_no = len(yes_no_questions)
    yes_count = sum(1 for q in yes_no_questions if q['answer_type'] == 'yes')
    no_count = sum(1 for q in yes_no_questions if q['answer_type'] == 'no')
    other_count = sum(1 for q in yes_no_questions if q['answer_type'] == 'other')
    
    logger.info(f"\n【总体统计】")
    logger.info(f"总样本数: {total_samples}")
    logger.info(f"一般疑问句数量: {total_yes_no}")
    logger.info(f"一般疑问句占比: {total_yes_no/total_samples*100:.2f}%")
    
    logger.info(f"\n【回答类型分布】")
    logger.info(f"Yes 回答: {yes_count} ({yes_count/total_yes_no*100:.2f}%)")
    logger.info(f"No 回答: {no_count} ({no_count/total_yes_no*100:.2f}%)")
    logger.info(f"其他回答: {other_count} ({other_count/total_yes_no*100:.2f}%)")
    
    if total_yes_no > 0:
        logger.info(f"\n【Yes/No 比例】")
        logger.info(f"Yes:No = {yes_count}:{no_count} (比例 = {yes_count/no_count if no_count > 0 else float('inf'):.2f}:1)")
        
        if yes_count / total_yes_no > 0.7:
            logger.warning("⚠️  警告: Yes 回答比例过高 (>70%)，可能导致模型偏向回答 Yes!")
        elif yes_count / total_yes_no > 0.6:
            logger.warning("⚠️  注意: Yes 回答比例偏高 (>60%)")
    
    logger.info(f"\n【问题长度统计】")
    question_lengths = [q['question_length'] for q in yes_no_questions]
    logger.info(f"平均问题长度: {sum(question_lengths)/len(question_lengths):.1f} 字符")
    logger.info(f"最短问题: {min(question_lengths)} 字符")
    logger.info(f"最长问题: {max(question_lengths)} 字符")
    
    logger.info(f"\n【回答长度统计】")
    answer_lengths = [q['answer_length'] for q in yes_no_questions]
    logger.info(f"平均回答长度: {sum(answer_lengths)/len(answer_lengths):.1f} 字符")
    logger.info(f"最短回答: {min(answer_lengths)} 字符")
    logger.info(f"最长回答: {max(answer_lengths)} 字符")
    
    logger.info(f"\n【Yes 回答示例 (前5个)】")
    yes_examples = [q for q in yes_no_questions if q['answer_type'] == 'yes'][:5]
    for i, ex in enumerate(yes_examples, 1):
        logger.info(f"\n示例 {i}:")
        logger.info(f"  问题: {ex['question'][:100]}..." if len(ex['question']) > 100 else f"  问题: {ex['question']}")
        logger.info(f"  回答: {ex['answer'][:100]}..." if len(ex['answer']) > 100 else f"  回答: {ex['answer']}")
    
    logger.info(f"\n【No 回答示例 (前5个)】")
    no_examples = [q for q in yes_no_questions if q['answer_type'] == 'no'][:5]
    for i, ex in enumerate(no_examples, 1):
        logger.info(f"\n示例 {i}:")
        logger.info(f"  问题: {ex['question'][:100]}..." if len(ex['question']) > 100 else f"  问题: {ex['question']}")
        logger.info(f"  回答: {ex['answer'][:100]}..." if len(ex['answer']) > 100 else f"  回答: {ex['answer']}")
    
    logger.info(f"\n【其他回答示例 (前5个)】")
    other_examples = [q for q in yes_no_questions if q['answer_type'] == 'other'][:5]
    for i, ex in enumerate(other_examples, 1):
        logger.info(f"\n示例 {i}:")
        logger.info(f"  问题: {ex['question'][:100]}..." if len(ex['question']) > 100 else f"  问题: {ex['question']}")
        logger.info(f"  回答: {ex['answer'][:100]}..." if len(ex['answer']) > 100 else f"  回答: {ex['answer']}")
    
    logger.info("\n" + "="*60)
    logger.info("分析完成！")
    logger.info("="*60)
    
    return {
        'total_samples': total_samples,
        'total_yes_no': total_yes_no,
        'yes_count': yes_count,
        'no_count': no_count,
        'other_count': other_count,
        'yes_ratio': yes_count / total_yes_no if total_yes_no > 0 else 0,
        'no_ratio': no_count / total_yes_no if total_yes_no > 0 else 0
    }


class BalancedSampler(Sampler):
    """
    平衡采样器，用于均衡 yes/no 样本的采样
    
    在训练时，no 样本重复 5 次，yes 和其他样本保持不变
    """
    
    def __init__(self, dataset: LLaVADataset, seed: int = 42):
        """
        初始化平衡采样器
        
        Args:
            dataset: LLaVADataset 数据集
            seed: 随机种子
        """
        self.dataset = dataset
        self.seed = seed
        random.seed(seed)
        
        # 分类样本索引
        self.yes_indices = []
        self.no_indices = []
        self.other_indices = []
        
        # 正则表达式模式
        self.yes_no_patterns = [
            r'\b(is|are|was|were|do|does|did|can|could|will|would|should|may|might|must|has|have|had)\b.*\?',
            r'\b(Is|Are|Was|Were|Do|Does|Did|Can|Could|Will|Would|Should|May|Might|Must|Has|Have|Had)\b.*\?',
        ]
        self.yes_keywords = [r'\byes\b', r'\bYes\b', r'\bYES\b', r'\byeah\b', r'\bYeah\b', r'\byup\b', r'\bYup\b', r'\bcorrect\b', r'\bCorrect\b', r'\bright\b', r'\bRight\b', r'\btrue\b', r'\bTrue\b']
        self.no_keywords = [r'\bno\b', r'\bNo\b', r'\bNO\b', r'\bnah\b', r'\bNah\b', r'\bnope\b', r'\bNope\b', r'\bincorrect\b', r'\bIncorrect\b', r'\bwrong\b', r'\bWrong\b', r'\bfalse\b', r'\bFalse\b']
        
        # 分析数据集
        self._analyze_dataset()
    
    def _analyze_dataset(self):
        """分析数据集，将样本分类"""
        logger.info("正在分析数据集以平衡 yes/no 样本...")
        
        for idx in range(len(self.dataset)):
            sample = self.dataset.data[idx]
            conversations = sample.get('conversations', [])
            
            # 检查对话中是否有 yes/no 问题
            for i in range(0, len(conversations) - 1, 2):
                if i + 1 >= len(conversations):
                    break
                    
                user_msg = conversations[i].get('value', '')
                assistant_msg = conversations[i + 1].get('value', '')
                
                if conversations[i].get('from') != 'human':
                    continue
                
                is_question = any(re.search(pattern, user_msg) for pattern in self.yes_no_patterns)
                
                if is_question:
                    answer_lower = assistant_msg.lower().strip()
                    
                    if any(re.search(kw, answer_lower) for kw in self.yes_keywords):
                        self.yes_indices.append(idx)
                        break
                    elif any(re.search(kw, answer_lower) for kw in self.no_keywords):
                        self.no_indices.append(idx)
                        break
            else:
                # 如果没有 yes/no 问题，归为其他类
                self.other_indices.append(idx)
        
        logger.info(f"数据集分析完成:")
        logger.info(f"  Yes 样本: {len(self.yes_indices)}")
        logger.info(f"  No 样本: {len(self.no_indices)} (将重复5倍)")
        logger.info(f"  其他样本: {len(self.other_indices)}")
        
        # 打乱索引
        random.shuffle(self.yes_indices)
        random.shuffle(self.no_indices)
        random.shuffle(self.other_indices)
    
    def __iter__(self):
        """生成采样索引"""
        # no 样本重复 5 次
        repeated_no = self.no_indices * 5
        
        # 合并所有样本
        all_indices = self.yes_indices + repeated_no + self.other_indices
        
        # 打乱
        random.shuffle(all_indices)
        
        # 迭代生成索引
        for idx in all_indices:
            yield idx
    
    def __len__(self):
        """返回采样器长度"""
        return len(self.yes_indices) + len(self.no_indices) * 5 + len(self.other_indices)


def create_balanced_dataloader(dataset: LLaVADataset, batch_size: int = 2, 
                                num_workers: int = 2,
                                seed: int = 42, shuffle: bool = True):
    """
    创建平衡采样器的 DataLoader
    
    Args:
        dataset: LLaVADataset 数据集
        batch_size: 批次大小
        num_workers: 数据加载线程数
        seed: 随机种子
        shuffle: 是否打乱数据
    
    Returns:
        DataLoader 实例
    """
    from torch.utils.data import DataLoader
    
    if shuffle:
        sampler = BalancedSampler(dataset, seed=seed)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    return dataloader



# 使用示例
if __name__ == "__main__":
    # # 创建数据集加载器
    analyze_yes_no_bias(sample_size=30000)
    # dataset = LLaVADataset(sample_size=1000)
    
    # # 加载数据
    # data = dataset.load()
    
    # # 确保数据存在
    # result = dataset.ensure_sample_data_exists()
    # logger.info(f"数据检查结果: {result}")
    
    # # 查看第一个样本
    # if data:
    #     sample = dataset.get_sample(0)
    #     logger.info("\n第一个样本:")
    #     logger.info(f"ID: {sample['id']}")
    #     logger.info(f"图片文件名: {sample['image']}")
    #     logger.info(f"图片路径: {sample.get('image_path', 'N/A')}")
    #     logger.info(f"图片存在: {sample.get('image_exists', False)}")
    #     logger.info(f"对话数量: {len(sample['conversations'])}")
    #     logger.info(f"第一个问题: {sample['conversations'][0]['value']}")
    #     logger.info(f"第一个回答: {sample['conversations'][1]['value']}")
    
    # # 测试获取训练数据项
    # if len(dataset) > 0:
    #     training_item = dataset[0]
    #     logger.info(f"\n训练数据项:")
    #     logger.info(f"input_ids shape: {training_item['input_ids'].shape}")
    #     logger.info(f"pixel_values shape: {training_item['pixel_values'].shape}")
    #     logger.info(f"labels shape: {training_item['labels'].shape}")
        
    #     # 解码并打印完整的labels（包括被遮挡的部分）
    #     labels = training_item['labels']
        
    #     # 创建一个可解码的标签副本，将-100替换为一个特殊标记
    #     decoded_labels = []
    #     for label_id in labels:
    #         if label_id == -100:
    #             decoded_labels.append("[MASKED]")
    #         else:
    #             try:
    #                 token = dataset.tokenizer.decode([label_id], skip_special_tokens=False)
    #                 decoded_labels.append(token)
    #             except:
    #                 decoded_labels.append(f"[ID:{label_id}]")
        
    #     # 打印标签内容，显示遮挡情况
    #     logger.info("完整labels（包括遮挡部分）:")
    #     logger.info(" ".join(decoded_labels[:512]))  # 只打印前50个token，避免输出过长
        
    #     # 同时打印原始input_ids的解码结果用于对比
    #     input_ids = training_item['input_ids']
    #     decoded_inputs = dataset.tokenizer.decode(input_ids, skip_special_tokens=False)
    #     logger.info(f"\n原始输入文本:")
    #     logger.info(decoded_inputs[:512] + "..." if len(decoded_inputs) > 300 else decoded_inputs)
    # logger.info("\n数据集准备完成！")

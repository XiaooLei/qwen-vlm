
"""
VLM 视觉语言模型训练脚本
适配 LLaVA 格式数据集
"""

from model import VLMModel
from data_set import LLaVADataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
import logging
import os
from datetime import datetime
from tqdm import tqdm
import argparse
from torch.cuda.amp import autocast, GradScaler
import signal, sys


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)




def handle_interrupt(sig, frame):
    """处理 Ctrl+C 中断"""
    logger.info("检测到 Ctrl+C 中断信号")
    global interrupted
    interrupted = True


def extract_model_name(model_name):
    """从模型名称中提取 LLM 和 Vision 模型名称"""

    # Qwen/Qwen2.5-0.5B-Instruct -> qwen2.5_0.5b_instruct
    
    # openai/clip-vit-base-patch32 -> openai_clip_vit_base_patch32
    # 还要去掉前缀 openai_
    model_name = model_name.replace("Qwen/", "")
    model_name  = os.path.basename(model_name)
    model_name = model_name.replace("/", "_").lower()
    return model_name


scaler = GradScaler() # 1. 初始化缩放器
def train_one_epoch(model, train_dataloader, optimizer, scheduler, device, epoch, grad_accum_steps=4):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    num_batches = len(train_dataloader)
    
    optimizer.zero_grad()
    
    progress_bar = tqdm(
        enumerate(train_dataloader), 
        total=num_batches, 
        desc=f"Epoch {epoch}",
        ncols=120
    )
    
    for batch_idx, batch in progress_bar:

        input_ids = batch["input_ids"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # 简化：如果是 Mac (MPS) 或 CPU，去掉 autocast 和 scaler
        if device == "cuda":
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            # Mac / CPU 路径
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss / grad_accum_steps
            # 打印调试：如果这里是 None，说明 forward 内部还是断了
            if batch_idx == 0: 
                print(f"DEBUG: loss grad_fn = {loss.grad_fn}")
            loss.backward()

        # 每累积 grad_accum_steps 步更新一次参数
        if (batch_idx + 1) % grad_accum_steps == 0:
            # 1. 直接进行 unscale_（只做一次，不要放进 try 里）
            # 这是为了让 clip_grad_norm_ 能看到正确的梯度值
            # 1. 必须先 unscale 才能裁剪
            scaler.unscale_(optimizer)
            
            # 2. 裁剪梯度（防止 NaN 的第二道防线）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 3. 更新权重
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad()
            scheduler.step()
        
        current_loss = loss.item() * grad_accum_steps
        total_loss += current_loss
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
        
        # 每 100 个 batch 记录日志
        if batch_idx % 100 == 0:
            logger.info(f"Epoch {epoch} - Batch {batch_idx}/{num_batches} - Loss: {current_loss:.4f}")
    
    # 处理剩余的梯度
    if (batch_idx + 1) % grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / num_batches
    logger.info(f"Epoch {epoch} completed - Average Training Loss: {avg_loss:.4f}")
    
    return avg_loss


def evaluate(model, val_dataloader, device, epoch):
    """验证模型"""
    model.eval()
    total_loss = 0
    num_batches = len(val_dataloader)
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"Evaluating Epoch {epoch}", ncols=120):
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()
    
    avg_loss = total_loss / num_batches
    logger.info(f"Epoch {epoch} - Validation Loss: {avg_loss:.4f}")
    
    return avg_loss


from torch.cuda.amp import autocast, GradScaler

def train_model(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    device,
    num_epochs=3,
    checkpoint_dir="./checkpoints",
    config=None
):
    """完整的训练流程"""
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    logger.info("=" * 60)
    logger.info("Starting Training...")
    logger.info(f"Training samples: {len(train_dataloader.dataset)}")
    logger.info(f"Validation samples: {len(val_dataloader.dataset)}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info("=" * 60)
    
    for epoch in range(num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        logger.info(f"{'='*60}")
        
        # 训练
        train_loss = train_one_epoch(
            model, train_dataloader, optimizer, scheduler, device, epoch
        )
        train_losses.append(train_loss)
        
        # 验证
        if val_dataloader is not None:
            val_loss = evaluate(model, val_dataloader, device, epoch)
            val_losses.append(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 同时保存最佳模型
                best_model_path = os.path.join(checkpoint_dir, f"projector_best_{extract_model_name(config['llm_name'])}_{extract_model_name(config['vision_name'])}.pt")
                torch.save(model.projector.state_dict(), best_model_path)
                logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
        
        torch.save(model.projector.state_dict(), os.path.join(checkpoint_dir, f"projector_epoch{epoch}_{extract_model_name(config['llm_name'])}_{extract_model_name(config['vision_name'])}.pt"))    
        # 每个 epoch 保存检查点    
    # 保存最终模型
    final_model_path = os.path.join(checkpoint_dir, f"projector_final_{extract_model_name(config['llm_name'])}_{extract_model_name(config['vision_name'])}.pt")
    torch.save(model.projector.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # 训练总结
    logger.info("\n" + "=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    logger.info(f"Final Training Loss: {train_losses[-1]:.4f}")
    if val_losses:
        logger.info(f"Best Validation Loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")
    
    return model


def main():
    # 配置参数
    parser = argparse.ArgumentParser(description="Train VLMPretrainedModel")
    parser.add_argument("--data_dir", type=str, default="./llava_data", help="Path to the dataset directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--sample_size", type=int, default=20000, help="Sample size for training")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="LLM model name")
    parser.add_argument("--vision_name", type=str, default="openai/clip-vit-base-patch16", help="Vision model name")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()

    config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_ratio': args.warmup_ratio,
        'llm_name': args.llm_name,
        'vision_name': args.vision_name,
        'grad_accum_steps': 4,
        'checkpoint_dir': args.checkpoint_dir,
        'sample_size': args.sample_size
    }
    
    # 设备配置
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    # 临时：使用 CPU 训练
    # device = "cpu"

    logger.info(f"Using device: {device}")
    
    # 初始化模型
    logger.info("Loading model...")

    projector_path = os.path.join(config['checkpoint_dir'], f"projector_best_{extract_model_name(config['llm_name'])}_{extract_model_name(config['vision_name'])}.pt")
    if os.path.exists(projector_path):
        projector_params = torch.load(projector_path)
    else:
        projector_params = None

    model = VLMModel(llm_name=config['llm_name'], vision_name=config['vision_name'], projector_params=projector_params)
    model = model.to(device)


    # 这两行一定要紧跟在初始化 model 之后，否则索引不到 <image> 会在 forward 报错！
    model.tokenizer.add_tokens(["<image>"], special_tokens=True)
    model.language_model.resize_token_embeddings(len(model.tokenizer))


    print(f"DEBUG: <image> token id = {model.tokenizer.convert_tokens_to_ids('<image>')}")
    test_text = "核心测试 <image> 结束"
    test_ids = model.tokenizer.encode(test_text)
    print(f"DEBUG: 编码测试 '{test_text}' -> {test_ids}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # 加载数据集
    logger.info("Loading datasets...")
    train_dataset = LLaVADataset(
        data_dir=config['data_dir'],
        is_train=True,
        sample_size=config['sample_size']
    )
    train_dataset.load()
    train_dataset.ensure_sample_data_exists()
    
    val_dataset = LLaVADataset(
        data_dir=config['data_dir'],
        is_train=False,
        sample_size=100
    )
    val_dataset.load()
    
    # 创建 DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    logger.info(f"Train batches per epoch: {len(train_dataloader)}")
    logger.info(f"Val batches per epoch: {len(val_dataloader)}")
    

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # 优化器
    optimizer = optim.AdamW(
        trainable_params,
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度器
    total_steps = len(train_dataloader) * config['num_epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    
    scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=total_steps
    )
    
    # 设置中断处理
    def save_on_interrupt(signum, frame):
        logger.info("\n接收到中断信号，正在保存最新权重...")
        interrupt_checkpoint_path = os.path.join(config['checkpoint_dir'], f"projector_interrupt_{extract_model_name(config['llm_name'])}_{extract_model_name(config['vision_name'])}.pt")
        torch.save(model.projector.state_dict(), interrupt_checkpoint_path)
        logger.info(f"中断时权重已保存到: {interrupt_checkpoint_path}")
        exit(0)
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, save_on_interrupt)
    signal.signal(signal.SIGTERM, save_on_interrupt)
    
    # 开始训练
    trained_model = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=config['num_epochs'],
        checkpoint_dir=config['checkpoint_dir'],
        config=config
    )
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
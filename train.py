
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



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
        
        # 开启自动混合精度上下文
        with torch.amp.autocast('cuda', dtype=torch.float16): # 这里才是真正触发 T4 加速的地方
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            # 梯度累积
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()      

        # 每累积 grad_accum_steps 步更新一次参数
        if (batch_idx + 1) % grad_accum_steps == 0:
            try:
                # 4. 梯度裁剪（非常重要，防止 Loss 爆炸）
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            except ValueError:
                # 如果它报错说“已经是 FP16”或“不需要 unscale”，就直接进行下一步
                pass
            
            # B. 使用 scaler.step 而不是 optimizer.step
            scaler.step(optimizer)
            
            # C. 更新缩放因子
            scaler.update()
            
            # D. 清空梯度
            optimizer.zero_grad()
            
            # E. 如果有 scheduler，通常在这里更新
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


def save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_dir):
    """保存检查点"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    
    torch.save({
        'epoch': epoch,
        'projector_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    logger.info(f"Checkpoint saved to {checkpoint_path}")



from torch.cuda.amp import autocast, GradScaler

def train_model(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    device,
    num_epochs=3,
    checkpoint_dir="./checkpoints"
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
        
        # 更新学习率
        scheduler.step()
        
        # 验证
        if val_dataloader is not None:
            val_loss = evaluate(model, val_dataloader, device, epoch)
            val_losses.append(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model.projector, optimizer, scheduler, epoch, val_loss, checkpoint_dir)
                # 同时保存最佳模型
                best_model_path = os.path.join(checkpoint_dir, "projector.pt")
                torch.save(model.projector.state_dict(), best_model_path)
                logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
        
        # 每个 epoch 保存检查点
        save_checkpoint(model.projector, optimizer, scheduler, epoch, train_loss, checkpoint_dir)
    
    # 保存最终模型
    final_model_path = os.path.join(checkpoint_dir, "projector.pt")
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
    
    logger.info(f"Using device: {device}")
    
    # 初始化模型
    logger.info("Loading model...")


    projector_path = os.path.join(config['checkpoint_dir'], "projector.pt")
    if os.path.exists(projector_path):
        projector_params = torch.load(projector_path)
    else:
        projector_params = None

    model = VLMModel(projector_params=projector_params)
    model = model.to(device)
    
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
        num_workers=0,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    logger.info(f"Train batches per epoch: {len(train_dataloader)}")
    logger.info(f"Val batches per epoch: {len(val_dataloader)}")
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
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
    
    # 开始训练
    trained_model = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=config['num_epochs'],
        checkpoint_dir=config['checkpoint_dir']
    )
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
import torch
from PIL import Image
import os
from model import VLMModel  # 确保你的类定义在 model.py 中

def run_test():
    # --- 1. 配置参数 ---
    # 指向你明早跑出来的最强权重
    checkpoint_path = "./checkpoints/projector.pt" 
    # 测试图片路径
    test_image_path = "./llava_data/train2017/000000000081.jpg" 
    
    # 自动选择设备
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"检测到设备: {device}")

    # --- 2. 初始化模型并加载权重 ---
    print("正在加载模型...")
    # 注意：初始化时先不传 projector_params，手动 load 更清晰
    model = VLMModel() 
    
    if os.path.exists(checkpoint_path):
        print(f"正在加载训练成果: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.projector.load_state_dict(state_dict)
    else:
        print(f"⚠️ 警告: 未找到权重文件 {checkpoint_path}，将使用随机初始化的 Projector 进行测试。")

    model.to(device)
    model.eval()

    # --- 3. 准备测试图片 ---
    if not os.path.exists(test_image_path):
        print(f"❌ 错误: 找不到测试图片 {test_image_path}")
        return

    image = Image.open(test_image_path).convert("RGB")
    print(f"成功加载图片: {test_image_path}")

    # --- 4. 开始提问 ---
    test_questions = [
        "What is in this image?",
        "Describe the cat and its environment.",
        "What color is the object in the picture?"
    ]

    print("\n" + "="*30)
    print("🚀 VLM 推理测试开始")
    print("="*30)

    for i, q in enumerate(test_questions):
        print(f"\n[问题 {i+1}]: {q}")
        try:
            # 直接调用你集成在类里的 answer 方法
            response = model.answer(image, q, max_new_tokens=128)
            print(f"AI 回复: {response}")
        except Exception as e:
            print(f"❌ 推理出错: {e}")

    print("\n" + "="*30)
    print("测试完成！")

if __name__ == "__main__":
    run_test()
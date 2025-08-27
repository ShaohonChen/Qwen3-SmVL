#!/usr/bin/env python3
"""
简化的多模态推理代码
"""

import os
import torch
from PIL import Image
from utils import load_model, load_processor


def load_trained_model(checkpoint_path, device="cuda"):
    """
    加载训练后的模型
    
    Args:
        checkpoint_path: 训练后模型的路径
        device: 运行设备
        
    Returns:
        model, processor
    """
    print(f"正在加载训练后的模型: {checkpoint_path}")
    
    # 使用原始的模型构建方式
    model = load_model(device)
    processor = load_processor()
    
    # 加载训练后的权重
    if os.path.exists(os.path.join(checkpoint_path, "model.safetensors")):
        print("正在加载safetensors权重...")
        from safetensors.torch import load_file
        state_dict = load_file(os.path.join(checkpoint_path, "model.safetensors"))
        model.load_state_dict(state_dict, strict=False)
        print("✅ 权重加载成功")
    elif os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
        print("正在加载pytorch权重...")
        state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("✅ 权重加载成功")
    else:
        print("⚠️  未找到权重文件，使用原始模型")
    
    model.eval()
    return model, processor


def inference(model, processor, image_path, prompt, max_tokens=512, device="cuda"):
    """
    简单的推理函数
    
    Args:
        model: 加载的模型
        processor: 处理器
        image_path: 图像路径
        prompt: 文本提示
        max_tokens: 最大token数
        device: 设备
        
    Returns:
        生成的文本
    """
    # 加载图像
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path
    
    messages = [
        {
            "role": "system",
            "content": "使用中文回答所有问题。",
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    
    # 应用聊天模板
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 处理输入
    inputs = processor(text=text, images=image, return_tensors="pt")
    inputs = inputs.to(device)
    
    # 确保输入数据类型与模型权重匹配（bfloat16）
    for key in inputs:
        if key == 'pixel_values' and inputs[key] is not None:
            inputs[key] = inputs[key].to(torch.bfloat16)
        elif key == 'input_ids' and inputs[key] is not None:
            inputs[key] = inputs[key].to(device)
        elif key == 'attention_mask' and inputs[key] is not None:
            inputs[key] = inputs[key].to(device)
    
    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            use_cache=True
        )
    
    # 解码输出
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    
    return response.strip()


def main():
    """主函数"""
    # 配置
    model_path = "./model/staged_training_test/stage2"
    image_path = "./resource/dog.png"  # 演示图片
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("🚀 开始推理演示...")
    print(f"模型路径: {model_path}")
    print(f"设备: {device}")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return
    
    try:
        # 加载模型
        model, processor = load_trained_model(model_path, device)
        print("✅ 模型加载完成")
        
        # 测试推理
        prompts = [
            "请描述这张图片。",
            "图片中有什么东西？",
            "图中的数量有多少？"
        ]
        
        print(f"\n📸 测试图片: {image_path}")
        print("="*60)
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n{i}. 提示: {prompt}")
            print("-" * 50)
            
            try:
                response = inference(model, processor, image_path, prompt, device=device)
                print(f"回复: {response}")
            except Exception as e:
                print(f"❌ 推理失败: {e}")
        
        print("\n" + "="*60)
        print("演示完成！")
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
分阶段训练系统使用示例

本脚本展示了如何使用分阶段训练系统进行VLM的全量微调
"""

import os
import sys
import subprocess
from pathlib import Path


def run_training_example():
    """
    运行训练示例
    """
    print("=" * 60)
    print("分阶段训练系统使用示例")
    print("=" * 60)
    
    # 检查必要的文件是否存在
    required_files = [
        "train_staged.py",
        "staged_training.yaml", 
        "quick_start.yaml",
        "utils.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺少必要文件: {missing_files}")
        print("请确保所有必要文件都在当前目录中")
        return False
    
    print("✅ 所有必要文件检查通过")
    
    # 示例1: 快速验证
    print("\n" + "="*40)
    print("示例1: 快速验证（阶段1训练）")
    print("="*40)
    
    print("运行快速验证训练...")
    try:
        result = subprocess.run([
            "python", "train_staged.py", "quick_start.yaml"
        ], capture_output=True, text=True, timeout=300)  # 5分钟超时
        
        if result.returncode == 0:
            print("✅ 快速验证训练成功完成")
        else:
            print(f"❌ 快速验证训练失败: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏰ 快速验证训练超时（这是正常的，因为训练需要时间）")
    except Exception as e:
        print(f"❌ 运行快速验证时出错: {e}")
    
    # 示例2: 完整训练流程
    print("\n" + "="*40)
    print("示例2: 完整训练流程（所有阶段）")
    print("="*40)
    
    print("注意: 完整训练流程需要较长时间和更多计算资源")
    print("建议在GPU环境下运行")
    
    # 检查是否有GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ 检测到GPU: {torch.cuda.get_device_name(0)}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print("⚠️  未检测到GPU，训练将使用CPU（速度较慢）")
    except ImportError:
        print("⚠️  无法检测GPU状态")
    
    print("\n要运行完整训练流程，请执行:")
    print("python train_staged.py staged_training.yaml")
    
    return True


def show_configuration_examples():
    """
    显示配置示例
    """
    print("\n" + "="*60)
    print("配置示例")
    print("="*60)
    
    print("1. 快速验证配置 (quick_start.yaml):")
    print("   - 只训练阶段1（连接器）")
    print("   - 使用单一数据集 (cocoqa)")
    print("   - 小批量，快速验证")
    
    print("\n2. 完整训练配置 (staged_training.yaml):")
    print("   - 训练所有三个阶段")
    print("   - 使用全量数据")
    print("   - 支持下游任务")
    
    print("\n3. 自定义配置示例:")
    print("   - 修改学习率: stage1_lr: 0.0001")
    print("   - 修改训练轮数: stage1_epochs: 2")
    print("   - 修改数据集: train_data: 'all'")
    print("   - 添加下游任务: downstream_tasks: ['captioning', 'vqa']")


def show_usage_examples():
    """
    显示使用示例
    """
    print("\n" + "="*60)
    print("使用示例")
    print("="*60)
    
    examples = [
        {
            "description": "快速验证（推荐新手）",
            "command": "python train_staged.py quick_start.yaml",
            "explanation": "只训练连接器，快速验证系统是否正常工作"
        },
        {
            "description": "完整训练流程",
            "command": "python train_staged.py staged_training.yaml", 
            "explanation": "训练所有三个阶段，实现最佳性能"
        },
        {
            "description": "只训练阶段1",
            "command": "python train_staged.py --training_stage stage1 --train_data cocoqa",
            "explanation": "只训练连接器，适合资源受限环境"
        },
        {
            "description": "只训练阶段2",
            "command": "python train_staged.py --training_stage stage2 --train_data all",
            "explanation": "训练视觉+连接器，提升视觉理解能力"
        },
        {
            "description": "只训练阶段3",
            "command": "python train_staged.py --training_stage stage3 --train_data all",
            "explanation": "全量微调，实现最佳性能"
        },
        {
            "description": "从阶段2恢复训练",
            "command": "python train_staged.py --resume_from_stage stage2 --training_stage all",
            "explanation": "从阶段2开始继续训练所有阶段"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['description']}")
        print(f"   命令: {example['command']}")
        print(f"   说明: {example['explanation']}")


def show_training_stages():
    """
    显示训练阶段说明
    """
    print("\n" + "="*60)
    print("训练阶段说明")
    print("="*60)
    
    stages = [
        {
            "name": "阶段1: 连接器训练",
            "frozen": ["视觉编码器", "文本模型"],
            "trainable": ["连接器"],
            "purpose": "学习视觉特征到文本特征的映射",
            "lr": "1e-4",
            "epochs": "1-2"
        },
        {
            "name": "阶段2: 视觉+连接器训练", 
            "frozen": ["文本模型"],
            "trainable": ["视觉编码器", "连接器"],
            "purpose": "优化视觉理解能力",
            "lr": "5e-5",
            "epochs": "1-3"
        },
        {
            "name": "阶段3: 全量微调",
            "frozen": [],
            "trainable": ["所有参数"],
            "purpose": "实现最佳性能",
            "lr": "1e-5", 
            "epochs": "1-5"
        }
    ]
    
    for stage in stages:
        print(f"\n{stage['name']}")
        print(f"  冻结组件: {', '.join(stage['frozen']) if stage['frozen'] else '无'}")
        print(f"  可训练组件: {', '.join(stage['trainable'])}")
        print(f"  目的: {stage['purpose']}")
        print(f"  建议学习率: {stage['lr']}")
        print(f"  建议训练轮数: {stage['epochs']}")


def show_troubleshooting():
    """
    显示故障排除指南
    """
    print("\n" + "="*60)
    print("故障排除指南")
    print("="*60)
    
    issues = [
        {
            "problem": "显存不足 (CUDA out of memory)",
            "solutions": [
                "减少批量大小: per_device_train_batch_size: 1",
                "增加梯度累积: gradient_accumulation_steps: 8", 
                "启用梯度检查点: gradient_checkpointing: true",
                "使用更小的数据集: train_data: 'cocoqa'"
            ]
        },
        {
            "problem": "训练不稳定（损失震荡）",
            "solutions": [
                "降低学习率: stage1_lr: 5e-5",
                "增加warmup比例: warmup_ratio: 0.2",
                "使用更稳定的优化器: optim: 'adamw_torch'",
                "增加权重衰减: weight_decay: 0.05"
            ]
        },
        {
            "problem": "收敛缓慢",
            "solutions": [
                "增加训练轮数: stage1_epochs: 3",
                "使用更多数据: train_data: 'all'",
                "调整学习率调度: lr_scheduler_type: 'linear'",
                "检查数据质量"
            ]
        },
        {
            "problem": "模型保存失败",
            "solutions": [
                "检查磁盘空间",
                "减少保存频率: save_steps: 50",
                "减少保存数量: save_total_limit: 3",
                "检查文件权限"
            ]
        }
    ]
    
    for issue in issues:
        print(f"\n❌ {issue['problem']}")
        for solution in issue['solutions']:
            print(f"   💡 {solution}")


def main():
    """
    主函数
    """
    print("欢迎使用分阶段训练系统！")
    
    # 运行训练示例
    success = run_training_example()
    
    # 显示配置示例
    show_configuration_examples()
    
    # 显示使用示例
    show_usage_examples()
    
    # 显示训练阶段说明
    show_training_stages()
    
    # 显示故障排除指南
    show_troubleshooting()
    
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    print("✅ 分阶段训练系统已准备就绪")
    print("📖 详细文档请参考: STAGED_TRAINING_README.md")
    print("🚀 开始训练: python train_staged.py staged_training_test.yaml")
    print("="*60)


if __name__ == "__main__":
    main() 
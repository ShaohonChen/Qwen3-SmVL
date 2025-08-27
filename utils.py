# 增强版工具函数
# 支持下游任务和视频数据处理

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import torch
import torch.nn as nn
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.models.smolvlm.modeling_smolvlm import SmolVLMConnector


def load_processor():
    """
    加载和配置数据处理器
    
    此函数的作用：
    1. 加载SmolVLM2的图像处理器
    2. 加载Qwen3的分词器  
    3. 将两者组合并配置特殊token
    4. 设置聊天模板
    
    这样做的原因是要将SmolVLM2的视觉处理能力与Qwen3的文本处理能力结合，
    创建一个支持中文的多模态处理器。
    
    Returns:
        processor: 配置好的多模态处理器
    """
    print("正在加载SmolVLM2处理器...")
    smolvlm2_processor = AutoProcessor.from_pretrained(
        "model/SmolVLM2-256M-Video-Instruct"
    )
    
    print("正在加载Qwen3分词器...")
    qwen3_tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-0.6B")

    print("正在配置处理器...")
    smolvlm2_processor.tokenizer = qwen3_tokenizer
    
    # 加载聊天模板文件
    with open("chat_template.jinja", "r") as f:
        smolvlm2_processor.chat_template = f.read()
    
    # 配置特殊token
    smolvlm2_processor.fake_image_token = "<vision_start>"
    smolvlm2_processor.image_token = "<|image_pad|>"
    smolvlm2_processor.image_token_id = 151655
    smolvlm2_processor.end_of_utterance_token = "<im_end>"
    smolvlm2_processor.global_image_token = "<|vision_pad|>"
    smolvlm2_processor.video_token = "<|video_pad|>"

    return smolvlm2_processor


def load_model(device="cuda:0"):
    """
    加载和构建混合多模态模型
    
    此函数实现了一个创新的模型架构组合：
    1. 使用SmolVLM2的视觉编码器处理图像
    2. 使用Qwen3的语言模型处理文本
    3. 创建新的连接器将视觉特征映射到文本特征空间
    
    这种组合的优势：
    - SmolVLM2：优秀的视觉理解能力
    - Qwen3：强大的中文语言能力
    - 自定义连接器：优化的跨模态特征映射
    
    Args:
        device: 运行设备，默认为"cuda:0"
    
    Returns:
        smolvlm2_02B_model: 配置好的混合多模态模型
    """
    print("正在加载SmolVLM2视觉-语言模型...")
    smolvlm2_02B_model = AutoModelForImageTextToText.from_pretrained(
        "model/SmolVLM2-256M-Video-Instruct",
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",
    ).to(device)
    
    print("正在加载Qwen3语言模型...")
    qwen3_06b_model = AutoModelForCausalLM.from_pretrained(
        "model/Qwen3-0.6B", 
        torch_dtype=torch.bfloat16
    ).to(device)

    print("正在构建连接器配置...")
    @dataclass
    class VisionConfig:
        hidden_size: int = 768

    @dataclass
    class TextConfig:
        hidden_size: int = 1024

    @dataclass
    class ConnectConfig:
        scale_factor: int = 4
        vision_config: VisionConfig = field(default_factory=VisionConfig)
        text_config: TextConfig = field(default_factory=TextConfig)

    new_connector_config = ConnectConfig()

    print("正在创建新的连接器...")
    new_connector = SmolVLMConnector(new_connector_config).to(device).to(torch.bfloat16)
    smolvlm2_02B_model.model.connector = new_connector

    print("正在替换语言模型组件...")
    smolvlm2_02B_model.model.text_model = qwen3_06b_model.model
    smolvlm2_02B_model.lm_head = qwen3_06b_model.lm_head
    
    print("正在更新模型配置...")
    vocab_size = qwen3_06b_model.vocab_size
    smolvlm2_02B_model.vocab_size = vocab_size
    smolvlm2_02B_model.model.vocab_size = vocab_size
    smolvlm2_02B_model.config.vocab_size = vocab_size
    smolvlm2_02B_model.config.text_config.vocab_size = vocab_size
    smolvlm2_02B_model.model.config.vocab_siz = vocab_size
    smolvlm2_02B_model.model.config.text_config.vocab_size = vocab_size
    
    image_token_id = 151655
    smolvlm2_02B_model.image_token_id = image_token_id
    smolvlm2_02B_model.model.image_token_id = image_token_id
    smolvlm2_02B_model.config.image_token_id = image_token_id
    smolvlm2_02B_model.model.config.image_token_id = image_token_id
    
    smolvlm2_02B_model.generation_config.eos_token_id = 151645
    
    print("模型构建完成！")
    return smolvlm2_02B_model


def load_downstream_datasets(task_names: List[str]):
    """
    加载下游任务数据集
    
    Args:
        task_names: 下游任务名称列表
    
    Returns:
        datasets: 下游任务数据集字典
    """
    downstream_datasets = {}
    
    # 定义可用的下游任务
    available_tasks = {
        "captioning": ["coco_caption", "flickr30k", "nocaps"],
        "vqa": ["vqa_v2", "gqa", "okvqa"],
        "video": ["msrvtt", "activitynet", "youcook2"],
        "ocr": ["textvqa", "docvqa", "funsd"],
        "reasoning": ["clevr", "nlvr2", "vcr"],
    }
    
    for task_name in task_names:
        if task_name in available_tasks:
            print(f"加载下游任务: {task_name}")
            # 这里可以添加具体的数据集加载逻辑
            # 目前返回空数据集作为占位符
            downstream_datasets[task_name] = []
    
    return downstream_datasets


def create_video_processor():
    """
    创建视频数据处理器
    
    Returns:
        video_processor: 视频处理器
    """
    # 这里可以添加视频处理器的创建逻辑
    # 目前返回None作为占位符
    return None


def apply_parameter_efficient_finetuning(model, method="lora"):
    """
    应用参数高效微调方法
    
    Args:
        model: 要微调的模型
        method: 微调方法 ("lora", "adapter", "prefix_tuning")
    
    Returns:
        model: 应用了参数高效微调的模型
    """
    if method == "lora":
        # 应用LoRA微调
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        
    elif method == "adapter":
        # 应用Adapter微调
        from peft import AdapterConfig, get_peft_model
        
        adapter_config = AdapterConfig(
            adapter_size=64,
            adapter_non_linearity="relu",
            adapter_dropout=0.1
        )
        model = get_peft_model(model, adapter_config)
    
    return model


def create_custom_loss_function():
    """
    创建自定义损失函数
    
    Returns:
        loss_fn: 自定义损失函数
    """
    def custom_loss(logits, labels, attention_mask=None):
        """
        自定义损失函数，支持多任务学习
        
        Args:
            logits: 模型输出
            labels: 真实标签
            attention_mask: 注意力掩码
        
        Returns:
            loss: 计算得到的损失
        """
        # 基础交叉熵损失
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        # 移位logits和labels用于语言建模
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 计算损失
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                       shift_labels.view(-1))
        
        return loss
    
    return custom_loss


def setup_mixed_precision_training():
    """
    设置混合精度训练
    
    Returns:
        scaler: 梯度缩放器
    """
    from torch.cuda.amp import GradScaler
    
    scaler = GradScaler()
    return scaler


def create_optimizer_with_different_lrs(model, stage_config):
    """
    为不同组件创建不同学习率的优化器
    
    Args:
        model: 模型
        stage_config: 阶段配置
    
    Returns:
        optimizer: 优化器
    """
    # 为不同组件设置不同的学习率
    param_groups = []
    
    # 连接器使用较高学习率
    if hasattr(model, 'model') and hasattr(model.model, 'connector'):
        param_groups.append({
            'params': model.model.connector.parameters(),
            'lr': stage_config.get('connector_lr', 1e-4)
        })
    
    # 视觉编码器使用中等学习率
    if hasattr(model, 'model') and hasattr(model.model, 'vision_model'):
        param_groups.append({
            'params': model.model.vision_model.parameters(),
            'lr': stage_config.get('vision_lr', 5e-5)
        })
    
    # 文本模型使用较低学习率
    if hasattr(model, 'model') and hasattr(model.model, 'text_model'):
        param_groups.append({
            'params': model.model.text_model.parameters(),
            'lr': stage_config.get('text_lr', 1e-5)
        })
    
    # 其他参数使用默认学习率
    other_params = []
    for name, param in model.named_parameters():
        if not any(name.startswith(prefix) for prefix in 
                  ['model.connector', 'model.vision_model', 'model.text_model']):
            other_params.append(param)
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': stage_config.get('default_lr', 1e-4)
        })
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    return optimizer


def save_model_checkpoint(model, processor, output_dir, stage_name, 
                         save_optimizer=True, save_scheduler=True):
    """
    保存模型检查点
    
    Args:
        model: 模型
        processor: 处理器
        output_dir: 输出目录
        stage_name: 阶段名称
        save_optimizer: 是否保存优化器状态
        save_scheduler: 是否保存调度器状态
    """
    import os
    
    checkpoint_dir = os.path.join(output_dir, stage_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 保存模型
    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    
    # 保存训练状态
    training_state = {
        'stage_name': stage_name,
        'model_config': model.config.to_dict(),
        'processor_config': processor.config.to_dict() if hasattr(processor, 'config') else {}
    }
    
    with open(os.path.join(checkpoint_dir, 'training_state.json'), 'w') as f:
        import json
        json.dump(training_state, f, indent=2)
    
    print(f"模型检查点已保存到: {checkpoint_dir}")


def load_model_checkpoint(checkpoint_dir, device="cuda:0"):
    """
    加载模型检查点
    
    Args:
        checkpoint_dir: 检查点目录
        device: 设备
    
    Returns:
        model: 加载的模型
        processor: 加载的处理器
        training_state: 训练状态
    """
    # 加载模型和处理器
    model = load_model(device)
    processor = load_processor()
    
    # 加载模型权重
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'pytorch_model.bin')))
    
    # 加载训练状态
    training_state_path = os.path.join(checkpoint_dir, 'training_state.json')
    if os.path.exists(training_state_path):
        with open(training_state_path, 'r') as f:
            import json
            training_state = json.load(f)
    else:
        training_state = {}
    
    print(f"模型检查点已从 {checkpoint_dir} 加载")
    return model, processor, training_state 
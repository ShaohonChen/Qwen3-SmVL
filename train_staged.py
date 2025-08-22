# 导入必要的库
import os
import sys
from dataclasses import dataclass
from functools import partial
from PIL import Image
from typing import Optional, List

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    HfArgumentParser,
)
from transformers.trainer_utils import get_last_checkpoint
import datasets
import swanlab

from utils import load_model, load_processor


device = "cuda"

################
# 分阶段训练参数配置类
################
@dataclass
class StagedTrainingArgs(TrainingArguments):
    """
    分阶段训练参数配置类
    
    支持三个阶段的全量微调：
    1. 阶段1: 冻结视觉和文本，只训练连接器
    2. 阶段2: 训练视觉+连接器，冻结文本
    3. 阶段3: 全量微调
    """
    # 数据相关参数
    train_data: str = "cocoqa"
    seed: int = 42
    data_seed: int = 42
    max_steps: Optional[int] = None  # 最大训练步数
    
    # 分阶段训练参数
    training_stage: str = "stage1"  # stage1, stage2, stage3
    resume_from_stage: Optional[str] = None  # 从指定阶段恢复训练
    stage1_epochs: int = 1
    stage2_epochs: int = 1
    stage3_epochs: int = 1
    
    # 各阶段的学习率
    stage1_lr: float = 1e-4
    stage2_lr: float = 5e-5
    stage3_lr: float = 1e-5
    
    # 批量大小设置
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    
    # 数据加载参数
    dataloader_pin_memory: bool = False
    
    # 优化器设置
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    optim: str = "adamw_torch"
    
    # 日志和评估设置
    logging_steps: int = 5
    evaluation_strategy: str = "steps"
    eval_steps: int = 10
    
    # 模型保存设置
    save_strategy: str = "steps"
    save_steps: int = 10
    save_total_limit: int = 8
    
    # 精度和性能设置
    bf16: bool = True
    gradient_checkpointing: bool = False
    
    # 输出和实验跟踪
    output_dir: str = "./model/staged_training"
    overwrite_output_dir: bool = True
    report_to: str = "swanlab"
    run_name: str = "staged_training"
    remove_unused_columns: bool = False
    
    # 下游任务配置
    downstream_tasks: Optional[List[str]] = None  # 下游任务列表
    video_data: bool = False  # 是否包含视频数据


################
# 多模态数据集加载函数
################
def load_mm_data(select_data, data_seed=42):
    """
    加载多模态训练数据集
    
    Args:
        select_data: 选择的数据集名称
        data_seed: 数据划分的随机种子
    """
    # 定义所有可用的数据集列表（来自Cauldron数据集集合）
    all_data_names = [
        "chartqa",              # 图表问答数据集
        "finqa",                # 金融问答数据集
        "aokvqa",               # A-OKVQA视觉问答数据集
        # "mimic_cgd",          # 医学图像数据集（已注释，质量不佳）
        "figureqa",             # 图形问答数据集
        "diagram_image_to_text",# 图表转文本数据集
        "geomverse",            # 几何推理数据集
        "ai2d",                 # AI2科学图表数据集
        "iam",                  # 手写文档数据集
        "infographic_vqa",      # 信息图表问答数据集
        # "localized_narratives", # 局部化叙述数据集（已注释，质量不佳）
        "intergps",             # 地理空间推理数据集
        "hateful_memes",        # 仇恨表情包检测数据集
        "clevr",                # CLEVR视觉推理数据集
        "iconqa",               # 图标问答数据集
        "multihiertt",          # 层次表格推理数据集
        "mapqa",                # 地图问答数据集
        "datikz",               # TikZ图形数据集
        # "okvqa",              # OK-VQA数据集（已注释，质量不佳）
        "hitab",                # 层次表格问答数据集
        "chart2text",           # 图表转文本数据集
        # "ocrvqa",             # OCR视觉问答数据集（已注释，质量不佳）
        # "clevr_math",         # CLEVR数学推理数据集（已注释，质量不佳）
        # "nlvr2",              # NLVR2视觉推理数据集（已注释，质量不佳）
        "cocoqa",               # COCO问答数据集
        "docvqa",               # 文档视觉问答数据集
        "dvqa",                 # 条形图问答数据集
    ]
    
    if select_data == "all":
        tmp_data = all_data_names
    elif select_data in all_data_names:
        tmp_data = [select_data]
    else:
        raise ValueError(f"cannot find {select_data}")

    data_list = []
    for data_name in tmp_data:
        try:
            data_list.append(
                datasets.load_dataset("data/the_cauldron", data_name)["train"]
            )
        except Exception:
            print(f"bad dataset:{data_name}")
    
    # 将所有数据集合并为一个数据集
    raw_data = datasets.concatenate_datasets(data_list)
    
    # 划分训练集和测试集：随机选择64条作为测试集，其余作为训练集
    # 使用固定种子确保结果可复现，64条测试集是为了减少评估时间
    raw_data = raw_data.train_test_split(
        64, shuffle=True, seed=data_seed
    )
    
    if select_data == "all":
        raw_data["train"] = raw_data["train"].select(range(60 * 1024))
    
    return raw_data


################
# 分阶段参数冻结函数
################
def apply_stage_freeze(model, stage: str):
    """
    冻结模型的大部分参数，只训练连接器层
    
    这是一种高效的微调策略：
    - 冻结视觉编码器：保持图像特征提取能力
    - 冻结语言模型：保持文本生成能力  
    - 只训练连接器：学习视觉特征到语言特征的映射
    
    Args:
        model: 要冻结的模型
        stage: 训练阶段 ("stage1", "stage2", "stage3")
    
    Returns:
        model: 冻结参数后的模型
    """
    print(f"应用阶段 {stage} 的参数冻结策略...")
    
    if stage == "stage1":
        # 阶段1: 冻结视觉和文本模块，只训练连接器
        print("冻结视觉编码器...")
        for _, param in model.model.vision_model.named_parameters():
            param.requires_grad = False
            
        print("冻结文本模型...")
        for _, param in model.model.text_model.named_parameters():
            param.requires_grad = False
            
        print("只训练连接器...")
        for _, param in model.model.connector.named_parameters():
            param.requires_grad = True
            
    elif stage == "stage2":
        # 阶段2: 训练视觉+连接器，冻结文本
        print("解冻视觉编码器...")
        for _, param in model.model.vision_model.named_parameters():
            param.requires_grad = True
            
        print("保持连接器可训练...")
        for _, param in model.model.connector.named_parameters():
            param.requires_grad = True
            
        print("冻结文本模型...")
        for _, param in model.model.text_model.named_parameters():
            param.requires_grad = False
            
    elif stage == "stage3":
        # 阶段3: 全量微调
        print("全量微调：所有参数都可训练...")
        for _, param in model.named_parameters():
            param.requires_grad = True
            
    else:
        raise ValueError(f"未知的训练阶段: {stage}")
    
    return model


def print_trainable_parameters(model):
    """
    打印模型中可训练参数的数量和比例
    
    这有助于了解：
    - 有多少参数参与训练
    - 训练效率如何
    - 内存占用情况
    
    Args:
        model: 要统计的模型
    """
    trainable_params = 0
    all_param = 0
    
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(
        f"trainable params: {trainable_params/(2**20):.2f}M || "
        f"all params: {all_param/(2**20):.2f}M || "
        f"trainable%: {100 * trainable_params / all_param:.2f}%"
    )


################
# 数据处理函数
################
def data_collate_fix2k(examples, processor, device, max_length=2048):
    """
    数据整理函数：将原始数据转换为模型可以处理的格式
    
    此函数的作用：
    1. 处理图像和文本数据
    2. 应用聊天模板格式化对话
    3. 进行分词和编码
    4. 创建训练标签
    5. 处理填充和截断
    
    Args:
        examples: 批量的原始数据样本
        processor: 模型的处理器（包含分词器和图像处理器）
        device: 运行设备
        max_length: 最大序列长度
    
    Returns:
        batch: 处理后的批量数据，包含input_ids、attention_mask、pixel_values、labels等
    """
    batch_text = []
    batch_image = []
    
    for example in examples:
        images = example["images"][:1]
        batch_image.append(images)
        image_num = len(images)
        
        chat_texts = example["texts"][0]
        
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}] * image_num
                + [{"type": "text", "text": chat_texts["user"]}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": chat_texts["assistant"]}],
            },
        ]
        
        text = processor.apply_chat_template(
            messages, enable_thinking=False, add_generation_prompt=False
        )
        batch_text.append(text)

    batch = processor(
        text=batch_text,
        images=batch_image,
        max_length=max_length,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
    
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == processor.image_token_id] = -100
    
    batch["labels"] = labels
    return batch.to(device, dtype=torch.bfloat16)


################
# 分阶段训练函数
################
def train_stage(model, processor, raw_data, stage_args, stage_name):
    """
    执行单个阶段的训练
    
    Args:
        model: 要训练的模型
        processor: 数据处理器
        raw_data: 训练数据
        stage_args: 当前阶段的训练参数
        stage_name: 阶段名称
    
    Returns:
        trainer: 训练完成的训练器
    """
    print(f"\n{'='*20} 开始 {stage_name} 训练 {'='*20}")
    
    # 应用参数冻结策略
    model = apply_stage_freeze(model, stage_name)
    print_trainable_parameters(model)
    
    # 创建数据整理函数
    collate_fn = partial(
        data_collate_fix2k, processor=processor, device=device
    )
    
    # 检查检查点
    last_checkpoint = None
    stage_output_dir = os.path.join(stage_args.output_dir, stage_name)
    
    if os.path.isdir(stage_output_dir) and not stage_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(stage_output_dir)
        if last_checkpoint is not None:
            print(f"从检查点恢复训练: {last_checkpoint}")
    
    # 创建阶段特定的训练参数
    stage_training_args = TrainingArguments(
        output_dir=stage_output_dir,
        overwrite_output_dir=stage_args.overwrite_output_dir,
        num_train_epochs=getattr(stage_args, f"{stage_name}_epochs", 1),
        learning_rate=getattr(stage_args, f"{stage_name}_lr", 1e-4),
        per_device_train_batch_size=stage_args.per_device_train_batch_size,
        per_device_eval_batch_size=stage_args.per_device_eval_batch_size,
        gradient_accumulation_steps=stage_args.gradient_accumulation_steps,
        warmup_ratio=stage_args.warmup_ratio,
        lr_scheduler_type=stage_args.lr_scheduler_type,
        weight_decay=stage_args.weight_decay,
        optim=stage_args.optim,
        logging_steps=stage_args.logging_steps,
        eval_strategy=stage_args.evaluation_strategy,
        eval_steps=stage_args.eval_steps,
        save_strategy=stage_args.save_strategy,
        save_steps=stage_args.save_steps,
        save_total_limit=stage_args.save_total_limit,
        bf16=stage_args.bf16,
        gradient_checkpointing=stage_args.gradient_checkpointing,
        report_to=stage_args.report_to,
        run_name=f"{stage_args.run_name}_{stage_name}",
        remove_unused_columns=stage_args.remove_unused_columns,
        dataloader_pin_memory=stage_args.dataloader_pin_memory,
        max_steps=stage_args.max_steps,  # 添加最大步数限制
    )
    
    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=stage_training_args,
        train_dataset=raw_data["train"],
        eval_dataset=raw_data["test"],
        data_collator=collate_fn,
    )
    
    # 开始训练
    print(f"开始 {stage_name} 训练...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # 保存模型
    print(f"保存 {stage_name} 模型...")
    model.save_pretrained(stage_output_dir)
    processor.save_pretrained(stage_output_dir)
    
    # 记录到SwanLab
    if trainer.state.is_world_process_zero:
        swanlab.log({
            f"{stage_name}/final_loss": trainer.state.log_history[-1]["train_loss"],
            f"{stage_name}/total_steps": trainer.state.global_step,
        })
    
    return trainer


################
# 主训练函数
################
def main(training_args):
    """
    主训练函数：执行分阶段训练流程
    """
    print("=" * 50)
    print("开始分阶段训练流程")
    print("=" * 50)
    
    # 初始化模型和处理器
    print("正在加载模型和处理器...")
    processor = load_processor()
    model = load_model(device)
    
    # 准备训练数据
    print(f"正在加载数据集: {training_args.train_data}")
    raw_data = load_mm_data(
        select_data=training_args.train_data, 
        data_seed=training_args.data_seed
    )
    print(f"数据集加载完成，总数据条数：{raw_data}")
    
    # 确定训练阶段
    stages = ["stage1", "stage2", "stage3"]
    if training_args.resume_from_stage:
        start_idx = stages.index(training_args.resume_from_stage)
        stages = stages[start_idx:]
        print(f"从阶段 {training_args.resume_from_stage} 恢复训练")
    
    # 执行分阶段训练
    for stage in stages:
        if stage == training_args.training_stage or training_args.training_stage == "all":
            trainer = train_stage(model, processor, raw_data, training_args, stage)
            
            # 阶段间推理测试
            if trainer.state.is_world_process_zero:
                test_inference(model, processor, stage)
    
    print("分阶段训练完成！")


################
# 推理测试函数
################
def test_inference(model, processor, stage_name):
    """
    在训练阶段后进行推理测试
    """
    print(f"\n{'='*20} {stage_name} 推理测试 {'='*20}")
    
    with torch.no_grad():
        question = "图中有什么动物？"
        
        messages = [
            {
                "role": "system",
                "content": "使用中文回答所有问题。",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]
        
        texts = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=True,
        )
        
        print("输入文本:")
        print(texts)
        
        images = [[Image.open("./resource/dog.png")]]
        
        batch = processor(
            text=[texts],
            images=images,
            max_length=1024,
            return_tensors="pt",
            padding_side="left",
            padding=True,
        ).to(model.device, dtype=torch.bfloat16)
        
        generated_ids = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=256
        )
        
        input_ids_len = batch["input_ids"].shape[1]
        generated_texts = processor.batch_decode(
            generated_ids[:, input_ids_len:], skip_special_tokens=True
        )
        
        print(f"{stage_name} 生成结果:")
        print(generated_texts[0])
        
        # 记录到SwanLab
        table = swanlab.echarts.Table()
        headers = ["阶段", "问题", "回答"]
        rows = [[stage_name, question, generated_texts[0]]]
        table.add(headers, rows)
        
        swanlab.log({
            f"{stage_name}/sample/输入图像": swanlab.Image(images[0][0]),
            f"{stage_name}/sample/问答": table,
        })


# 程序入口点
if __name__ == "__main__":
    parser = HfArgumentParser(StagedTrainingArgs)
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        print(f"从YAML配置文件加载参数: {sys.argv[1]}")
        (training_args,) = parser.parse_yaml_file(
            yaml_file=os.path.abspath(sys.argv[1])
        )
    else:
        print("从命令行参数加载配置")
        (training_args,) = parser.parse_args_into_dataclasses()
    
    main(training_args) 
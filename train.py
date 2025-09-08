# 导入必要的库
import os  # 操作系统接口，用于文件路径操作
import sys  # 系统相关的参数和函数
from dataclasses import dataclass  # 数据类装饰器
from functools import partial  # 偏函数工具
from PIL import Image  # Python图像处理库

import torch  # PyTorch深度学习框架
from transformers import (
    TrainingArguments,  # Transformers训练参数类
    Trainer,  # Transformers训练器
    HfArgumentParser,  # HuggingFace参数解析器
)
from transformers.trainer_utils import get_last_checkpoint  # 获取最新检查点工具
import datasets  # HuggingFace数据集库
import swanlab  # 实验跟踪和可视化工具

from utils import load_model, load_processor  # 导入自定义的模型和处理器加载函数

device = "cuda"  # 设置运行设备为GPU

################
# 多模态数据集加载函数
################
def load_mm_data(select_data):
    """
    加载多模态训练数据集
    
    Args:
        select_data (str): 选择的数据集名称，可以是具体的数据集名或"all"
    
    Returns:
        datasets.DatasetDict: 包含train和test的数据集字典
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
    
    # 根据选择确定要加载的数据集
    if select_data == "all":
        tmp_data = all_data_names  # 使用所有数据集
    elif select_data in all_data_names:
        tmp_data = [select_data]   # 使用指定的单个数据集
    else:
        raise f"cannot find {tmp_data}"  # 抛出错误：找不到指定数据集

    # 逐个加载数据集并合并
    data_list = []
    for data_name in tmp_data:
        try:
            # 从Cauldron数据集集合中加载指定数据集的训练部分
            data_list.append(
                datasets.load_dataset("data/the_cauldron", data_name)["train"]
            )
        except:
            print(f"bad dataset:{data_name}")  # 打印加载失败的数据集
    
    # 将所有数据集合并为一个数据集
    raw_data = datasets.concatenate_datasets(data_list)
    
    # 划分训练集和测试集：随机选择64条作为测试集，其余作为训练集
    # 使用固定种子确保结果可复现，64条测试集是为了减少评估时间
    raw_data = raw_data.train_test_split(
        64, shuffle=True, seed=training_args.data_seed
    )
    
    # 如果使用全部数据，则限制训练集大小为60K条，避免训练时间过长
    if select_data == "all":
        raw_data["train"] = raw_data["train"].select(range(60 * 1024))
    
    return raw_data


################
# 模型参数冻结和参数统计函数
################
def freeze_model(qwen_smvl):
    """
    冻结模型的大部分参数，只训练连接器层
    
    这是一种高效的微调策略：
    - 冻结视觉编码器：保持图像特征提取能力
    - 冻结语言模型：保持文本生成能力  
    - 只训练连接器：学习视觉特征到语言特征的映射
    
    Args:
        qwen_smvl: 要冻结的多模态模型
    
    Returns:
        qwen_smvl: 冻结参数后的模型
    """
    # 冻结文本模型（语言模型）的所有参数
    for _, param in qwen_smvl.model.text_model.named_parameters():
        param.requires_grad = False
    
    # 冻结视觉模型（图像编码器）的所有参数
    for _, param in qwen_smvl.model.vision_model.named_parameters():
        param.requires_grad = False
    
    # 注释掉的代码：如果需要也可以冻结语言模型头部
    # for _, param in qwen_smvl.lm_head.named_parameters():
    #     param.requires_grad = False
    
    return qwen_smvl


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
    trainable_params = 0  # 可训练参数数量
    all_param = 0         # 总参数数量
    
    # 遍历模型的所有参数
    for _, param in model.named_parameters():
        all_param += param.numel()           # 累加总参数数
        if param.requires_grad:              # 如果参数需要梯度更新
            trainable_params += param.numel() # 累加可训练参数数
    
    # 打印参数统计信息（以百万为单位显示）
    print(
        f"trainable params: {trainable_params/(2**20):.2f}M || "
        f"all params: {all_param/(2**20):.2f}M || "
        f"trainable%: {100 * trainable_params / all_param:.2f}%"
    )


################
# 数据处理和批量整理函数
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
    batch_text = []   # 存储处理后的文本
    batch_image = []  # 存储图像数据
    
    # 处理批量中的每个样本
    for example in examples:
        # 只取第一张图像，避免显存不足（多图像会占用大量显存）
        images = example["images"][:1]
        batch_image.append(images)
        image_num = len(images)  # 图像数量
        
        # 获取对话文本内容
        chat_texts = example["texts"][0]
        
        # 构建对话消息格式，符合聊天模型的输入要求
        messages = [
            {
                "role": "user",  # 用户角色
                "content": [{"type": "image"}] * image_num  # 图像占位符
                + [{"type": "text", "text": chat_texts["user"]}],  # 用户问题
            },
            {
                "role": "assistant",  # 助手角色
                "content": [{"type": "text", "text": chat_texts["assistant"]}],  # 助手回答
            },
        ]
        
        # 应用聊天模板，将对话格式化为模型需要的文本格式
        # enable_thinking=False: 不启用思考模式
        # add_generation_prompt=False: 不添加生成提示符
        text = processor.apply_chat_template(
            messages, enable_thinking=False, add_generation_prompt=False
        )

        batch_text.append(text)

    # 使用处理器对文本和图像进行编码
    batch = processor(
        text=batch_text,           # 文本列表
        images=batch_image,        # 图像列表
        max_length=max_length,     # 最大长度
        return_tensors="pt",       # 返回PyTorch张量
        padding="max_length",      # 填充到最大长度
        truncation=True,           # 启用截断
    )
    
    # 创建训练标签：复制input_ids作为标签
    labels = batch["input_ids"].clone()
    
    # 设置特殊token的标签为-100，这样在计算损失时会被忽略
    labels[labels == processor.tokenizer.pad_token_id] = -100  # 忽略填充token
    labels[labels == processor.image_token_id] = -100          # 忽略图像token
    
    batch["labels"] = labels
    
    # 将数据移动到指定设备并转换为bfloat16精度以节省显存
    return batch.to(device, dtype=torch.bfloat16)


################
# 训练参数配置类
################
@dataclass
class MyTrainArgs(TrainingArguments):
    """
    自定义训练参数类，继承自TrainingArguments
    
    这个类定义了训练过程中的所有重要参数，包括：
    - 数据相关参数
    - 训练策略参数  
    - 优化器参数
    - 保存和评估策略
    - 实验跟踪参数
    """
    # 数据相关参数
    train_data: str = "cocoqa"              # 训练数据集名称，默认使用cocoqa
    seed: int = 42                          # 随机种子，确保实验可复现
    data_seed: int = 42                     # 数据划分的随机种子
    max_steps: Optional[int] = None  # 最大训练步数
    
    # 批量大小设置
    per_device_train_batch_size: int = 1    # 每个设备的训练批量大小
    per_device_eval_batch_size: int = 1     # 每个设备的评估批量大小（设为1防止显存溢出）
    gradient_accumulation_steps: int = 4    # 梯度累积步数，有效批量大小 = batch_size * gradient_accumulation_steps
    
    # 数据加载参数
    dataloader_pin_memory: bool = False     # 是否将数据固定在内存中（可能导致显存不足）
    
    # 学习率和优化器设置
    warmup_ratio: float = 0.1              # 热身阶段的比例（总训练步数的10%）
    learning_rate: float = 1e-4            # 学习率
    lr_scheduler_type: str = "cosine"      # 学习率调度器类型（余弦退火）
    weight_decay: float = 0.01             # 权重衰减（L2正则化）
    optim: str = "adamw_torch"             # 优化器类型
    
    # 日志和评估设置
    logging_steps: int = 5                 # 每5步记录一次日志
    evaluation_strategy: str = "steps"     # 评估策略：按步数评估
    eval_steps: int = 10                   # 每10步进行一次评估
    
    # 模型保存设置
    save_strategy: str = "steps"           # 保存策略：按步数保存
    save_steps: int = 10                   # 每10步保存一次检查点
    save_total_limit: int = 8              # 最多保留8个检查点
    
    # 精度和性能设置
    bf16: bool = True                      # 使用bfloat16混合精度训练（节省显存，加速训练）
    gradient_checkpointing: bool = False   # 梯度检查点（可节省显存但会增加计算时间）
    
    # 输出和实验跟踪
    output_dir: str = "./model/qwen-smovlm"              # 模型输出目录
    overwrite_output_dir: bool = True                    # 是否覆盖输出目录
    report_to: str = "swanlab"                          # 实验跟踪工具
    run_name: str = "freeze_except_connector_fulldata"  # 实验运行名称
    remove_unused_columns: bool = False                 # 不移除未使用的数据列


def main(training_args):
    """
    主训练函数：执行完整的模型训练流程
    
    包含以下主要步骤：
    1. 初始化模型和处理器
    2. 配置训练参数（冻结部分参数）
    3. 加载和预处理数据
    4. 设置训练器并开始训练
    5. 保存训练好的模型
    6. 进行推理测试
    
    Args:
        training_args: 训练参数配置对象
    """
    ################
    # 初始化模型和处理器
    ################
    print("正在加载模型和处理器...")
    qwen_smvl_processor = load_processor()  # 加载数据处理器（分词器+图像处理器）
    qwen_smvl = load_model(device)          # 加载多模态模型到GPU
    
    # 应用参数冻结策略：只训练连接器层，冻结视觉编码器和语言模型
    print("应用参数冻结策略...")
    qwen_smvl = freeze_model(qwen_smvl)
    
    # 统计并打印可训练参数信息
    print_trainable_parameters(qwen_smvl)

    ################
    # 准备训练数据集
    ################
    print(f"正在加载数据集: {training_args.train_data}")
    raw_data = load_mm_data(select_data=training_args.train_data)
    print(f"数据集加载完成，总数据条数：{raw_data}")

    # 创建数据整理函数（用于批量处理数据）
    collate_fn = partial(
        data_collate_fix2k, processor=qwen_smvl_processor, device=device
    )

    ################
    # 检查和恢复训练检查点
    ################
    last_checkpoint = None  # 初始化检查点变量
    
    # 检查输出目录是否存在且不为空
    if (
        os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        # 尝试获取最新的检查点
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        
        # 如果没有找到检查点但目录不为空，抛出错误
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"输出目录 ({training_args.output_dir}) 已存在且不为空。"
                "使用 --overwrite_output_dir 来覆盖现有内容。"
            )
        
        # 如果找到检查点，打印恢复训练的信息
        if last_checkpoint is not None:
            print(
                f"检测到检查点，将从 {last_checkpoint} 恢复训练。"
                "如要避免此行为，请更改 `--output_dir` 或添加 `--overwrite_output_dir` 从头开始训练。"
            )
    
    ################
    # 初始化训练器并开始训练
    ################
    print("初始化训练器...")
    trainer = Trainer(
        model=qwen_smvl,                    # 要训练的模型
        args=training_args,                 # 训练参数
        train_dataset=raw_data["train"],    # 训练数据集
        eval_dataset=raw_data["test"],      # 评估数据集
        data_collator=collate_fn,           # 数据整理函数
    )
    
    print("开始训练...")
    # 开始训练（如果有检查点则从检查点恢复）
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    print("训练完成，保存模型...")
    # 保存训练好的模型
    qwen_smvl.save_pretrained(training_args.output_dir)

    ################
    # 训练后推理测试
    ################
    print("进行推理测试...")
    with torch.no_grad():  # 禁用梯度计算以节省内存
        # 只在主进程中进行推理测试
        if trainer.state.is_world_process_zero:
            # 定义测试问题
            question = "图中有什么动物？"
            
            # 构建对话消息格式
            messages = [
                {
                    "role": "system",
                    "content": "使用中文回答所有问题。",
                    # 注释掉的内容：如果需要思考模式可以启用
                    # "content": "使用中文回答所有问题，在<think>和</think>中写出思考过程，如果没有思考则为<think> </think>",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},                    # 图像占位符
                        {"type": "text", "text": question},  # 用户问题
                    ],
                },
            ]
            
            # 应用聊天模板格式化输入
            texts = qwen_smvl_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,  # 添加生成提示符
                tokenize=False,             # 不进行分词
                enable_thinking=True,       # 启用思考模式
            )
            
            print("################# 输入文本 #################")
            print(texts)
            
            # 加载测试图像
            images = [[Image.open("./resource/dog.png")]]
            
            # 处理输入数据
            batch = qwen_smvl_processor(
                text=[texts],           # 文本列表
                images=images,          # 图像列表
                max_length=1024,        # 最大长度
                return_tensors="pt",    # 返回PyTorch张量
                padding_side="left",    # 左侧填充
                padding=True,           # 启用填充
            ).to(qwen_smvl.device, dtype=torch.bfloat16)
            
            # 生成回答
            generated_ids = qwen_smvl.generate(
                **batch, 
                do_sample=False,        # 不使用随机采样（确定性生成）
                max_new_tokens=256      # 最大生成token数
            )
            
            # 解码生成的文本（包含输入部分）
            model_context = qwen_smvl_processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )
            
            # 提取仅生成的部分（去除输入部分）
            input_ids_len = batch["input_ids"].shape[1]
            generated_texts = qwen_smvl_processor.batch_decode(
                generated_ids[:, input_ids_len:], skip_special_tokens=True
            )
            
            print("################# 生成文本 #################")
            print(generated_texts[0])

            # 创建实验记录表格
            table = swanlab.echarts.Table()
            headers = ["输入问题", "模型输出"]
            rows = [[question, generated_texts[0]]]
            table.add(headers, rows)
            
            # 记录实验结果到SwanLab
            swanlab.log(
                {
                    "sample/输入图像": swanlab.Image(images[0][0]),    # 输入图像
                    "sample/问题&回复": table,                        # 问答表格
                    "sample/上下文": swanlab.Text(model_context[0]),  # 完整上下文
                }
            )


# 程序入口点：解析命令行参数并启动训练
if __name__ == "__main__":
    """
    程序的入口点
    
    支持两种参数传递方式：
    1. 通过YAML配置文件：python train.py config.yaml
    2. 通过命令行参数：python train.py --train_data cocoqa --learning_rate 1e-4 ...
    
    YAML配置文件方式更适合复杂的实验配置管理。
    """
    # 创建参数解析器，用于解析自定义的训练参数类
    parser = HfArgumentParser(MyTrainArgs)
    
    # 判断参数传递方式
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # 方式1：从YAML文件解析参数
        # 如果只传递了一个参数且是YAML文件，则从文件中解析训练参数
        print(f"从YAML配置文件加载参数: {sys.argv[1]}")
        (training_args,) = parser.parse_yaml_file(
            yaml_file=os.path.abspath(sys.argv[1])
        )
    else:
        # 方式2：从命令行参数解析
        # 从命令行参数中解析训练参数
        print("从命令行参数加载配置")
        (training_args,) = parser.parse_args_into_dataclasses()
    
    # 可选：直接从指定的YAML文件加载参数（用于调试）
    # (training_args,) = parser.parse_yaml_file(yaml_file='full_train.yaml')
    
    # 启动主训练流程
    print("=" * 50)
    print("开始训练流程")
    print("=" * 50)
    main(training_args)

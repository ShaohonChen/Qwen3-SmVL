from dataclasses import dataclass

import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.models.smolvlm.modeling_smolvlm import SmolVLMConnector


def load_processor():
    smolvlm2_processor = AutoProcessor.from_pretrained(
        "model/SmolVLM2-256M-Video-Instruct"
    )
    qwen3_tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-0.6B")

    smolvlm2_processor.tokenizer = qwen3_tokenizer
    with open("chat_template.jinja", "r") as f:
        smolvlm2_processor.chat_template = f.read()
    smolvlm2_processor.fake_image_token = "<vision_start>"
    smolvlm2_processor.image_token = "<|image_pad|>"
    smolvlm2_processor.image_token_id = 151655
    smolvlm2_processor.end_of_utterance_token = "<im_end>"
    smolvlm2_processor.global_image_token = "<|vision_pad|>"
    smolvlm2_processor.video_token = "<|video_pad|>"

    return smolvlm2_processor


def load_model(device="cuda:0"):
    smolvlm2_02B_model = AutoModelForImageTextToText.from_pretrained(
        "model/SmolVLM2-256M-Video-Instruct",
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",
    ).to(device)
    qwen3_06b_model = AutoModelForCausalLM.from_pretrained(
        "model/Qwen3-0.6B", torch_dtype=torch.bfloat16
    ).to(device)

    # 构建配置并且创建连接器
    @dataclass
    class VisionConfig:
        hidden_size: int = 768

    @dataclass
    class TextConfig:
        hidden_size: int = 1024

    @dataclass
    class ConnectConfig:
        scale_factor: int = 4
        vision_config: VisionConfig = VisionConfig()
        text_config: TextConfig = TextConfig()

    new_connector_config = ConnectConfig()

    # 替换 SigLit 到 LLM 的 connector 层
    new_connector = SmolVLMConnector(new_connector_config).to(device).to(torch.bfloat16)
    smolvlm2_02B_model.model.connector = new_connector

    # 替换 VL 模型的 LLM 部分
    # 替换text模型和head
    smolvlm2_02B_model.model.text_model = qwen3_06b_model.model
    smolvlm2_02B_model.lm_head = qwen3_06b_model.lm_head
    # 替换词表大小
    smolvlm2_02B_model.vocab_size = qwen3_06b_model.vocab_size
    smolvlm2_02B_model.model.vocab_size = qwen3_06b_model.vocab_size
    smolvlm2_02B_model.config.vocab_size = qwen3_06b_model.vocab_size
    smolvlm2_02B_model.config.text_config.vocab_size = qwen3_06b_model.vocab_size
    smolvlm2_02B_model.model.config.vocab_siz = qwen3_06b_model.vocab_size
    smolvlm2_02B_model.model.config.text_config.vocab_size = qwen3_06b_model.vocab_size
    # 替换图像token
    smolvlm2_02B_model.image_token_id = 151655
    smolvlm2_02B_model.model.image_token_id = 151655
    smolvlm2_02B_model.config.image_token_id = 151655
    smolvlm2_02B_model.model.config.image_token_id = 151655
    # 替换模型生成停止符
    smolvlm2_02B_model.generation_config.eos_token_id = 151645
    return smolvlm2_02B_model

# 下载模型
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./model/Qwen3-0.6B
modelscope download --model HuggingFaceTB/SmolVLM2-256M-Video-Instruct --local_dir ./model/SmolVLM2-256M-Video-Instruct

# 下载数据集
modelscope download --dataset AI-ModelScope/the_cauldron --local_dir ./data/the_cauldron

modelscope download --dataset HuggingFaceM4/Docmatix --local_dir ./data/Docmatix

# https://blog.csdn.net/Toky_min/article/details/147514735
modelscope download --dataset TIGER-Lab/VideoFeedback --local_dir ./data/VideoFeedback
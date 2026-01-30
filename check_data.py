# # check_data.py
# import os
# import sys
# # 确保能搜到 areal 模块
# sys.path.append(os.getcwd())

# from areal.dataset.dsbench import get_dsbench_modeling_rl_dataset
# from transformers import AutoProcessor

# print("🚀 开始检查数据逻辑...")

# # 模拟配置
# DATA_PATH = "/home/rxl210009/MLAgent_AReaL/data/dsbench_modeling"
# MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# try:
#     print(f"📡 正在尝试加载 Processor: {MODEL_ID}")
#     processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

#     print(f"📦 正在触发 get_dsbench_modeling_rl_dataset...")
#     print(f"📍 目标路径: {DATA_PATH}")

#     # 这一步会触发你写的 hf_hub_download 和 zipfile.extractall
#     dataset = get_dsbench_modeling_rl_dataset(
#         path=DATA_PATH,
#         split="train",
#         processor=processor
#     )

#     print("✅ 数据加载成功！")
#     print(f"📊 样本总数: {len(dataset)}")
#     print(f"📝 第一条数据消息: {dataset[0]['messages']}")

# except Exception as e:
#     print(f"❌ 报错了，这就是你‘没动静’的原因:")
#     print(str(e))

# debug_dsbench.py
import os
import sys
sys.path.append(os.getcwd())

from areal.dataset.dsbench import get_dsbench_modeling_rl_dataset
from transformers import AutoProcessor

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATA_PATH = "/home/rxl210009/MLAgent_AReaL/data/dsbench_modeling"

try:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    # Testing your dataset loading logic
    dataset = get_dsbench_modeling_rl_dataset(DATA_PATH, "train", processor)
    print(f"Success! Loaded {len(dataset)} samples.")
    print(f"Sample data: {dataset[0]['messages']}")
except Exception as e:
    print(f"Error found in your code: {e}")

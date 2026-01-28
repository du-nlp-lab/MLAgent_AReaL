import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error


class DSBenchReward:
    async def __call__(self, code_text, **data):
        work_dir, metric = data.get("work_dir"), data.get("metrics", "accuracy")
        pred_p, gt_p = (
            os.path.join(work_dir, "pred.csv"),
            os.path.join(work_dir, "test_label.csv"),
        )

        if not os.path.exists(pred_p):
            return 0.0
        try:
            y_p = pd.read_csv(pred_p).iloc[:, -1].values
            y_t = pd.read_csv(gt_p).iloc[:, -1].values
            if len(y_p) != len(y_t):
                return 0.0

            if metric == "rmse":
                score = np.sqrt(mean_squared_error(y_t, y_p))
                return 1.0 / (1.0 + score)  # 归一化奖励
            return float(accuracy_score(y_t, y_p))
        except Exception:
            return 0.0


# import os
# import pandas as pd
# import numpy as np
# from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
# from typing import Any

# from areal.utils import logging

# logger = logging.getLogger("DSBenchReward")

# class DSBenchReward:
#     """
#     DSBench 建模任务专用奖励函数。
#     支持 Accuracy, F1, RMSE, R2 等多种指标。
#     """

#     def compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray, metric_name: str) -> float:
#         metric_name = metric_name.lower()
#         try:
#             if metric_name == "accuracy":
#                 return accuracy_score(y_true, y_pred)
#             elif metric_name == "f1":
#                 return f1_score(y_true, y_pred, average='weighted')
#             elif metric_name == "rmse":
#                 return np.sqrt(mean_squared_error(y_true, y_pred))
#             elif metric_name == "r2":
#                 return r2_score(y_true, y_pred)
#             return accuracy_score(y_true, y_pred)
#         except Exception:
#             return 0.0

#     async def __call__(self, code_text: str, **data: Any) -> float:
#         """
#         被 mlagent.py 工作流调用。
#         """
#         work_dir = data.get("work_dir")
#         metric_name = data.get("metrics", "accuracy")

#         # 1. 检查预测结果
#         pred_path = os.path.join(work_dir, "pred.csv")
#         gt_path = os.path.join(work_dir, "test_label.csv") # DSBench 真值文件名

#         if not os.path.exists(pred_path):
#             return 0.0

#         try:
#             df_pred = pd.read_csv(pred_path)
#             df_gt = pd.read_csv(gt_path)

#             if len(df_pred) != len(df_gt):
#                 return 0.0

#             # 假设最后一列是目标值
#             y_pred = df_pred.iloc[:, -1].values
#             y_true = df_gt.iloc[:, -1].values

#             score = self.compute_metric(y_true, y_pred, metric_name)

#             # 归一化奖励到 [0, 1]
#             if metric_name in ["rmse", "mse"]:
#                 return 1.0 / (1.0 + score)
#             return float(score)

#         except Exception as e:
#             logger.error(f"Reward error: {e}")
#             return 0.0

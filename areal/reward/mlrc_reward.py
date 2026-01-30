import os
import re

import numpy as np
from openai import OpenAI

from areal.utils import logging

logger = logging.getLogger("MLRCReward")


class MLRCReward:
    def __init__(self, api_key=None, model="gpt-4o"):
        """
        Integrated Reward for MLRC-Bench.
        Targets the complex logs produced by the MLAgentBench framework.
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        # Common metrics in MLRC-Bench (e.g., Accuracy, F1, MSE)
        self.metric_keys = ["accuracy", "f1", "score", "rmse", "mae", "top1"]

    def _parse_quantitative_score(self, stdout, target_val):
        """Extracts the final score using regex and calculates relative error."""
        extracted = {}
        for key in self.metric_keys:
            pattern = rf"{key}[:=\s]+([0-9]*\.?[0-9]+)"
            matches = re.findall(pattern, stdout, re.IGNORECASE)
            if matches:
                extracted[key.lower()] = float(matches[-1])

        if not extracted:
            return 0.0

        # Compare with the baseline score provided in the dataset
        # We assume 'score' or the first available metric is the primary one
        pred_val = list(extracted.values())[0]

        # Exponential reward: e^(-5.0 * |target - pred| / target)
        rel_err = abs(target_val - pred_val) / (abs(target_val) + 1e-6)
        return float(np.exp(-rel_err * 5.0))

    def _get_gpt4o_audit_score(self, stdout, task_metadata):
        """Uses GPT-4o to audit the reproducibility process and detect cheating."""
        # Focus on the last 5000 characters where results and errors usually reside
        log_segment = stdout[-5000:] if len(stdout) > 5000 else stdout

        prompt = f"""
        Role: Senior ML Research Auditor
        Task: Audit the following reproduction attempt from MLRC-Bench.

        [Target Metric]: {task_metadata.get("score")}
        [Execution Log]:
        ...{log_segment}

        Criteria:
        1. Validity: Did the agent actually run training/inference scripts? (Look for tqdm, epoch logs, etc.)
        2. Honesty: Did the agent just 'print' the target number to cheat?
        3. Improvement: Did the agent successfully modify the code to match or beat the baseline?

        Score: Provide a single float between 0.0 (Fail/Cheat) and 1.0 (Perfect Reproduction).
        Output: ONLY the number.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            score_text = response.choices[0].message.content.strip()
            return float(re.search(r"([0-9]*\.?[0-9]+)", score_text).group(1))
        except Exception as e:
            logger.error(f"GPT-4o Judge error: {e}")
            return 0.0

    def __call__(self, execution_info, target_results):
        """
        Main entry point.
        target_results: {'score': float} (from constants.BASELINE_PERFORMANCE)
        """
        stdout = execution_info.get("stdout", "")
        exit_code = execution_info.get("exit_code", -1)
        target_val = target_results.get("score", 0.0)

        # Level 0: Hard fail if the code didn't run
        if exit_code != 0 or not stdout:
            return 0.0

        # Part 1: Quantitative (Regex) - 50% weight
        regex_score = self._parse_quantitative_score(stdout, target_val)

        # Part 2: Qualitative (GPT-4o Audit) - 50% weight
        # This is critical for 3B models which might try to hallucinate results.
        llm_score = self._get_gpt4o_audit_score(stdout, target_results)

        # Final Hybrid Reward
        return (0.5 * regex_score) + (0.5 * llm_score)

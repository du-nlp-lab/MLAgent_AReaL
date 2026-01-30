import hydra
from omegaconf import DictConfig
from transformers import AutoProcessor

# Import the components we built in previous steps
from areal.dataset.mlrc import get_mlrc_bench_dataset
from areal.reward.mlrc_reward import MLRCReward
from areal.tools.mlrc_executor import MLRCExecutor
from areal.trainer.grpo_trainer import GRPOTrainer
from areal.utils import logging
from areal.workflow.mlagent import MLRCWorkflow

logger = logging.getLogger("MLRC_RL_Main")


@hydra.main(version_base=None, config_path=".", config_name="mlrc_grpo")
def main(cfg: DictConfig):
    """
    Main entry point for MLRC-bench RL training.
    Integrates Qwen-3B with GPT-4o for research reproducibility.
    """
    logger.info(f"Starting MLRC-bench training: {cfg.name}")

    # 1. Initialize Processor (Qwen-2.5-3B-Instruct)
    processor = AutoProcessor.from_pretrained(cfg.actor.path, trust_remote_code=True)

    # 2. Setup Dataset (Step 1)
    # repo_path points to your local copy of yunx-z/MLRC-Bench
    dataset = get_mlrc_bench_dataset(repo_path=cfg.dataset.path, processor=processor)
    logger.info(f"Loaded {len(dataset)} research tasks from MLRC-Bench.")

    # 3. Setup Integrated Reward & Judge (Step 2)
    # Requires OPENAI_API_KEY in environment
    reward_fn = MLRCReward(model=cfg.reward.judge_model)

    # 4. Setup Execution Sandbox & Self-Healing Workflow (Step 3 & 4)
    executor = MLRCExecutor(timeout=cfg.workflow.timeout)
    workflow = MLRCWorkflow(executor=executor, judge_client=reward_fn.client)

    # 5. Initialize GRPO Trainer
    # AReaL handles the distribution of rollout workers and vLLM servers
    trainer = GRPOTrainer(
        config=cfg,
        dataset=dataset,
        reward_fn=reward_fn,
        workflow=workflow,
        processor=processor,
    )

    # 6. Start the Reinforcement Learning Loop
    logger.info("Launching RL Training Loop...")
    trainer.train()


if __name__ == "__main__":
    main()

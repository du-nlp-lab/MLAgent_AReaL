import sys

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.experimental.trainer import PPOTrainer
from areal.utils.hf_utils import load_hf_tokenizer


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    trainer = PPOTrainer(
        config,
        train_dataset=get_custom_dataset(
            split="train", dataset_config=config.train_dataset, tokenizer=tokenizer
        ),
    )

    with trainer:
        trainer.train(
            workflow="areal.workflow.mlagent.MultiturnDSWorkflow",
            workflow_kwargs=dict(
                reward_fn="areal.reward.dsbench.DSBenchReward",
                gconfig=config.gconfig,
                tokenizer=config.tokenizer_path,
            ),
        )


if __name__ == "__main__":
    main(sys.argv[1:])

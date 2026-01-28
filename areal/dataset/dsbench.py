import os
import zipfile

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from areal.utils import logging

logger = logging.getLogger("DSBenchDataset")


def get_dsbench_modeling_rl_dataset(path, split, processor, max_length=None):
    """
    Load the DSBench modeling dataset for Reinforcement Learning.
    Includes robust key checking and English messaging.
    """
    hub_path = os.path.join(path, "data_modeling")
    data_json = os.path.join(hub_path, "data.json")
    data_zip = os.path.join(hub_path, "data.zip")

    # 1. Check/Download
    if not os.path.exists(data_json):
        logger.info(f"Downloading DSBench dataset to {path}...")
        for f in ["data.json", "data.zip"]:
            hf_hub_download(
                repo_id="liqiang888/DSBench",
                filename=f,
                subfolder="data_modeling",
                local_dir=path,
                repo_type="dataset",
            )

    # 2. Extract
    extracted_dir = os.path.join(hub_path, "data")
    if os.path.exists(data_zip) and not os.path.exists(extracted_dir):
        logger.info(f"Extracting {data_zip}...")
        with zipfile.ZipFile(data_zip, "r") as z:
            z.extractall(hub_path)
        logger.info("Extraction completed.")

    # 3. Load
    logger.info(f"Loading dataset from: {data_json}")
    dataset = load_dataset("json", data_files={split: data_json}, split=split)

    def process(sample):
        # RESILIENT KEY CHECK: Try common DSBench field names
        comp_name = (
            sample.get("competition_name")
            or sample.get("name")
            or sample.get("task_name")
        )

        if not comp_name:
            available_keys = list(sample.keys())
            raise KeyError(
                f"Missing competition name in JSON! Found keys: {available_keys}. "
                "Please check if your data.json uses a different field name."
            )

        # Set work_dir to the specific competition folder
        task_work_dir = os.path.join(hub_path, "data", comp_name)

        # English Agent Messages
        messages = [
            {
                "role": "system",
                "content": "You are a professional data scientist. Analyze the dataset and save your predictions to 'pred.csv'.",
            },
            {
                "role": "user",
                "content": f"Competition: {comp_name}\nFiles: {sample.get('files', 'Check working directory')}\n\nTask: Solve this modeling problem using Python.",
            },
        ]
        return {
            "messages": messages,
            "metrics": sample.get("metrics", "accuracy"),
            "work_dir": task_work_dir,
        }

    # Use num_proc=1 to avoid the Python 3.12 multiprocess cleanup noise during debug
    dataset = dataset.map(process, num_proc=1)

    if max_length:
        dataset = dataset.filter(
            lambda x: len(processor.tokenizer.apply_chat_template(x["messages"]))
            <= max_length
        )

    return dataset

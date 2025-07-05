# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import numpy as np
import torch
import tyro
from transformers import TrainingArguments

# 抑制 stderr 输出（慎用）
sys.stderr = open(os.devnull, "w")
import warnings
import logging

# 忽略所有警告
warnings.filterwarnings("ignore")

# 将日志级别设置为 ERROR，忽略所有 WARNING 和 INFO 级别的日志
logging.basicConfig(level=logging.ERROR)

# Force float32 for MPS compatibility
if torch.backends.mps.is_available():
    torch.set_default_dtype(torch.float32)
    # Also set numpy default to float32
    np.seterr(all='ignore')  # Ignore numpy warnings
    import numpy as np
    np.set_printoptions(precision=4, suppress=True)  # Cleaner numpy output
    # Also set numpy default to float32
    import numpy as np
    np.set_printoptions(suppress=True)
    # Force numpy default dtype to float32
    np.seterr(all='ignore')

# Suppress warnings to clean up output
import warnings
warnings.filterwarnings("ignore")
# Disable Albumentations update check
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
from gr00t.utils.peft import get_lora_model


def ensure_float32_collator(batch):
    """Custom collator to ensure all tensors are float32 for MPS compatibility."""
    import torch
    
    def convert_to_float32(item):
        if isinstance(item, torch.Tensor):
            if item.dtype == torch.float64:
                return item.to(torch.float32)
            return item
        elif isinstance(item, dict):
            return {k: convert_to_float32(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [convert_to_float32(v) for v in item]
        else:
            return item
    
    # Convert all items in batch to float32
    return [convert_to_float32(item) for item in batch]


@dataclass
class ArgsConfig:
    """Configuration for GR00T model fine-tuning."""

    # Dataset parameters
    dataset_path: List[str]
    """Path to the dataset directory or directories"""

    output_dir: str = "/tmp/gr00t"
    """Directory to save model checkpoints."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_only"
    """Data configuration name from DATA_CONFIG_MAP, we assume all datasets have the same data config"""

    # Training parameters
    batch_size: int = 8  # Reduced from 32 for MPS compatibility
    """Batch size per GPU for training."""

    max_steps: int = 10000
    """Maximum number of training steps."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 1000
    """Number of steps between saving checkpoints."""

    # Model parameters
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path or HuggingFace model ID for the base model."""

    tune_llm: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_visual: bool = False
    """Whether to fine-tune the vision tower."""

    tune_projector: bool = True
    """Whether to fine-tune the projector."""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the diffusion model."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    # Advanced training parameters
    learning_rate: float = 1e-4
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    lora_rank: int = 0
    """Rank for the LORA model. If 0, no LORA will be used."""

    lora_alpha: int = 16
    """Alpha value for the LORA model."""

    lora_dropout: float = 0.1
    """Dropout rate for the LORA model."""

    lora_full_model: bool = False
    """Whether to use the full model for LORA. If False, only the action head will be trained."""

    dataloader_num_workers: int = 2  # Reduced from 8 for MPS compatibility
    """Number of workers for data loading."""

    report_to: Literal["wandb", "tensorboard"] = "wandb"
    """Where to report training metrics (e.g., 'wandb', 'tensorboard')."""

    # Data loading parameters
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use for training. e.g. 'new_embodiment', 'gr1'"""

    video_backend: Literal["decord", "torchvision_av"] = "decord"
    """Video backend to use for training. [decord, torchvision_av]"""

    # Mixture dataset parameters
    balance_dataset_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, we will balance the dataset weights, by multiplying the total trajectory to each dataset"""

    # Mixture dataset parameters
    balance_trajectory_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, sample trajectories within a dataset weighted by their length; otherwise, equal weighting."""


#####################################################################################
# main training function
#####################################################################################


def main(config: ArgsConfig):
    """Main training function."""
    # Suppress additional warnings for clean output
    warnings.filterwarnings("ignore", message="`use_fast` is set to `True`")
    warnings.filterwarnings("ignore", message="Class AVF.*is implemented in both")
    
    # ------------ step 1: load dataset ------------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    # 1.1 modality configs and transforms
    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # 1.2 data loader: we will use either single dataset or mixture dataset
    if len(config.dataset_path) == 1:
        train_dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,  # This will override the dataset's embodiment tag to "new_embodiment"
            video_backend=config.video_backend,
        )
    else:
        single_datasets = []
        for p in config.dataset_path:
            assert os.path.exists(p), f"Dataset path {p} does not exist"
            ## We use the same transforms, modality configs, and embodiment tag for all datasets here,
            ## in reality, you can use dataset from different modalities and embodiment tags
            dataset = LeRobotSingleDataset(
                dataset_path=p,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
            )
            single_datasets.append(dataset)

        train_dataset = LeRobotMixtureDataset(
            data_mixture=[
                (dataset, 1.0)  # we will use equal weights for all datasets
                for dataset in single_datasets
            ],
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
            metadata_config={
                "percentile_mixing_method": "weighted_average",
            },
        )
        print(f"Loaded {len(single_datasets)} datasets, with {config.dataset_path} ")
    
    print(f"Loaded dataset with {len(train_dataset)} samples")

    # ------------ step 2: load model ------------
    # Check device type for model configuration
    device_type = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,  # backbone's LLM
        tune_visual=config.tune_visual,  # backbone's vision tower
        tune_projector=config.tune_projector,  # action head's projector
        tune_diffusion_model=config.tune_diffusion_model,  # action head's DiT
    )

    # Set the model's compute_dtype based on device capability
    if device_type == "cuda":
        model.compute_dtype = "bfloat16"
        model.config.compute_dtype = "bfloat16"
    else:
        # Use float32 for MPS and CPU
        model.compute_dtype = "float32"
        model.config.compute_dtype = "float32"

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
        )

    # 2.1 modify training args
    # Check device compatibility for training precision
    device_type = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    use_bf16 = device_type == "cuda"  # Only use bf16 on CUDA devices
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=use_bf16,
        fp16=False,  # Disable fp16 to avoid conflicts
        tf32=False,  # Disable TF32 for compatibility with non-NVIDIA GPUs
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        # evaluation_strategy="no",
        save_total_limit=8,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    # 2.2 run experiment
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )

    # 2.3 run experiment
    print("Starting training...")
    print(f"Total samples in dataset: {len(train_dataset)}")
    
    # For debugging - check a sample from the dataset
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        for key, value in sample.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                print(f"  {key}: {type(value)} shape={getattr(value, 'shape', 'N/A')} dtype={getattr(value, 'dtype', 'N/A')}")
            elif isinstance(value, dict):
                print(f"  {key}: dict with {len(value)} items")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (np.ndarray, torch.Tensor)):
                        print(f"    {sub_key}: {type(sub_value)} shape={getattr(sub_value, 'shape', 'N/A')} dtype={getattr(sub_value, 'dtype', 'N/A')}")
            else:
                print(f"  {key}: {type(value)}")
    
    experiment.train()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)

    # Print the tyro config
    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    # Check available devices (GPU/MPS/CPU)
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        available_gpus = 1  # MPS only supports one device
        device_type = "mps"
    else:
        available_gpus = 1  # CPU
        device_type = "cpu"

    # Validate GPU configuration
    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) is greater than the available devices ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} {device_type} device(s)")

    if config.num_gpus == 1:
        # Single device mode
        if device_type == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Run the script normally
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            # Multi-GPU mode - use torchrun (only for CUDA)
            if device_type != "cuda":
                raise ValueError("Multi-GPU training is only supported with CUDA devices")
            script_path = Path(__file__).absolute()
            # Remove any existing CUDA_VISIBLE_DEVICES from environment
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            # Use subprocess.run instead of os.system
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",  # default to 1 node for now
                str(script_path),
            ]

            # Convert config to command line arguments
            for key, value in vars(config).items():
                if isinstance(value, bool):
                    # For boolean values, use --flag or --no-flag format
                    if value:
                        cmd.append(f"--{key.replace('_', '-')}")
                    else:
                        cmd.append(f"--no-{key.replace('_', '-')}")
                else:
                    # For non-boolean values, use --key value format
                    cmd.append(f"--{key.replace('_', '-')}")

                    # if the value is a list (e.g. dataset_path), we need to add each element in the list
                    if isinstance(value, list):
                        for v in value:
                            cmd.append(str(v))
                    else:
                        cmd.append(str(value))
            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)

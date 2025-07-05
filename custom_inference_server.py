#!/usr/bin/env python3

"""
Custom inference service for SO101 model with wrist camera only
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from custom_data_config import CustomSo101DataConfig
from gr00t.eval.robot import RobotInferenceServer
from gr00t.model.policy import Gr00tPolicy


def main():
    # Configuration
    model_path = "so101-checkpoints"
    embodiment_tag = "new_embodiment"
    port = 5555
    denoising_steps = 4
    
    print(f"Starting custom SO101 inference server...")
    print(f"Model path: {model_path}")
    print(f"Embodiment tag: {embodiment_tag}")
    print(f"Port: {port}")
    
    # Create custom data config
    data_config = CustomSo101DataConfig()
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    
    print("Creating GR00T policy...")
    policy = Gr00tPolicy(
        model_path=model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=embodiment_tag,
        denoising_steps=denoising_steps,
    )
    
    print("Starting server...")
    server = RobotInferenceServer(policy, port=port)
    print(f"Server ready and listening on tcp://0.0.0.0:{port}")
    server.run()


if __name__ == "__main__":
    main()

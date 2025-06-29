#!/usr/bin/env python3

"""
Test client for custom SO101 inference server
"""

import numpy as np
import time
from gr00t.eval.robot import RobotInferenceClient


def main():
    host = "localhost"
    port = 5555
    
    print(f"Connecting to inference server at {host}:{port}")
    
    # Create client
    client = RobotInferenceClient(host=host, port=port)
    
    print("Getting modality config...")
    modality_configs = client.get_modality_config()
    print("Available modality configs:")
    for key, config in modality_configs.items():
        print(f"  {key}: {config}")
    
    # Create test observation data
    # Based on the error message, we need:
    # - video.wrist: video data
    # - state.single_arm: arm state (5 dimensions based on metadata)
    # - state.gripper: gripper state (1 dimension)
    # - annotation.human.task_description: task description
    
    print("\nCreating test observation...")
    obs = {
        "video.wrist": np.random.randint(0, 256, (1, 1, 480, 640, 3), dtype=np.uint8),  # T, H, W, C format
        "state.single_arm": np.random.rand(1, 5),  # 5 dimensions for single arm
        "state.gripper": np.random.rand(1, 1),     # 1 dimension for gripper
        "annotation.human.task_description": ["pick up the object"],
    }
    
    print("Observation shapes:")
    for key, value in obs.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {value}")
    
    print("\nSending observation to server...")
    time_start = time.time()
    try:
        action = client.get_action(obs)
        duration = time.time() - time_start
        print(f"✅ Received action in {duration:.3f} seconds")
        
        print("\nAction shapes:")
        for key, value in action.items():
            print(f"  {key}: {value.shape}")
            
    except Exception as e:
        print(f"❌ Error getting action: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

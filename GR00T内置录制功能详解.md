# GR00T 内置录制功能详解

GR00T 确实内置了完整的数据录制功能，基于 LeRobot 框架提供。本文档详细介绍如何使用这些录制功能。

## 1. 主要录制模块

### 1.1 核心录制脚本
- **主脚本**: `/hdd/dps/Isaac-GR00T/lerobot/lerobot/record.py`
- **功能**: 提供完整的机器人数据录制功能，支持遥操作或策略控制
- **调用方式**: `python -m lerobot.record`

### 1.2 其他录制工具
- **摄像头查找**: `/hdd/dps/Isaac-GR00T/lerobot/lerobot/find_cameras.py`
- **摄像头录制**: `/hdd/dps/Isaac-GR00T/lerobot/benchmarks/video/capture_camera_feed.py` 
- **控制脚本**: `/hdd/dps/Isaac-GR00T/lerobot/scripts/control_robot.py`
- **示例脚本**: `/hdd/dps/Isaac-GR00T/lerobot/examples/lekiwi/record.py`

## 2. 基本录制命令

### 2.1 标准录制命令格式
```bash
python -m lerobot.record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_robot \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=${HF_USER}/my_dataset \
    --dataset.single_task="Pick and place task" \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=30 \
    --dataset.fps=30 \
    --display_data=true \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader_arm
```

### 2.2 主要参数说明

#### 机器人配置 (--robot.*)
- `--robot.type`: 机器人类型 (so100_follower, so101_follower, 等)
- `--robot.port`: 机器人连接端口 (如 /dev/ttyACM0)
- `--robot.id`: 机器人标识符
- `--robot.cameras`: 摄像头配置 (JSON格式)

#### 数据集配置 (--dataset.*)
- `--dataset.repo_id`: 数据集仓库ID (格式: username/dataset_name)
- `--dataset.single_task`: 任务描述
- `--dataset.num_episodes`: 录制集数 (默认: 50)
- `--dataset.episode_time_s`: 每集录制时长 (默认: 60秒)
- `--dataset.reset_time_s`: 重置时长 (默认: 60秒)
- `--dataset.fps`: 帧率 (默认: 30)
- `--dataset.push_to_hub`: 是否上传到Hugging Face Hub (默认: true)
- `--dataset.video`: 是否录制为视频 (默认: true)

#### 遥操作配置 (--teleop.*)
- `--teleop.type`: 遥操作设备类型 (so100_leader, so101_leader, 等)
- `--teleop.port`: 遥操作设备端口
- `--teleop.id`: 遥操作设备标识符

#### 显示配置
- `--display_data`: 是否显示实时数据 (使用 Rerun 可视化)
- `--play_sounds`: 是否播放语音提示

## 3. 支持的机器人类型

### 3.1 主要机器人类型
- `so100_follower`: SO-100 机械臂
- `so101_follower`: SO-101 机械臂  
- `koch_follower`: Koch 机械臂
- `aloha`: ALOHA 双臂机器人
- `stretch`: Stretch 移动机器人
- `lekiwi`: LeKiwi 移动机械臂

### 3.2 对应的遥操作设备
- `so100_leader`: SO-100 主控设备
- `so101_leader`: SO-101 主控设备
- `koch_leader`: Koch 主控设备
- `keyboard`: 键盘控制

## 4. 摄像头配置

### 4.1 OpenCV 摄像头配置
```json
{
  "front": {
    "type": "opencv",
    "index_or_path": 0,  // 摄像头索引或路径
    "width": 640,
    "height": 480,
    "fps": 30
  }
}
```

### 4.2 RealSense 摄像头配置
```json
{
  "wrist": {
    "type": "intelrealsense", 
    "serial_number_or_name": "233522074606",
    "width": 640,
    "height": 480,
    "fps": 30
  }
}
```

## 5. 录制控制操作

### 5.1 键盘控制 (录制过程中)
- **右箭头 (→)**: 提前结束当前集或重置时间，进入下一集
- **左箭头 (←)**: 取消当前集并重新录制
- **ESC**: 立即停止录制，编码视频并上传数据集

### 5.2 录制流程
1. **准备阶段**: 机器人连接和校准
2. **录制阶段**: 执行任务并记录数据
3. **重置阶段**: 环境重置准备下一集
4. **后处理**: 视频编码和数据集上传

## 6. 数据集格式和存储

### 6.1 本地存储位置
- 默认路径: `~/.cache/huggingface/lerobot/{repo_id}`
- 可通过 `--dataset.root` 指定自定义路径

### 6.2 数据集结构
```
dataset/
├── data/
│   └── chunk-*/           # Parquet 数据文件
├── videos/
│   └── chunk-*/
│       └── observation.images.*/
│           └── episode_*.mp4
├── meta/
│   ├── episodes.jsonl     # 集描述
│   ├── tasks.jsonl        # 任务描述
│   └── modality.json      # 模态配置
```

## 7. 实际录制示例

### 7.1 SO-100 机械臂录制示例
```bash
# 设置Hugging Face用户名
HF_USER=$(huggingface-cli whoami | head -n 1)

# 录制数据集
python -m lerobot.record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_so100 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader_arm \
    --dataset.repo_id=${HF_USER}/pick_place_cubes \
    --dataset.single_task="Pick up the red cube and place it in the blue box" \
    --dataset.num_episodes=20 \
    --dataset.episode_time_s=45 \
    --dataset.reset_time_s=15 \
    --display_data=true
```

### 7.2 使用策略录制 (评估模式)
```bash
python -m lerobot.record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=${HF_USER}/eval_results \
    --dataset.single_task="Evaluation run" \
    --dataset.num_episodes=10 \
    --policy.path=path/to/your/policy/checkpoint \
    --display_data=true
```

## 8. 高级功能

### 8.1 断点续传
```bash
# 使用 --resume=true 继续之前的录制
python -m lerobot.record \
    --resume=true \
    --dataset.repo_id=${HF_USER}/my_dataset \
    # ... 其他参数
```

### 8.2 不上传到Hub
```bash
# 只在本地保存，不上传
python -m lerobot.record \
    --dataset.push_to_hub=false \
    # ... 其他参数
```

### 8.3 私有数据集
```bash
# 上传为私有数据集
python -m lerobot.record \
    --dataset.private=true \
    --dataset.tags='["private", "experiment"]' \
    # ... 其他参数
```

## 9. 查找和测试摄像头

### 9.1 查找可用摄像头
```bash
# 列出所有可用摄像头
python -m lerobot.find_cameras

# 测试特定类型摄像头
python -m lerobot.find_cameras opencv
python -m lerobot.find_cameras realsense

# 录制测试图像
python -m lerobot.find_cameras --record-time-s=10
```

### 9.2 摄像头录制测试
```bash
# 录制摄像头视频流
python lerobot/benchmarks/video/capture_camera_feed.py \
    --fps=30 \
    --width=640 \
    --height=480 \
    --duration=10
```

## 10. 故障排除

### 10.1 常见问题
1. **摄像头无法连接**: 检查摄像头索引和权限
2. **机器人连接失败**: 确认端口和波特率
3. **录制fps不稳定**: 调整图像写入线程数
4. **内存不足**: 减少并发摄像头数量或降低分辨率

### 10.2 调试技巧
```bash
# 显示详细日志
python -m lerobot.record --help

# 测试机器人连接（不录制）
python -m lerobot.record \
    --dataset.num_episodes=0 \
    --display_data=true

# 使用 Rerun 可视化调试
python -m lerobot.record \
    --display_data=true
```

## 11. 与 GR00T 模型的集成

录制的数据集可以直接用于 GR00T 模型训练：

```bash
# 使用录制的数据集进行微调
python scripts/gr00t_finetune.py \
    --data_config_path="custom_data_config.py" \
    --dataset_repo_id="${HF_USER}/my_dataset" \
    --output_dir="./my_finetuned_model"
```

## 总结

GR00T 提供了完整的数据录制生态系统，支持：
- 多种机器人平台
- 灵活的摄像头配置  
- 遥操作和策略控制
- 自动数据集管理和上传
- 完整的可视化和调试工具

这个录制系统使得收集高质量的机器人演示数据变得简单和可靠。

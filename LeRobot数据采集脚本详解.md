# LeRobot 数据采集脚本详解 (record.py)

## 概述

`lerobot/lerobot/record.py` 是 LeRobot 框架的核心数据采集脚本，用于记录机器人执行任务时的多模态数据（视觉、关节状态、动作等），生成符合 LeRobot 标准格式的数据集。该脚本支持：

- **遥操作控制**：通过手柄或其他遥控设备控制机器人
- **策略控制**：使用预训练的策略模型自动控制机器人
- **混合控制**：在遥操作和策略之间切换
- **多摄像头录制**：同时记录多个摄像头的视频数据
- **实时可视化**：使用 Rerun 实时显示数据
- **自动上传**：录制完成后自动上传到 HuggingFace Hub

## 主要配置类

### 1. DatasetRecordConfig 数据集录制配置

```python
@dataclass
class DatasetRecordConfig:
    repo_id: str                    # 数据集标识符，格式：{用户名}/{数据集名}
    single_task: str                # 任务描述，例如："拿起积木放到盒子里"
    root: str | Path | None = None  # 数据集本地存储根目录
    fps: int = 30                   # 帧率限制
    episode_time_s: int = 60        # 每个回合的录制时长（秒）
    reset_time_s: int = 60          # 回合间重置环境的时长（秒）
    num_episodes: int = 50          # 总录制回合数
    video: bool = True              # 是否将帧编码为视频
    push_to_hub: bool = True        # 是否上传到 HuggingFace Hub
    private: bool = False           # 是否创建私有仓库
    tags: list[str] | None = None   # 数据集标签
    # 图像写入器配置
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
```

**重要参数说明：**
- `repo_id`：必须遵循 HuggingFace 格式，如 `"username/dataset_name"`
- `single_task`：任务描述会存储在每个数据帧中，用于条件化训练
- `fps`：建议 30fps，过高可能导致存储和处理压力
- `episode_time_s`：单回合时长，根据任务复杂度调整
- `reset_time_s`：给操作者足够时间重置环境

### 2. RecordConfig 总配置

```python
@dataclass
class RecordConfig:
    robot: RobotConfig                      # 机器人配置
    dataset: DatasetRecordConfig            # 数据集配置
    teleop: TeleoperatorConfig | None       # 遥操作器配置（可选）
    policy: PreTrainedConfig | None         # 策略配置（可选）
    display_data: bool = False              # 是否实时显示数据
    play_sounds: bool = True                # 是否播放语音提示
    resume: bool = False                    # 是否恢复现有数据集
```

## 核心函数解析

### record_loop() - 主录制循环

这是数据采集的核心函数，负责：

1. **观测获取**：调用 `robot.get_observation()` 获取当前状态
2. **动作生成**：
   - 如果有策略：调用 `predict_action()` 生成动作
   - 如果有遥操作：调用 `teleop.get_action()` 获取人类控制
3. **动作执行**：调用 `robot.send_action()` 执行动作
4. **数据记录**：将观测和动作保存到数据集
5. **可视化**：如果启用，使用 Rerun 显示数据
6. **时序控制**：使用 `busy_wait()` 保证固定帧率

```python
def record_loop(
    robot: Robot,
    events: dict,           # 键盘事件（开始/停止/重录等）
    fps: int,              # 目标帧率
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | None = None,
    policy: PreTrainedPolicy | None = None,
    control_time_s: int | None = None,  # 控制时长
    single_task: str | None = None,
    display_data: bool = False,
):
    # 主循环：获取观测 -> 生成动作 -> 执行动作 -> 记录数据
    while timestamp < control_time_s:
        observation = robot.get_observation()  # 获取多模态观测
        
        if policy is not None:
            # 策略控制：模型预测动作
            action = predict_action(observation_frame, policy, ...)
        elif teleop is not None:
            # 遥操作控制：人类操控
            action = teleop.get_action()
            
        sent_action = robot.send_action(action)  # 执行动作
        
        if dataset is not None:
            # 构建数据帧并保存
            frame = {**observation_frame, **action_frame}
            dataset.add_frame(frame, task=single_task)
```

### record() - 主入口函数

完整的录制流程：

1. **初始化**：
   - 日志系统初始化
   - 可视化系统初始化（Rerun）
   - 机器人和遥操作器连接

2. **数据集创建**：
   - 分析机器人的观测和动作特征
   - 创建或加载 LeRobotDataset
   - 启动图像写入器

3. **录制循环**：
   - 对每个回合执行 `record_loop()`
   - 处理键盘事件（重录、停止等）
   - 回合间重置环境

4. **后处理**：
   - 保存数据集
   - 上传到 HuggingFace Hub
   - 清理资源

## 使用方法

### 基本命令格式

```bash
python -m lerobot.record \
    --robot.type=机器人类型 \
    --robot.port=机器人端口 \
    --robot.cameras="摄像头配置" \
    --dataset.repo_id=数据集ID \
    --dataset.num_episodes=回合数 \
    --dataset.single_task="任务描述" \
    [--teleop.type=遥操作器类型] \
    [--policy.path=策略路径]
```

### 支持的机器人类型

- `so100_follower`：SO-100 机器人（单臂）
- `so101_follower`：SO-101 机器人（双臂）
- `koch_follower`：Koch 机器人

### 摄像头配置示例

```bash
# 单摄像头（OpenCV）
--robot.cameras="{laptop: {type: opencv, camera_index: 0, width: 640, height: 480}}"

# 多摄像头（RealSense + OpenCV）
--robot.cameras="{
    wrist_cam: {type: realsense, serial_number: 123456, width: 640, height: 480},
    top_cam: {type: opencv, camera_index: 0, width: 640, height: 480}
}"
```

### 实际使用示例

#### 1. 纯遥操作录制

```bash
python -m lerobot.record \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{laptop: {type: opencv, camera_index: 0, width: 640, height: 480}}" \
    --robot.id=black \
    --dataset.repo_id=myuser/pick_and_place_demo \
    --dataset.num_episodes=10 \
    --dataset.single_task="Pick up the red cube and place it in the blue box" \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue
```

#### 2. 策略自动录制

```bash
python -m lerobot.record \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{laptop: {type: opencv, camera_index: 0, width: 640, height: 480}}" \
    --dataset.repo_id=myuser/policy_demo \
    --dataset.num_episodes=5 \
    --dataset.single_task="Autonomous pick and place" \
    --policy.path=lerobot/lerobot_so100_pickup_lego_real
```

#### 3. 混合控制录制

```bash
python -m lerobot.record \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{laptop: {type: opencv, camera_index: 0, width: 640, height: 480}}" \
    --dataset.repo_id=myuser/hybrid_demo \
    --dataset.num_episodes=8 \
    --dataset.single_task="Mixed teleoperation and policy control" \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --policy.path=lerobot/lerobot_so100_pickup_lego_real
```

## 键盘控制

录制过程中支持以下键盘操作：

- **空格键**：开始/暂停录制
- **R 键**：重新录制当前回合
- **S 键**：停止录制并保存
- **Q 键**：退出程序
- **Esc 键**：紧急停止

## 数据格式输出

脚本会生成标准的 LeRobot 数据集结构：

```
dataset_name/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet    # 结构化数据
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       └── observation.images.{camera_name}/
│           ├── episode_000000.mp4    # 视频数据
│           ├── episode_000001.mp4
│           └── ...
└── meta/
    ├── info.json           # 数据集元信息
    ├── stats.json          # 统计信息
    ├── episodes.jsonl      # 回合信息
    └── tasks.jsonl         # 任务信息
```

## 关键技术细节

### 1. 时序同步机制

- 使用 `time.perf_counter()` 获取高精度时间戳
- `busy_wait()` 确保固定帧率执行
- 所有数据帧都有统一的时间戳，确保多模态数据对齐

### 2. 多线程图像处理

- `num_image_writer_processes`：子进程数量
- `num_image_writer_threads_per_camera`：每摄像头线程数
- 建议配置：4线程/摄像头，0进程（使用线程模式）

### 3. 内存管理

- 使用缓冲区机制避免内存溢出
- 图像数据异步写入磁盘
- 支持大规模数据集录制

### 4. 错误处理

- 自动重连机器人和摄像头
- 数据完整性检查
- 异常情况下的安全停止

## 最佳实践

### 1. 硬件配置建议

- **CPU**：至少 8 核，推荐 16 核
- **RAM**：至少 16GB，推荐 32GB
- **存储**：SSD，至少 500GB 可用空间
- **USB**：USB 3.0+ 端口用于摄像头

### 2. 录制前准备

1. 确保机器人标定正确
2. 测试所有摄像头连接
3. 检查存储空间充足
4. 设置合适的光照条件

### 3. 参数调优

- **fps**：根据任务复杂度选择，一般 15-30fps
- **episode_time_s**：确保足够完成任务，避免过长
- **num_episodes**：平衡数据量和录制时间

### 4. 质量控制

- 定期检查录制质量
- 使用 `--display_data` 实时监控
- 及时删除失败的回合数据

## 常见问题排查

### 1. 摄像头连接问题

```bash
# 检查摄像头设备
ls /dev/video*
# 测试摄像头
ffplay /dev/video0
```

### 2. 机器人连接问题

```bash
# 检查串口设备
ls /dev/tty*
# 测试串口通信
screen /dev/ttyUSB0 115200
```

### 3. 帧率不稳定

- 减少 `num_image_writer_threads_per_camera`
- 降低摄像头分辨率
- 检查系统负载

### 4. 磁盘空间不足

```bash
# 检查磁盘使用情况
df -h
# 清理临时文件
find /tmp -name "*.png" -delete
```

## 扩展开发

### 1. 添加新机器人支持

在 `lerobot/common/robots/` 下添加新的机器人配置类，继承 `Robot` 基类。

### 2. 添加新摄像头支持

在 `lerobot/common/cameras/` 下添加新的摄像头驱动，继承 `Camera` 基类。

### 3. 自定义数据预处理

可以在 `record_loop()` 中添加自定义的数据预处理逻辑。

## 总结

`lerobot/lerobot/record.py` 是一个功能完整、设计精良的机器人数据采集工具。它支持多种控制方式、多模态数据录制，并能自动生成符合 LeRobot 标准的数据集格式。通过合理配置参数和遵循最佳实践，可以高效地采集高质量的机器人学习数据。

该脚本的关键优势：
- **灵活性**：支持遥操作、策略控制和混合模式
- **可扩展性**：易于添加新的机器人和传感器支持
- **稳定性**：完善的错误处理和资源管理
- **标准化**：生成标准格式的数据集，便于后续处理和共享

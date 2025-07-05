# GR00T 模型架构与层数配置详解

## 概览

GR00T N1.5-3B 是一个多模态机器人策略模型，采用了**双脑架构**（Dual Brain），包含视觉-语言骨干网络（Backbone）和动作头（Action Head）。下面详细分析各组件的层数设置和参数配置。

## 🧠 模型总体架构

```
GR00T N1.5-3B 模型架构:
┌─────────────────────────────────────┐
│           Backbone (EAGLE)          │
│  ┌─────────────┐  ┌───────────────┐ │
│  │ Vision Model │  │ Language Model│ │ 
│  │   27 layers  │  │   12 layers   │ │
│  │   (SigLIP)   │  │   (Qwen3)     │ │
│  └─────────────┘  └───────────────┘ │
└─────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────┐
│        Action Head (Flow Matching)  │
│  ┌─────────────┐  ┌───────────────┐ │
│  │VL Self-Attn │  │ DiT Diffusion │ │
│  │   4 layers  │  │   16 layers   │ │
│  └─────────────┘  └───────────────┘ │
└─────────────────────────────────────┘
```

## 🎯 1. Backbone 配置 (EAGLE)

### 1.1 Language Model (Qwen3-1.7B)

```json
"text_config": {
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 2048,                    // 隐藏层维度
    "num_hidden_layers": 28,                // 总层数: 28层
    "num_attention_heads": 16,              // 注意力头数: 16个
    "num_key_value_heads": 8,               // KV头数: 8个 (GQA)
    "intermediate_size": 6144,              // FFN中间层维度
    "head_dim": 128,                        // 每个注意力头维度
    "max_position_embeddings": 40960,       // 最大位置编码
    "attention_dropout": 0,                 // 注意力dropout
    "hidden_act": "silu",                   // 激活函数
    "rms_norm_eps": 1e-6,                   // RMS归一化参数
}
```

**但是在GR00T中实际使用：**
```python
# 在 eagle_backbone.py 中
select_layer: int = 12    # 只使用前12层!!!

# 通过以下代码实现层数裁剪
while len(self.eagle_model.language_model.model.layers) > select_layer:
    self.eagle_model.language_model.model.layers.pop(-1)
```

**✨ 关键点**: 虽然Qwen3有28层，但GR00T只使用前12层来节省计算资源！

### 1.2 Vision Model (SigLIP)

```json
"vision_config": {
    "model_type": "siglip_vision_model",
    "hidden_size": 1152,                    // 隐藏层维度
    "num_hidden_layers": 27,                // 视觉编码器层数: 27层
    "num_attention_heads": 16,              // 注意力头数: 16个
    "intermediate_size": 4304,              // FFN中间层维度
    "image_size": 224,                      // 输入图像尺寸
    "patch_size": 14,                       // patch大小
    "num_channels": 3,                      // 输入通道数
    "attention_dropout": 0,                 // 注意力dropout
    "hidden_act": "gelu_pytorch_tanh",      // 激活函数
}
```

### 1.3 Backbone 训练配置

```python
# 在 so101-checkpoints/config.json 中
"backbone_cfg": {
    "select_layer": 12,          # LLM使用前12层
    "tune_llm": false,          # 不微调语言模型
    "tune_visual": true,        # 微调视觉模型
    "project_to_dim": null,     # 投影维度（使用默认1536）
}
```

## 🚀 2. Action Head 配置

### 2.1 VL Self-Attention 模块

```json
"vl_self_attention_cfg": {
    "num_layers": 4,                        // 自注意力层数: 4层
    "num_attention_heads": 32,              // 注意力头数: 32个
    "attention_head_dim": 64,               // 每个头维度: 64
    "dropout": 0.2,                         // dropout率
    "final_dropout": true,                  // 最后层dropout
    "positional_embeddings": null,          // 位置编码
}
```

### 2.2 DiT (Diffusion Transformer) 模块

```json
"diffusion_model_cfg": {
    "num_layers": 16,                       // DiT层数: 16层 ⭐
    "num_attention_heads": 32,              // 注意力头数: 32个
    "attention_head_dim": 48,               // 每个头维度: 48
    "output_dim": 1024,                     // 输出维度
    "cross_attention_dim": 2048,            // 交叉注意力维度
    "dropout": 0.2,                         // dropout率
    "final_dropout": true,                  // 最后层dropout
    "interleave_self_attention": true,      // 交错自注意力
    "norm_type": "ada_norm",                // 归一化类型
    "positional_embeddings": null,          // 位置编码
}
```

### 2.3 Action Head 其他配置

```json
"action_head_cfg": {
    "action_dim": 32,                       // 动作维度
    "action_horizon": 16,                   // 动作预测步长
    "hidden_size": 1024,                    // 隐藏层维度
    "input_embedding_dim": 1536,            // 输入嵌入维度
    "backbone_embedding_dim": 2048,         // 骨干网络嵌入维度
    "max_action_dim": 32,                   // 最大动作维度
    "max_state_dim": 64,                    // 最大状态维度
    "num_inference_timesteps": 4,           // 推理时间步数
    "num_timestep_buckets": 1000,           // 时间步桶数
    "tune_diffusion_model": true,           // 微调扩散模型
    "tune_projector": true,                 // 微调投影器
}
```

## 📊 3. 参数统计

### 3.1 各组件参数量

根据代码中的打印信息：

```python
# DiT 模块参数量
print("Total number of DiT parameters: ", 
      sum(p.numel() for p in self.parameters() if p.requires_grad))
# 输出: Total number of DiT parameters: 550386688  (约5.5亿参数)

# VL Self-Attention 模块参数量  
print("Total number of SelfAttentionTransformer parameters: ", 
      sum(p.numel() for p in self.parameters() if p.requires_grad))
# 输出: Total number of SelfAttentionTransformer parameters: 201433088  (约2亿参数)
```

### 3.2 层数汇总表

| 组件 | 子模块 | 层数 | 注意力头数 | 头维度 | 隐藏维度 | 是否微调 |
|------|--------|------|------------|--------|----------|----------|
| **Backbone** | Vision (SigLIP) | 27 | 16 | 72 | 1152 | ✅ |
| | Language (Qwen3) | 12* | 16 | 128 | 2048 | ❌ |
| **Action Head** | VL Self-Attention | 4 | 32 | 64 | 2048 | ✅ |
| | DiT Diffusion | 16 | 32 | 48 | 1536 | ✅ |

*注：语言模型虽然原本有28层，但实际只使用前12层

## 🔧 4. 层数设计原理

### 4.1 Backbone 层数选择

```python
# Language Model: 12层 (而非原始28层)
# 原因：
# 1. 机器人任务不需要复杂的语言理解
# 2. 节省计算资源和推理速度
# 3. 避免过拟合，提高泛化能力

# Vision Model: 27层 (保持完整)
# 原因：
# 1. 视觉特征提取对机器人任务至关重要
# 2. 需要丰富的视觉表示来理解复杂场景
# 3. 支持多视角融合和细节捕捉
```

### 4.2 Action Head 层数选择

```python
# VL Self-Attention: 4层
# 原因：
# 1. 足够融合视觉和语言特征
# 2. 不会过度复杂化特征表示
# 3. 保持计算效率

# DiT Diffusion: 16层  
# 原因：
# 1. 扩散模型需要足够深度来学习动作分布
# 2. Flow Matching 需要复杂的噪声-动作映射
# 3. 支持多步动作序列生成
```

## ⚙️ 5. 训练策略

### 5.1 分组训练配置

```python
# 微调配置
tune_llm = False           # 冻结语言模型（节省GPU内存）
tune_visual = True         # 微调视觉模型（学习新视觉特征）
tune_projector = True      # 微调投影器（适配新任务）
tune_diffusion_model = True # 微调扩散模型（学习新动作分布）
```

### 5.2 计算类型配置

```python
compute_dtype = "bfloat16"  # 使用混合精度训练
model_dtype = "float32"     # 模型权重精度
```

## 🎯 6. 推理配置

### 6.1 动作生成配置

```python
action_horizon = 16         # 预测未来16步动作
num_inference_timesteps = 4 # 扩散采样4步
noise_s = 0.999            # 噪声调度参数
```

### 6.2 多实体支持

```python
max_num_embodiments = 32    # 支持最多32种机器人实体
# 通过 CategorySpecificLinear 为每种机器人学习专门的参数
```

## 🔍 7. 关键设计亮点

1. **分层微调**：只微调需要的组件，节省计算资源
2. **层数优化**：语言模型剪枝到12层，平衡性能和效率
3. **多模态融合**：4层VL自注意力充分融合视觉-语言特征
4. **强大的动作建模**：16层DiT确保高质量动作生成
5. **多实体支持**：Category-specific 设计支持多种机器人

这种精心设计的层数配置使得GR00T在保持强大功能的同时，具有良好的计算效率和可扩展性。

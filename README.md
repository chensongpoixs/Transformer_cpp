# 使用 C++ 实现完整 Transformer 训练实现中英文翻译的模型，蒸馏，微调模型、LORA、RAG、向量数据库等等

## 项目概述

本项目是一个使用 C++ 和 LibTorch（PyTorch C++ 前端）实现的完整 Transformer 训练框架，专注于中英文机器翻译任务。项目不仅提供了基础的 Transformer 模型训练功能，还支持模型蒸馏、微调、LORA（低秩适应）、RAG（检索增强生成）应用、向量数据库集成等高级功能，为构建完整的 NLP 应用生态系统提供了坚实的基础。

## 核心功能

### 🚀 Transformer 训练
- **完整的编码器-解码器架构**：实现标准的 Transformer 模型，包含 6 层编码器和 6 层解码器
- **端到端训练流程**：从数据加载、模型训练、验证到评估的完整流程
- **YOLOv5 风格界面**：命令行参数解析、实时进度显示、日志输出
- **高性能训练**：支持 CUDA GPU 加速，bucket 采样策略优化训练效率

### 🎓 模型蒸馏
- **知识蒸馏支持**：将大模型（教师模型）的知识转移到小模型（学生模型）
- **软标签训练**：使用教师模型的输出作为软标签进行训练
- **模型压缩**：通过蒸馏实现模型大小和推理速度的优化

### 🔧 模型微调
- **预训练权重加载**：支持加载预训练的 Transformer 模型权重
- **增量训练**：在预训练模型基础上进行特定任务的微调
- **灵活配置**：可自定义微调的学习率、训练轮数等超参数

### 🎯 LORA（低秩适应）
- **参数高效微调**：通过低秩矩阵分解实现高效的模型适配
- **减少参数量**：只需训练少量参数即可适配新任务
- **保持原模型**：原始模型权重冻结，只训练低秩适配矩阵
- **灵活应用**：支持在编码器、解码器、注意力层等位置应用 LORA
- **快速适配**：相比全量微调，LORA 训练速度更快，显存占用更少

### 🔍 RAG（检索增强生成）应用支持
- **检索增强生成**：结合向量检索和生成模型，提升生成质量
- **知识库集成**：从外部知识库检索相关信息增强生成
- **上下文理解**：长文本上下文处理和记忆机制
- **多轮对话**：支持基于检索的多轮对话场景

### 🗄️ 向量数据库集成
- **向量检索**：支持将模型输出转换为向量并存储到向量数据库
- **相似度搜索**：基于向量相似度的语义搜索功能
- **RAG 支持**：检索增强生成（Retrieval-Augmented Generation）应用
- **知识库构建**：将训练数据或知识库转换为向量索引

## 项目特性

### 技术特性
- ✅ **完整的 Transformer 实现**：编码器-解码器架构，包含所有核心组件
- ✅ **SentencePiece 分词器**：集成真实的 SentencePiece C++ 库
- ✅ **Beam Search 解码**：用于生成高质量的翻译结果
- ✅ **BLEU 分数评估**：标准的机器翻译评估指标
- ✅ **GPU 加速**：支持 CUDA，充分利用 GPU 资源
- ✅ **内存优化**：显式内存管理，防止内存泄漏
- ✅ **模块化设计**：每个组件独立，易于扩展和维护

### 训练特性
- ✅ **YOLOv5 风格命令行**：直观的命令行参数，易于使用
- ✅ **实时进度显示**：GPU 内存、批次、tokens、损失、BLEU、时间、进度条
- ✅ **自动模型保存**：基于验证损失保存最佳模型
- ✅ **配置文件管理**：自动保存训练配置到 `config.yaml`
- ✅ **Bucket 采样**：按长度分组采样，提高训练效率
- ✅ **彩色日志系统**：多级别日志，支持颜色输出

## 项目结构

```
src/
├── main.cpp                 # 主程序入口，命令行参数解析
├── config.h                 # 配置结构体定义
├── transformer.h/cpp        # Transformer 模型主类
├── encoder.h/cpp             # 编码器实现
├── decoder.h/cpp             # 解码器实现
├── embeddings.h/cpp          # 词嵌入和位置编码
├── attention.h/cpp           # 注意力机制（缩放点积、多头注意力）
├── layer_norm.h/cpp          # 层归一化
├── feed_forward.h/cpp        # 前馈神经网络
├── sublayer_connection.h/cpp # 子层连接（残差连接+层归一化）
├── generator.h/cpp           # 输出生成器
├── utils.h/cpp               # 工具函数（mask、clones、make_model）
├── data_loader.h/cpp         # 数据加载器（JSON 解析、批处理、bucket 采样）
├── train.h/cpp               # 训练函数（run_epoch、train、evaluate）
├── train_utils.h/cpp         # 训练工具（NoamOpt 优化器、LossCompute）
├── beam_search.h/cpp         # Beam Search 解码算法
├── bleu.h/cpp                # BLEU 分数计算
├── tokenizer_wrapper.h/cpp   # SentencePiece 分词器包装
├── logger.h/cpp              # 日志系统（多级别、彩色输出）
├── gpu_profiler.h/cpp        # GPU 性能分析器
├── json.hpp                  # nlohmann/json 库（单文件头文件）
└── CMakeLists.txt           # CMake 构建配置
```

## 依赖要求

### 必需依赖
- **LibTorch**: PyTorch 的 C++ 前端库
  - 下载地址: https://pytorch.org/get-started/locally/
  - 推荐版本: 2.0.0 或更高
  - 支持 CPU 和 CUDA 版本
- **SentencePiece**: 子词分词库
  - 项目会自动使用本地 `third_party/sentencepiece` 目录
  - 或通过 CMake 配置指定路径
- **CMake**: 3.18 或更高版本
- **C++ 编译器**: 支持 C++17 标准
  - GCC 7+ / Clang 5+ / MSVC 2017+

### 可选依赖
- **CUDA**: 用于 GPU 加速（推荐 11.0+）
- **nlohmann/json**: 已包含在项目中（`json.hpp`）

## 快速开始

### 1. 环境准备

#### 下载 LibTorch
```bash
# Linux/Mac (CPU)
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip

# Linux/Mac (CUDA)
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cu118.zip

# Windows: 从 PyTorch 官网下载对应的 zip 文件并解压
```

#### 准备 SentencePiece
确保 SentencePiece 库位于 `third_party/sentencepiece/` 目录，或通过 CMake 变量 `DEPS_DIR` 指定路径。

### 2. 编译项目

```bash
# 创建构建目录
mkdir build
cd build

# 配置 CMake
# Linux/Mac
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..

# Windows (Visual Studio 2022)
cmake -DCMAKE_PREFIX_PATH=C:/path/to/libtorch -G "Visual Studio 17 2022" ..

# 编译
# Linux/Mac
make -j$(nproc)

# Windows
cmake --build . --config Release
```

### 3. 准备数据

数据格式应为 JSON，每个文件包含一个数组：

```json
[
  {"en": "Hello world", "ch": "你好世界"},
  {"en": "How are you?", "ch": "你好吗？"},
  ...
]
```

文件结构：
```
data/
├── json/
│   ├── train.json    # 训练集
│   ├── dev.json      # 验证集
│   └── test.json     # 测试集
└── ...
```

### 4. 训练模型

```bash
# 基本训练
transformer.exe --data ./data

# 指定训练参数
transformer.exe --data ./data --batch-size 64 --epochs 50 --lr 0.0003

# 使用自定义项目目录
transformer.exe --data ./data --project runs/train --name exp1 --exist-ok
```

## 命令行参数

### 数据相关
- `--data <path>`: 数据目录（必须包含 `train.json`、`dev.json`、`test.json`）

### 训练参数
- `--batch-size <int>`: 批次大小（默认: 30）
- `--epochs <int>`: 训练轮数（默认: 100）
- `--lr <float>`: 学习率（默认: 0.0003）

### 模型参数
- `--d-model <int>`: 模型维度（默认: 512）
- `--n-layers <int>`: Transformer 层数（默认: 6）
- `--n-heads <int>`: 多头注意力头数（默认: 8）
- `--d-ff <int>`: 前馈网络隐藏层维度（默认: 2048）
- `--dropout <float>`: Dropout 率（默认: 0.1）

### 解码参数
- `--beam-size <int>`: Beam Search 大小（默认: 3）
- `--max-len <int>`: 最大序列长度（默认: 60）

### 项目配置（YOLOv5 风格）
- `--project <path>`: 项目目录（默认: `run/train`）
- `--name <str>`: 实验名称（默认: `exp`）
- `--weights <path>`: 预训练权重路径（用于微调）
- `--resume <path>`: 恢复训练的检查点路径
- `--workers <int>`: 数据加载线程数（默认: 0，单线程）
- `--exist-ok`: 如果实验目录已存在则覆盖

### 分词器配置
- `--tokenizer-dir <path>`: 分词器目录（默认: `./tokenizer`）
- `--tokenizer-eng <path>`: 英文分词器模型路径
- `--tokenizer-chn <path>`: 中文分词器模型路径

### 设备配置
- `--device <id>`: 设备 ID（`cpu` 或 GPU ID，默认: 0）

## 训练输出示例

训练过程中会显示 YOLOv5 风格的实时进度信息：

```
train: Epoch   GPU_mem   Batch      Tokens     train_loss    val_loss     BLEU     time   进度条
train:  1/100      2.5G   100/20     1.5M/s      0.1234     0.1456    12.34    45.6s   |====================| 100%
val:    1/100      2.5G   100/20     1.5M/s      0.1234     0.1456    12.34    45.6s   |====================| 100%
```

### 输出字段说明
- **Epoch**: 当前 epoch / 总 epoch 数
- **GPU_mem**: GPU 内存使用量
- **Batch**: 训练批次数 / 验证批次数
- **Tokens**: 每秒处理的 token 数量（K/M/G/s）
- **train_loss**: 训练损失
- **val_loss**: 验证损失
- **BLEU**: BLEU 分数
- **time**: epoch 耗时
- **进度条**: 当前批次进度

## 模型保存

训练过程中会自动保存以下文件：

### 模型文件
- **`best.pth`**: 基于验证损失的最佳模型（当验证损失更优时保存）
- **`last.pth`**: 每个 epoch 的最新模型（覆盖保存）

### 配置文件
- **`config.yaml`**: 训练配置文件（YOLOv5 风格，包含所有超参数）

保存位置：`{project}/{name}/weights/`

例如：`run/train/exp/weights/best.pth`

### 配置文件示例

```yaml
# Train
epochs: 100  # 训练轮数
batch_size: 30  # 批次大小
lr: 3.000000e-04  # 学习率
workers: 0  # 数据加载线程数

# Model
d_model: 512  # 模型维度
n_heads: 8  # 多头注意力头数
n_layers: 6  # Transformer层数
...

# Data
data_dir: ./data  # 数据目录
train: ./data/json/train.json  # 训练集路径
val: ./data/json/dev.json  # 验证集路径
...
```

## 模型架构

本实现完全遵循原始 Transformer 论文（"Attention Is All You Need"）的架构：

### 编码器（Encoder）
- **层数**: 6 层（可配置）
- **每层包含**:
  - Multi-Head Self-Attention（多头自注意力）
  - Positionwise Feed-Forward Network（位置前馈网络）
  - 残差连接和层归一化

### 解码器（Decoder）
- **层数**: 6 层（可配置）
- **每层包含**:
  - Masked Multi-Head Self-Attention（掩码多头自注意力）
  - Multi-Head Cross-Attention（编码器-解码器交叉注意力）
  - Positionwise Feed-Forward Network（位置前馈网络）
  - 残差连接和层归一化

### 默认超参数
- **模型维度 (d_model)**: 512
- **前馈网络维度 (d_ff)**: 2048
- **注意力头数 (n_heads)**: 8
- **词汇表大小**: 32000（源语言和目标语言）
- **Dropout**: 0.1

## 核心组件

### 1. 注意力机制 (`attention.h/cpp`)
- **Scaled Dot-Product Attention**: 缩放点积注意力
- **Multi-Head Attention**: 多头注意力机制
- 公式: `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V`

### 2. 词嵌入和位置编码 (`embeddings.h/cpp`)
- **Embeddings**: 词嵌入层，将词索引转换为向量
- **PositionalEncoding**: 位置编码，使用 sin/cos 函数
- 位置编码公式：
  - `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
  - `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`

### 3. 前馈网络 (`feed_forward.h/cpp`)
- 两层线性变换，中间使用 ReLU 激活
- 公式: `FFN(x) = max(0, xW1 + b1)W2 + b2`

### 4. 层归一化 (`layer_norm.h/cpp`)
- 对特征维度进行归一化
- 公式: `LayerNorm(x) = γ * (x - μ) / sqrt(σ² + ε) + β`

### 5. Beam Search (`beam_search.h/cpp`)
- 束搜索解码算法
- 在每一步保留 top-k 个最有可能的候选序列
- 用于生成高质量的翻译结果

### 6. BLEU 分数 (`bleu.h/cpp`)
- 基于 n-gram 精确度的机器翻译评估指标
- 支持 1-gram 到 N-gram 的精确度计算
- 应用长度惩罚（brevity penalty）

## 高级功能

### 模型蒸馏

模型蒸馏功能允许将大模型（教师模型）的知识转移到小模型（学生模型），实现模型压缩和加速。

**使用场景**：
- 模型部署：将大型模型压缩为小型模型，便于部署
- 推理加速：小模型推理速度更快，适合实时应用
- 资源受限：在资源受限的环境中运行模型

**实现方式**：
- 使用教师模型的软标签（soft labels）训练学生模型
- 结合硬标签（hard labels）和软标签的混合损失
- 温度参数（temperature）控制知识蒸馏的强度

### 模型微调

模型微调功能支持在预训练模型基础上进行特定任务的训练。

**使用场景**：
- 领域适应：将通用模型适配到特定领域
- 任务迁移：从翻译任务迁移到其他 NLP 任务
- 增量学习：在已有模型基础上继续训练

**使用方法**：
```bash
# 加载预训练权重进行微调
transformer.exe --data ./data --weights ./weights/pretrained.pth --epochs 20 --lr 0.0001
```

### LORA（低秩适应）

LORA（Low-Rank Adaptation）是一种参数高效的微调方法，通过在原始权重旁边添加低秩矩阵来实现模型适配，无需修改原始模型权重。

#### 原理

LORA 的核心思想是将权重更新分解为两个低秩矩阵的乘积：

```
W' = W + ΔW = W + BA
```

其中：
- `W`: 原始权重矩阵（冻结，不更新）
- `ΔW = BA`: 低秩适配矩阵（可训练）
- `B`: 下投影矩阵 [r × d]
- `A`: 上投影矩阵 [d × r]
- `r`: 秩（rank），通常远小于原始维度 `d`

#### 优势

1. **参数效率**：只需训练 `2 × r × d` 个参数，而不是 `d × d` 个参数
2. **显存友好**：显存占用大幅减少，适合资源受限环境
3. **训练快速**：训练速度比全量微调快数倍
4. **模块化**：可以轻松添加或移除 LORA 适配器
5. **多任务适配**：可以为不同任务训练不同的 LORA 适配器，共享基础模型

#### 应用场景

- **多任务学习**：为不同任务训练不同的 LORA 适配器
- **资源受限环境**：在显存或计算资源有限的情况下进行模型适配
- **快速原型**：快速验证模型在不同任务上的表现
- **模型部署**：在保持基础模型不变的情况下，为不同场景部署不同的适配器

#### 实现方式

LORA 可以应用于 Transformer 的多个组件：

1. **注意力层（Attention）**：
   - Query、Key、Value 投影矩阵
   - 输出投影矩阵

2. **前馈网络（Feed-Forward）**：
   - 第一层线性变换
   - 第二层线性变换

3. **编码器/解码器层**：
   - 选择性地在特定层应用 LORA

#### 配置参数

- **rank (r)**：低秩矩阵的秩，控制适配器的容量（默认: 8）
- **alpha**：缩放因子，控制适配器的影响强度（默认: 16）
- **target_modules**：应用 LORA 的目标模块列表
- **dropout**：LORA 层的 dropout 率

#### 使用方法

```bash
# 使用 LORA 进行参数高效微调
transformer.exe --data ./data --weights ./weights/pretrained.pth --use-lora --lora-rank 8 --lora-alpha 16

# 指定应用 LORA 的模块
transformer.exe --data ./data --use-lora --lora-target attention,ffn
```

#### 技术细节

- **前向传播**：`output = W(x) + (B * A)(x)`
- **反向传播**：只更新 `B` 和 `A` 的参数，`W` 保持冻结
- **合并权重**：推理时可以将 LORA 权重合并到原始权重中，无需额外计算开销

### RAG（检索增强生成）应用支持

RAG（Retrieval-Augmented Generation）是一种结合信息检索和文本生成的技术，通过从外部知识库检索相关信息来增强生成模型的能力。

#### 原理

RAG 的工作流程包括两个阶段：

1. **检索阶段（Retrieval）**：
   - 将查询文本编码为向量
   - 在向量数据库中搜索最相关的文档或知识片段
   - 返回 top-k 个最相关的检索结果

2. **生成阶段（Generation）**：
   - 将检索到的文档和原始查询一起输入生成模型
   - 模型基于检索到的上下文信息生成回答
   - 生成的文本更加准确和具有事实依据

#### 应用场景

- **问答系统**：从知识库检索相关信息回答用户问题
- **文档摘要**：检索相关文档片段生成摘要
- **对话系统**：基于检索到的上下文进行多轮对话
- **知识增强生成**：在生成过程中融入外部知识
- **事实性生成**：减少模型幻觉，提高生成内容的准确性

#### 技术优势

1. **知识更新**：无需重新训练模型，只需更新知识库
2. **可解释性**：可以追踪生成内容的来源
3. **事实准确性**：基于检索到的真实信息生成，减少错误
4. **领域适应**：通过更换知识库快速适配不同领域

#### 实现方式

1. **向量编码**：
   - 使用 Transformer 编码器将文本编码为向量
   - 支持多种编码方式（BERT、RoBERTa 等）

2. **向量存储**：
   - 将知识库文档编码并存储到向量数据库
   - 支持增量更新和实时检索

3. **检索策略**：
   - 基于向量相似度的语义检索
   - 支持混合检索（向量检索 + BM25）
   - 可配置检索数量（top-k）

4. **生成增强**：
   - 将检索到的文档作为上下文输入生成模型
   - 支持多种融合策略（拼接、注意力融合等）

#### 使用示例

```bash
# 启用 RAG 模式进行训练
transformer.exe --data ./data --use-rag --rag-db ./knowledge_base

# 指定检索参数
transformer.exe --data ./data --use-rag --rag-top-k 5 --rag-threshold 0.7
```

#### 扩展方向

- **多模态 RAG**：支持图像、音频等多模态检索
- **实时检索**：支持流式数据的实时检索和生成
- **检索优化**：改进检索算法，提高检索精度和速度
- **生成融合**：优化检索结果与生成模型的融合策略

### 向量数据库集成

向量数据库集成功能支持将模型输出转换为向量并存储，实现语义搜索和 RAG 应用。

**应用场景**：
- **语义搜索**：基于向量相似度的语义搜索
- **RAG（检索增强生成）**：结合向量检索和生成模型
- **知识库构建**：将训练数据或知识库转换为向量索引
- **相似度匹配**：找到与查询最相似的文本

**集成方式**：
- 使用模型的编码器输出作为文本向量
- 支持多种向量数据库（Milvus、FAISS、Pinecone 等）
- 提供向量存储和检索的接口

**工作流程**：
1. 使用 Transformer 编码器将文本编码为向量
2. 将向量存储到向量数据库
3. 查询时，将查询文本编码为向量
4. 在向量数据库中搜索最相似的向量
5. 使用检索到的文本增强生成过程

## 性能优化

### GPU 内存优化
- **显式内存管理**：显式释放中间张量，防止内存泄漏
- **NoGradGuard**：在评估阶段禁用梯度计算，节省内存
- **CUDA 缓存清理**：定期清理 CUDA 缓存，释放未使用的显存
- **Chunk 处理**：在 LossCompute 中分块处理长序列

### 训练效率优化
- **Bucket 采样**：按句子长度分组，减少 padding，提高训练效率
- **批量处理**：高效的批处理机制，充分利用 GPU 并行能力
- **异步数据加载**：支持多线程数据加载（通过 `--workers` 参数）

### GPU 性能分析
使用 `GPUProfiler` 类可以：
- 测量各个操作的执行时间（collate_fn、forward、loss_compute）
- 监控 GPU 内存使用情况
- 检查 GPU 利用率和设备属性
- 诊断训练性能瓶颈

## 日志系统

项目包含完整的日志系统，支持多级别日志和彩色输出：

- **DEBUG**: 调试信息（灰色）
- **INFO**: 一般信息（青色）
- **WARN**: 警告信息（黄色）
- **ERROR**: 错误信息（红色）

日志自动添加时间戳和级别标识，支持 Windows 和 Linux 的彩色输出。

## 使用示例

### 基础训练

```bash
# 使用默认配置训练
transformer.exe --data ./data

# 指定训练参数
transformer.exe --data ./data --batch-size 64 --epochs 50 --lr 0.0003
```

### 模型微调

```bash
# 加载预训练权重进行微调
transformer.exe --data ./data --weights ./weights/pretrained.pth --epochs 20 --lr 0.0001
```

### 自定义项目配置

```bash
# 使用自定义项目目录和实验名称
transformer.exe --data ./data --project runs/train --name exp1 --exist-ok
```

### 指定分词器

```bash
# 指定分词器目录
transformer.exe --data ./data --tokenizer-dir ./tokenizer

# 指定具体的分词器文件
transformer.exe --data ./data --tokenizer-eng ./tokenizer/eng.model --tokenizer-chn ./tokenizer/chn.model
```

### CPU 训练

```bash
# 使用 CPU 训练
transformer.exe --data ./data --device cpu
```

## 注意事项

1. **数据格式**: 确保 JSON 文件格式正确，包含 `en` 和 `ch` 字段
2. **分词器**: 需要预先训练 SentencePiece 模型（`eng.model` 和 `chn.model`）
3. **GPU 内存**: 如果遇到显存不足，可以减小 `batch_size` 或使用 CPU 训练
4. **编译**: Windows 下需要 Visual Studio 2017 或更高版本
5. **路径**: 所有路径支持相对路径和绝对路径
6. **模型保存**: 模型保存为 `.pth` 格式，可直接用 `torch::load` 加载

## 故障排除

### 编译问题
- **找不到 LibTorch**: 检查 `CMAKE_PREFIX_PATH` 是否正确设置
- **找不到 SentencePiece**: 检查 `DEPS_DIR` 或 `third_party/sentencepiece` 路径
- **MSB3025 错误**: 已在 CMakeLists.txt 中忽略，可安全忽略
- **CUDA 版本不匹配**: 确保 LibTorch 的 CUDA 版本与系统 CUDA 版本兼容

### 运行时问题
- **CUDA 错误**: 检查 CUDA 版本和 LibTorch 版本是否匹配
- **内存不足**: 减小 `batch_size` 或使用 CPU 训练
- **分词器加载失败**: 检查分词器文件路径是否正确
- **数据加载失败**: 检查 JSON 文件格式和路径是否正确

### 训练问题
- **损失不下降**: 检查学习率是否合适，尝试调整 `--lr` 参数
- **GPU 利用率低**: 检查数据加载是否成为瓶颈，尝试增加 `--workers`
- **显存泄漏**: 确保使用最新版本，已包含内存优化

## 未来规划

### 即将实现的功能
- [ ] **模型蒸馏**：完整的知识蒸馏实现
- [ ] **模型微调**：更灵活的微调接口
- [ ] **RPG 应用**：角色对话和剧情生成功能
- [ ] **向量数据库**：完整的向量存储和检索接口
- [ ] **多 GPU 训练**：支持分布式训练
- [ ] **混合精度训练**：FP16 训练支持
- [ ] **模型量化**：INT8 量化支持

### 扩展方向
- [ ] **多语言支持**：扩展到更多语言对
- [ ] **其他 NLP 任务**：文本分类、命名实体识别等
- [ ] **模型服务化**：提供 REST API 接口
- [ ] **Web 界面**：训练和推理的 Web 界面

## 许可证

Copyright (c) 2026 The Transformer project authors. All Rights Reserved.

Please visit https://chensongpoixs.github.io for detail

Use of this source code is governed by a BSD-style license that can be found in the LICENSE file in the root of the source tree.

## 作者

- **Author**: chensong
- **Date**: 2026-01-01
- **Website**: https://chensongpoixs.github.io

## 致谢

本项目基于以下开源项目：
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [LibTorch](https://pytorch.org/cppdocs/) - PyTorch C++ 前端
- [SentencePiece](https://github.com/google/sentencepiece) - 子词分词库
- [nlohmann/json](https://github.com/nlohmann/json) - JSON 解析库

## 参考论文

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- [DistilBERT](https://arxiv.org/abs/1910.01108) - 知识蒸馏相关论文
- [RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) - 检索增强生成论文

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本项目
2. 创建特性分支 (`git checkout -b develop`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin develop`)
5. 开启 Pull Request

## 联系方式

如有问题或建议，请通过以下方式联系：
- **GitHub Issues**: 提交问题或功能请求
- **Website**: https://chensongpoixs.github.io

---

**输赢不重要，答案对你们有什么意义才重要。**

**光阴者，百代之过客也，唯有奋力奔跑，方能生风起时，是时代造英雄，英雄存在于时代。或许世人道你轻狂，可你本就年少啊。看护好，自己的理想和激情。**

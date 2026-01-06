# Transformer C++ 实现

这是基于Python版本Transformer训练代码的C++实现，使用LibTorch（PyTorch C++前端）构建。

## 项目结构

```
src/
├── main.cpp                 # 主程序入口
├── config.h                 # 配置类定义
├── transformer.h/cpp        # Transformer模型主类
├── encoder.h/cpp             # 编码器实现
├── decoder.h/cpp             # 解码器实现
├── embeddings.h/cpp          # 词嵌入和位置编码
├── attention.h/cpp           # 注意力机制
├── layer_norm.h/cpp          # 层归一化
├── feed_forward.h/cpp        # 前馈网络
├── sublayer_connection.h/cpp # 子层连接
├── generator.h/cpp           # 生成器
├── utils.h/cpp               # 工具函数
└── CMakeLists.txt           # CMake构建文件
```

## 依赖要求

- **LibTorch**: PyTorch的C++前端库
  - 下载地址: https://pytorch.org/get-started/locally/
  - 选择适合你系统的版本（CPU或CUDA版本）

- **CMake**: 3.18或更高版本
- **C++编译器**: 支持C++17标准（GCC 7+, Clang 5+, MSVC 2017+）

## 编译步骤

### 1. 下载LibTorch

从PyTorch官网下载LibTorch，解压到合适的位置，例如：
```bash
# Linux/Mac
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip

# Windows
# 下载对应的zip文件并解压
```

### 2. 配置CMake

```bash
mkdir build
cd build

# Linux/Mac
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..

# Windows (使用Visual Studio)
cmake -DCMAKE_PREFIX_PATH=C:/path/to/libtorch -G "Visual Studio 16 2019" ..
```

### 3. 编译

```bash
# Linux/Mac
make -j4

# Windows
cmake --build . --config Release
```

### 4. 运行

```bash
# Linux/Mac
./transformer

# Windows
Release\transformer.exe
```

## 模型架构

本实现完全遵循原始Transformer论文（"Attention Is All You Need"）的架构：

- **编码器**: 6层，每层包含多头自注意力和前馈网络
- **解码器**: 6层，每层包含自注意力、编码器-解码器注意力和前馈网络
- **模型维度**: 512
- **前馈网络维度**: 2048
- **注意力头数**: 8
- **词汇表大小**: 32000（源语言和目标语言）

## 主要特性

1. **模块化设计**: 每个组件都是独立的模块，易于理解和修改
2. **与Python版本对齐**: 实现逻辑与Python版本保持一致
3. **支持CUDA**: 可以使用GPU加速训练和推理
4. **完整实现**: 包含所有核心组件（注意力、前馈网络、层归一化等）

## 使用示例

```cpp
#include "transformer.h"
#include "config.h"

// 创建模型
auto model = make_model(
    32000,  // src_vocab_size
    32000,  // tgt_vocab_size
    6,      // n_layers
    512,    // d_model
    2048,   // d_ff
    8,      // n_heads
    0.1f    // dropout
);

// 移动到GPU（如果可用）
model->to(torch::kCUDA);

// 前向传播
auto output = model->forward(src, tgt, src_mask, tgt_mask);
```

## 注意事项

1. **数据加载**: 当前实现不包含数据加载部分，需要根据实际需求实现数据加载器
2. **训练循环**: 主程序仅包含模型创建和简单测试，完整的训练循环需要实现优化器和损失函数
3. **分词器**: 需要集成SentencePiece或其他分词器来处理文本数据

## 与Python版本的对应关系

| Python模块 | C++文件 |
|-----------|---------|
| `model/tf_model.py` | `transformer.h/cpp` |
| `model/tf_model.py::Embeddings` | `embeddings.h/cpp` |
| `model/tf_model.py::MultiHeadedAttention` | `attention.h/cpp` |
| `model/tf_model.py::Encoder/Decoder` | `encoder.h/cpp`, `decoder.h/cpp` |
| `config.py` | `config.h` |

## 后续开发

- [ ] 实现数据加载器（DataLoader）
- [ ] 实现训练循环和优化器
- [ ] 实现损失函数计算
- [ ] 集成SentencePiece分词器
- [ ] 实现模型保存和加载
- [ ] 实现Beam Search解码
- [ ] 性能优化和并行化

## 许可证

与主项目保持一致。


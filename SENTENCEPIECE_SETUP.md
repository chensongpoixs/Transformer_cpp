# SentencePiece C++库集成指南

本文档说明如何在C++项目中集成和使用真正的SentencePiece库。

## 1. 安装SentencePiece C++库

### Ubuntu/Debian系统

```bash
# 安装依赖
sudo apt-get update
sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev

# 克隆SentencePiece仓库
git clone https://github.com/google/sentencepiece.git
cd sentencepiece

# 构建和安装
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig -v
```

### macOS系统

```bash
# 使用Homebrew安装
brew install sentencepiece

# 或者从源码编译
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
sudo make install
```

### Windows系统

```bash
# 使用vcpkg安装（推荐）
vcpkg install sentencepiece

# 或者从源码编译
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=install
cmake --build . --config Release
cmake --install . --config Release
```

## 2. 配置CMake

### 方法1：使用CMake选项启用

```bash
cd vibe-coding-cn/src
mkdir build && cd build
cmake .. -DUSE_SENTENCEPIECE=ON
make
```

### 方法2：手动指定路径

如果SentencePiece安装在非标准位置，可以手动指定：

```bash
cmake .. -DUSE_SENTENCEPIECE=ON \
         -DSENTENCEPIECE_INCLUDE_DIR=/path/to/sentencepiece/include \
         -DSENTENCEPIECE_LIB=/path/to/sentencepiece/lib/libsentencepiece.so
```

## 3. 验证安装

编译后，检查是否成功链接：

```bash
# 检查可执行文件是否链接了SentencePiece
ldd transformer | grep sentencepiece

# 或者在macOS上
otool -L transformer | grep sentencepiece
```

## 4. 使用

### 代码中的使用

代码会自动检测是否定义了`USE_SENTENCEPIECE`宏：

- **如果定义了**：使用真正的SentencePiece库
- **如果未定义**：使用简化的字符级编码（fallback）

### 运行时

确保模型文件在正确的位置：

```bash
# 模型文件路径
./tokenizer/eng.model  # 英文模型
./tokenizer/chn.model  # 中文模型
```

## 5. 故障排除

### 问题1：找不到SentencePiece库

**错误信息**：
```
CMake Error: Could not find sentencepiece
```

**解决方法**：
1. 确认已安装SentencePiece库
2. 检查库文件位置：`find /usr -name "libsentencepiece*"`
3. 如果安装在非标准位置，手动指定路径

### 问题2：运行时找不到库

**错误信息**：
```
error while loading shared libraries: libsentencepiece.so.0: cannot open shared object file
```

**解决方法**：
```bash
# 添加库路径到系统配置
echo "/usr/local/lib" | sudo tee -a /etc/ld.so.conf
sudo ldconfig

# 或者设置环境变量
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

### 问题3：头文件找不到

**错误信息**：
```
fatal error: sentencepiece_processor.h: No such file or directory
```

**解决方法**：
1. 确认头文件位置：`find /usr -name "sentencepiece_processor.h"`
2. 在CMakeLists.txt中手动添加include路径

## 6. 性能对比

使用真正的SentencePiece库相比简化版本的优势：

1. **准确性**：使用训练好的模型，分词更准确
2. **性能**：优化的C++实现，速度更快
3. **功能**：支持完整的SentencePiece功能（BPE、Unigram等）

## 7. 测试

编译后运行程序，检查是否使用真正的SentencePiece：

```bash
./transformer
```

如果看到以下信息，说明使用了简化版本：
```
警告: SentencePiece模型文件不存在: ./tokenizer/eng.model, 使用简化模式
```

如果成功加载模型，说明使用了真正的SentencePiece库。

## 8. 模型文件

确保使用与Python版本相同的模型文件：

- `./tokenizer/eng.model` - 英文模型
- `./tokenizer/chn.model` - 中文模型

这些文件应该与Python训练代码生成的模型文件相同。

## 9. API对应关系

C++ API与Python API的对应关系：

| Python | C++ |
|--------|-----|
| `sp.Load("model.model")` | `processor_->Load("model.model")` |
| `sp.EncodeAsIds(text)` | `processor_->Encode(text, &ids)` |
| `sp.DecodeIds(ids)` | `processor_->Decode(ids, &text)` |
| `sp.pad_id()` | `processor_->pad_id()` |
| `sp.bos_id()` | `processor_->bos_id()` |
| `sp.eos_id()` | `processor_->eos_id()` |

## 10. 开发建议

1. **开发阶段**：可以使用简化版本快速开发
2. **生产环境**：必须使用真正的SentencePiece库
3. **CI/CD**：在构建脚本中检查SentencePiece是否可用


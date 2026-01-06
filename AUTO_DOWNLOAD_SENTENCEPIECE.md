# SentencePiece自动下载功能说明

## 功能概述

CMakeLists.txt现在支持自动从GitHub下载并编译SentencePiece库。如果系统未安装SentencePiece，构建系统会自动下载源码并编译。

## 使用方法

### 方法1：自动下载（推荐）

```bash
cd vibe-coding-cn/src
mkdir build && cd build
cmake .. -DUSE_SENTENCEPIECE=ON -DSENTENCEPIECE_DOWNLOAD=ON
make
```

或者直接使用默认设置（默认启用自动下载）：

```bash
cd vibe-coding-cn/src
mkdir build && cd build
cmake .. -DUSE_SENTENCEPIECE=ON
make
```

### 方法2：仅使用系统安装的版本

如果不想自动下载，可以禁用下载功能：

```bash
cmake .. -DUSE_SENTENCEPIECE=ON -DSENTENCEPIECE_DOWNLOAD=OFF
```

### 方法3：使用简化版本

如果不想使用SentencePiece：

```bash
cmake .. -DUSE_SENTENCEPIECE=OFF
```

## 工作流程

1. **首先查找系统安装的版本**
   - 使用pkg-config查找
   - 使用find_library查找库文件
   - 使用find_path查找头文件

2. **如果未找到且允许下载**
   - 使用FetchContent从GitHub下载SentencePiece源码
   - 自动配置并编译为静态库
   - 自动链接到项目

3. **如果都未找到**
   - 使用简化版本（字符级编码）
   - 输出警告信息

## CMake选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `USE_SENTENCEPIECE` | `ON` | 是否启用SentencePiece支持 |
| `SENTENCEPIECE_DOWNLOAD` | `ON` | 如果未找到，是否自动下载 |

## 下载的版本

默认下载SentencePiece v0.1.99（稳定版本）。可以在CMakeLists.txt中修改`GIT_TAG`来使用其他版本。

## 编译位置

下载的源码和编译产物位于：
```
build/third_party/sentencepiece/
```

## 依赖要求

自动下载和编译SentencePiece需要：

1. **Git** - 用于克隆仓库
2. **CMake 3.18+** - 用于构建
3. **C++编译器** - 支持C++11
4. **Protobuf** - SentencePiece的依赖（通常会自动处理）

### 安装依赖（Ubuntu/Debian）

```bash
sudo apt-get install git cmake build-essential libprotobuf-dev protobuf-compiler
```

### 安装依赖（macOS）

```bash
brew install git cmake protobuf
```

### 安装依赖（Windows）

使用vcpkg或手动安装：
- Git for Windows
- CMake
- Protobuf

## 首次构建时间

首次构建时，下载和编译SentencePiece可能需要几分钟时间。后续构建会使用缓存，速度更快。

## 离线构建

如果需要在离线环境中构建：

1. 预先下载SentencePiece源码：
```bash
git clone https://github.com/google/sentencepiece.git
```

2. 修改CMakeLists.txt，将`GIT_REPOSITORY`改为本地路径：
```cmake
GIT_REPOSITORY ${CMAKE_SOURCE_DIR}/third_party/sentencepiece
```

## 故障排除

### 问题1：下载失败

**错误信息**：
```
Failed to fetch sentencepiece
```

**解决方法**：
1. 检查网络连接
2. 检查Git是否安装
3. 尝试手动下载并指定本地路径

### 问题2：编译失败

**错误信息**：
```
Error building sentencepiece
```

**解决方法**：
1. 检查是否安装了所有依赖（特别是protobuf）
2. 检查C++编译器版本（需要支持C++11）
3. 查看详细错误信息：`make VERBOSE=1`

### 问题3：链接错误

**错误信息**：
```
undefined reference to sentencepiece
```

**解决方法**：
1. 确认`USE_SENTENCEPIECE=ON`
2. 检查CMake输出，确认找到了SentencePiece
3. 清理并重新构建：`rm -rf build && mkdir build`

## 性能对比

| 方式 | 构建时间 | 运行时性能 | 推荐场景 |
|------|----------|------------|----------|
| 系统安装 | 快 | 最优 | 生产环境 |
| 自动下载 | 中等（首次慢） | 最优 | 开发环境 |
| 简化版本 | 最快 | 较差 | 快速测试 |

## 最佳实践

1. **开发环境**：使用自动下载功能，方便快捷
2. **CI/CD**：预先安装SentencePiece，加快构建速度
3. **生产环境**：使用系统安装的版本，更稳定可靠

## 验证安装

构建完成后，检查是否成功：

```bash
# 检查CMake输出
grep -i "sentencepiece" CMakeCache.txt

# 检查可执行文件
ldd transformer | grep sentencepiece  # Linux
otool -L transformer | grep sentencepiece  # macOS
```

如果看到SentencePiece相关的输出，说明已成功链接。


# 使用本地SentencePiece配置说明

## 目录结构

项目现在使用本地 `third_party` 目录中的 SentencePiece，不再从网络下载。

### 目录位置

SentencePiece应该位于以下位置之一：

1. **推荐位置**（相对于CMakeLists.txt）：
   ```
   vibe-coding-cn/src/third_party/sentencepiece/
   ```

2. **备选位置**（相对于项目根目录）：
   ```
   vibe-coding-cn/../third_party/sentencepiece/
   ```

### 目录结构示例

```
vibe-coding-cn/
├── src/
│   ├── CMakeLists.txt
│   ├── main.cpp
│   └── third_party/
│       └── sentencepiece/
│           ├── CMakeLists.txt
│           ├── src/
│           │   └── sentencepiece_processor.h
│           └── ...
```

## 获取SentencePiece源码

### 方法1：从GitHub克隆

```bash
# 在vibe-coding-cn/src目录下
cd vibe-coding-cn/src
mkdir -p third_party
cd third_party
git clone https://github.com/google/sentencepiece.git sentencepiece
cd sentencepiece
# 可选：切换到特定版本
git checkout v0.1.99
```

### 方法2：下载ZIP文件

```bash
# 下载并解压
cd vibe-coding-cn/src
mkdir -p third_party
cd third_party
# 下载master分支的ZIP文件
# 解压后重命名为sentencepiece（小写）
```

### 方法3：从已有位置复制

```bash
# 如果你已经有SentencePiece源码
cp -r /path/to/sentencepiece vibe-coding-cn/src/third_party/sentencepiece
```

## 验证目录结构

确保以下文件存在：

```bash
# 检查CMakeLists.txt
ls vibe-coding-cn/src/third_party/sentencepiece/CMakeLists.txt

# 检查头文件
ls vibe-coding-cn/src/third_party/sentencepiece/src/sentencepiece_processor.h
```

## 重新生成CMake配置

### Windows (VS2022)

```powershell
# 删除旧的构建目录
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue

# 创建新的构建目录
New-Item -ItemType Directory -Path build
cd build

# 重新配置CMake
cmake .. -G "Visual Studio 17 2022" -A x64

# 或者使用CMake GUI重新配置
```

### Linux/macOS

```bash
# 删除旧的构建目录
rm -rf build

# 创建新的构建目录
mkdir build
cd build

# 重新配置CMake
cmake ..

# 编译
make -j4
```

## CMake配置输出

如果配置成功，你应该看到：

```
找到本地SentencePiece: D:/Work/AI/Transformer/vibe-coding-cn/src/third_party/sentencepiece
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  使用本地SentencePiece
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

配置SentencePiece编译选项...
  编译选项: 静态库, 无测试, Release模式
  源码目录: D:/Work/AI/Transformer/vibe-coding-cn/src/third_party/sentencepiece
  → 找到目标: sentencepiece_static
✓ SentencePiece配置完成！
```

## 故障排除

### 问题1：找不到本地SentencePiece

**错误信息**：
```
未找到本地SentencePiece目录
```

**解决方法**：
1. 确认目录路径正确（注意是小写：`sentencepiece`，不是 `SentencePiece`）
2. 确认目录中有 `CMakeLists.txt` 文件
3. 检查路径中的大小写（Windows可能区分大小写）

### 问题2：CMakeLists.txt不存在

**错误信息**：
```
CMake Error: Could not find CMakeLists.txt in ...
```

**解决方法**：
1. 确认克隆/下载的是完整的SentencePiece仓库
2. 检查是否解压完整
3. 确认目录名称是 `sentencepiece`（小写）

### 问题3：头文件找不到

**错误信息**：
```
fatal error: sentencepiece_processor.h: No such file or directory
```

**解决方法**：
1. 确认 `src/sentencepiece_processor.h` 文件存在
2. 检查CMake是否正确设置了包含目录
3. 重新生成CMake配置

## 优势

使用本地SentencePiece的优势：

1. **无需网络**：编译时不需要网络连接
2. **版本控制**：可以锁定特定版本
3. **离线开发**：完全离线环境也能工作
4. **自定义修改**：可以修改SentencePiece源码
5. **更快配置**：不需要下载，配置更快

## 更新SentencePiece

如果需要更新到新版本：

```bash
cd vibe-coding-cn/src/third_party/sentencepiece
git pull origin master
# 或者切换到特定版本
git checkout v0.1.99
```

然后重新运行CMake配置。


# 三个关键功能的实现说明

## 1. SentencePiece分词器集成 ✅

### 实现文件
- `tokenizer_wrapper.h/cpp` - SentencePiece分词器包装类

### 功能说明
- 提供了 `SentencePieceTokenizer` 类，封装SentencePiece C++ API
- 实现了 `encode_as_ids()` 和 `decode_ids()` 方法
- 提供了 `english_tokenizer_load()` 和 `chinese_tokenizer_load()` 函数

### 当前实现状态
- **简化实现**：当前使用基于字符的简单编码/解码作为fallback
- **完整实现**：需要链接真正的SentencePiece C++库

### 使用方法
```cpp
// 加载分词器
auto sp_eng = english_tokenizer_load();
auto sp_chn = chinese_tokenizer_load();

// 编码文本
std::vector<int> ids = sp_eng->encode_as_ids("Hello world");

// 解码token ID
std::string text = sp_chn->decode_ids(ids);
```

### 集成到数据加载器
- `MTDataset::collate_fn()` 现在使用SentencePiece分词器进行编码
- 自动添加BOS和EOS标记
- 自动填充到批次最大长度

### 注意事项
1. 如果安装了SentencePiece C++库，需要取消注释相关代码并链接库
2. 当前简化版本仅用于演示，实际训练需要真正的SentencePiece模型

---

## 2. Beam Search解码实现 ✅

### 实现文件
- `beam_search.h/cpp` - Beam Search解码器

### 功能说明
- `Beam` 类：管理单个beam的状态（分数、backpointers、输出序列）
- `beam_search()` 函数：执行完整的beam search解码

### 核心功能
1. **Beam状态管理**
   - 维护多个候选序列
   - 跟踪每个序列的累积分数
   - 使用backpointers回溯构建完整序列

2. **解码流程**
   - 编码源语言序列
   - 为每个batch样本创建beam
   - 迭代解码直到达到最大长度或所有beam完成
   - 动态管理活跃的beam（已完成的不再处理）

3. **优化**
   - 只处理未完成的beam
   - 动态调整tensor大小以节省内存

### 使用方法
```cpp
auto [decode_results, scores] = beam_search(
    model,           // Transformer模型
    src,             // 源语言序列 [batch_size, src_len]
    src_mask,        // 源语言mask
    max_len,         // 最大解码长度
    pad_idx,         // padding token ID
    bos_idx,         // BOS token ID
    eos_idx,         // EOS token ID
    beam_size,       // beam大小
    device           // 设备
);

// decode_results: 每个样本的候选列表（每个候选是token ID列表）
// scores: 每个候选的分数
```

### 与Python版本的对应关系
- `Beam` 类对应Python的 `Beam` 类
- `beam_search()` 函数对应Python的 `beam_search()` 函数
- 实现了相同的算法逻辑

---

## 3. BLEU分数计算 ✅

### 实现文件
- `bleu.h/cpp` - BLEU分数计算

### 功能说明
实现了标准的BLEU-4分数计算，包括：

1. **N-gram精确度计算**
   - `compute_modified_precision()` - 修正的n-gram精确度（考虑多个参考）
   - 支持1-gram到4-gram

2. **Brevity Penalty（长度惩罚）**
   - `brevity_penalty()` - 惩罚过短的翻译

3. **BLEU分数计算**
   - `compute_bleu()` - 单个句子的BLEU分数
   - `corpus_bleu()` - 语料库级别的BLEU分数

4. **分词工具**
   - `tokenize()` - 英文分词（按空格）
   - `tokenize_chinese()` - 中文分词（按字符，简化版）

### 使用方法
```cpp
// 单个句子
std::vector<std::string> candidate = {"我", "爱", "你"};
std::vector<std::vector<std::string>> references = {{"我", "喜欢", "你"}};
float bleu = compute_bleu(candidate, references, 4);

// 语料库级别
std::vector<std::vector<std::string>> candidates = {...};
std::vector<std::vector<std::vector<std::string>>> references = {...};
float corpus_bleu_score = corpus_bleu(candidates, references, 4);
```

### 集成到评估函数
- `evaluate()` 函数现在使用beam search生成翻译
- 使用SentencePiece解码器将token ID转换为文本
- 计算语料库级别的BLEU分数

### 与Python版本的对应关系
- 实现了与 `sacrebleu` 库相同的BLEU计算逻辑
- 支持BLEU-4（1-gram到4-gram的几何平均）

---

## 使用示例

### 完整的训练和评估流程

```cpp
// 1. 加载数据集（自动使用SentencePiece分词器）
MTDataset train_dataset("./data/json/train.json");
MTDataset dev_dataset("./data/json/dev.json");

// 2. 创建模型
auto model = make_model(...);

// 3. 训练
train(train_dataset, dev_dataset, model, criterion, optimizer, config, device);

// 4. 评估（自动使用beam search和BLEU计算）
float bleu_score = evaluate(dev_dataset, model, config, device);
```

---

## 注意事项

### SentencePiece
1. **当前实现**：使用简化的字符级编码作为fallback
2. **完整实现**：需要安装SentencePiece C++库并链接
3. **模型文件**：需要 `./tokenizer/eng.model` 和 `./tokenizer/chn.model`

### Beam Search
1. **内存使用**：beam search会显著增加内存使用（batch_size × beam_size）
2. **性能**：解码速度取决于beam_size和序列长度
3. **设备**：确保所有tensor在正确的设备上（GPU/CPU）

### BLEU计算
1. **分词**：当前中文分词使用字符级分割（简化版）
2. **精度**：与sacrebleu可能有细微差异（实现细节不同）
3. **性能**：语料库级别的BLEU计算可能较慢（O(n²)复杂度）

---

## 后续改进建议

1. **SentencePiece**
   - 集成真正的SentencePiece C++库
   - 支持从文件加载模型

2. **Beam Search**
   - 支持长度惩罚（length penalty）
   - 支持覆盖惩罚（coverage penalty）
   - 优化内存使用

3. **BLEU**
   - 优化计算性能
   - 支持更多BLEU变体（BLEU-1, BLEU-2等）
   - 改进中文分词（使用真正的分词器）


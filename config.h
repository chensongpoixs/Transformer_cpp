/******************************************************************************
 *  Copyright (c) 2026 The Transformer project authors . All Rights Reserved.
 *
 *  Please visit https://chensongpoixs.github.io for detail
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 ******************************************************************************/
/*****************************************************************************
				   Author: chensong
				   date:  2026-01-01
 * 配置结构体 (Configuration Structure)
 * 
 * 定义 Transformer 模型和训练的所有超参数和配置：
 * - 模型超参数：d_model, n_heads, n_layers 等
 * - 词汇表配置：vocab_size, padding_idx, bos_idx, eos_idx
 * - 训练配置：batch_size, epoch_num, lr
 * - 文件路径：数据路径、分词器路径等
 * - YOLOv5 风格配置：project, name, workers 等
				   
				   
				   
				   
 输赢不重要，答案对你们有什么意义才重要。

 光阴者，百代之过客也，唯有奋力奔跑，方能生风起时，是时代造英雄，英雄存在于时代。或许世人道你轻狂，可你本就年少啊。 看护好，自己的理想和激情。


 我可能会遇到很多的人，听他们讲好2多的故事，我来写成故事或编成歌，用我学来的各种乐器演奏它。
 然后还可能在一个国家遇到一个心仪我的姑娘，她可能会被我帅气的外表捕获，又会被我深邃的内涵吸引，在某个下雨的夜晚，她会全身淋透然后要在我狭小的住处换身上的湿衣服。
 3小时候后她告诉我她其实是这个国家的公主，她愿意向父皇求婚。我不得已告诉她我是穿越而来的男主角，我始终要回到自己的世界。
 然后我的身影慢慢消失，我看到她眼里的泪水，心里却没有任何痛苦，我才知道，原来我的心被丢掉了，我游历全世界的原因，就是要找回自己的本心。
 于是我开始有意寻找各种各样失去心的人，我变成一块砖头，一颗树，一滴水，一朵白云，去听大家为什么会失去自己的本心。
 我发现，刚出生的宝宝，本心还在，慢慢的，他们的本心就会消失，收到了各种黑暗之光的侵蚀。
 从一次争论，到嫉妒和悲愤，还有委屈和痛苦，我看到一只只无形的手，把他们的本心扯碎，蒙蔽，偷走，再也回不到主人都身边。
 我叫他本心猎手。他可能是和宇宙同在的级别 但是我并不害怕，我仔细回忆自己平淡的一生 寻找本心猎手的痕迹。
 沿着自己的回忆，一个个的场景忽闪而过，最后发现，我的本心，在我写代码的时候，会回来。
 安静，淡然，代码就是我的一切，写代码就是我本心回归的最好方式，我还没找到本心猎手，但我相信，顺着这个线索，我一定能顺藤摸瓜，把他揪出来。

 ******************************************************************************/

#ifndef TRANSFORMER_CONFIG_H
#define TRANSFORMER_CONFIG_H

#include <string>

/**
 * Transformer模型配置类
 * 包含所有超参数和路径配置
 */
struct TransformerConfig {
    // 模型超参数
    int d_model = 512;              // 模型维度
    int n_heads = 8;                // 多头注意力头数
    int n_layers = 6;               // Transformer层数
    int d_k = 64;                   // 每个头的键向量维度
    int d_v = 64;                   // 每个头的值向量维度
    int d_ff = 2048;                // 前馈网络隐藏层维度
    float dropout = 0.1f;           // Dropout率
    
    // 词汇表配置
    int src_vocab_size = 32000;     // 源语言词汇表大小
    int tgt_vocab_size = 32000;     // 目标语言词汇表大小
    int padding_idx = 0;            // Padding标记索引
    int bos_idx = 2;                // 开始符索引
    int eos_idx = 3;                // 结束符索引
    
    // 训练配置
    int batch_size = 30;            // 批次大小
    int epoch_num = 100;            // 训练轮数
    float lr = 3e-4f;               // 学习率
    
    // 解码配置
    int max_len = 60;               // 最大序列长度
    int beam_size = 3;              // Beam Search大小
    
    // 文件路径
    std::string data_dir = "./data";
    std::string train_data_path = "./data/json/train.json";
    std::string dev_data_path = "./data/json/dev.json";
    std::string test_data_path = "./data/json/test.json";
    std::string model_path = "./weights/transformer_model.pth";
    
    // 分词器路径
    std::string tokenizer_dir = "./tokenizer";  // 分词器目录
    std::string tokenizer_eng = "./tokenizer/eng.model";  // 英文分词器模型路径
    std::string tokenizer_chn = "./tokenizer/chn.model";  // 中文分词器模型路径
    
    // YOLOv5 风格配置
    std::string project = "run/train";  // 项目目录（类似 YOLOv5 的 --project）
    std::string name = "exp";           // 实验名称（类似 YOLOv5 的 --name）
    std::string weights = "";           // 预训练权重路径（类似 YOLOv5 的 --weights）
    std::string resume = "";            // 恢复训练的检查点路径（类似 YOLOv5 的 --resume）
    int workers = 0;                    // 数据加载线程数（类似 YOLOv5 的 --workers，0=单线程）
    bool pin_memory = true;             // 是否使用固定内存（pin_memory），加速 CPU->GPU 传输
    int prefetch_factor = 2;             // 每个 worker 预取的 batch 数量
    bool exist_ok = false;              // 如果实验目录已存在是否覆盖（类似 YOLOv5 的 --exist-ok）
    
    // 阶段 3：数据缓存 + 混合精度训练
    int cache_size = 2;                 // GPU 数据缓存大小（预加载的 batch 数量，默认 2）
    bool use_amp = false;               // 是否使用混合精度训练（FP16，默认 false）
    float amp_init_scale = 65536.0f;    // AMP 初始缩放因子（2^16，默认 65536）
    int amp_scale_window = 2000;        // AMP 缩放窗口（每 N 次迭代更新一次缩放因子）
    
    // CUDA Stream 配置
    bool use_cuda_stream = true;        // 是否使用 CUDA Stream 进行流水线并行（默认 true）
    int cuda_stream_count = 4;          // CUDA Stream 数量（默认 4，用于深度流水线）
    
    // 设备配置
    bool use_cuda = true;           // 是否使用CUDA
    int device_id = 0;              // GPU设备ID（或 "cpu"）
};

#endif // TRANSFORMER_CONFIG_H


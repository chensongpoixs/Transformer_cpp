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
 * 多进程数据加载器 (Multi-Process Data Loader)
 * 
 * 实现类似 PyTorch DataLoader 的多进程/多线程数据加载：
 * - 多线程 worker 池并行处理数据
 * - 线程安全的队列管理
 * - pin_memory 优化（固定内存，加速 CPU->GPU 传输）
 * - 预取多个 batch，实现流水线并行
 * 
 * 性能优化：
 * - 多线程并行：充分利用多核 CPU
 * - pin_memory：CPU->GPU 传输速度提升 3-4x
 * - 批量 tokenization：减少函数调用开销
				   
				   
				   
				   
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

#ifndef TRANSFORMER_MULTI_PROCESS_LOADER_H
#define TRANSFORMER_MULTI_PROCESS_LOADER_H

#include "data_loader.h"
#include <torch/torch.h>
#include <queue>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <vector>
#include <future>
#include "config.h"
/**
 * 多进程数据加载器
 * 参考 PyTorch DataLoader 的设计，实现多线程并行数据加载
 */
class MultiProcessDataLoader {
public:
    /**
     * 构造函数
     * @param dataset 数据集引用（注意：必须是线程安全的，或者每个 worker 有自己的副本）
     * @param indices 数据索引列表（已排序，用于 bucket 采样）
     * @param batch_size 批次大小
     * @param device 目标设备（GPU）
     * @param config 训练配置
     * @param num_workers worker 线程数（默认 4，0 表示单线程）
     * @param pin_memory 是否使用固定内存（加速 CPU->GPU 传输）
     * @param prefetch_factor 预取因子（每个 worker 预取多少个 batch，默认 2）
     */
    MultiProcessDataLoader(
        MTDataset& dataset,
        const std::vector<size_t>& indices,
        size_t batch_size,
        torch::Device device,
        const TransformerConfig& config,
        int num_workers = 4,
        bool pin_memory = true,
        int prefetch_factor = 2
    );
    
    /**
     * 析构函数：确保所有线程正确退出
     */
    ~MultiProcessDataLoader();
    
    /**
     * 获取下一个 batch
     * @return Batch 对象，如果数据加载完成返回空的 Batch（通过检查 batch.src.defined() 判断）
     */
    Batch next();
    
    /**
     * 重置数据加载器（用于下一个 epoch）
     * @param new_indices 新的索引列表（可选，如果为空则使用原来的）
     */
    void reset(const std::vector<size_t>* new_indices = nullptr);
    
    /**
     * 获取数据加载器状态
     */
    size_t size() const { return total_batches_; }
    bool empty() const { return current_batch_idx_ >= total_batches_; }

private:
    // 数据集和配置
    MTDataset& dataset_;
    std::vector<size_t> indices_;
    size_t batch_size_;
    torch::Device device_;
    TransformerConfig config_;
    
    // Worker 配置
    int num_workers_;
    bool pin_memory_;
    int prefetch_factor_;
    
    // 批次管理
    size_t total_batches_;
    std::atomic<size_t> current_batch_idx_;
    std::atomic<size_t> loaded_batch_count_;
    
    // 线程安全队列（存储已加载的 batch）
    // 使用 priority_queue 确保按 batch_idx 顺序输出
    struct BatchItem {
        size_t batch_idx;
        Batch batch;
        bool is_valid;
        
        BatchItem() : batch_idx(0),   is_valid(false) {}
        BatchItem(size_t idx, Batch b) : batch_idx(idx), batch(std::move(b)), is_valid(true) {}
        
        // 用于 priority_queue 排序（最小堆，batch_idx 小的优先）
        bool operator<(const BatchItem& other) const {
            return batch_idx > other.batch_idx;  // 注意：priority_queue 是最大堆，所以这里反转
        }
    };
    
    std::priority_queue<BatchItem> batch_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // Worker 线程管理
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_;
    std::atomic<bool> should_stop_;
    
    // Worker 线程函数
    void worker_thread_func(int worker_id);
    
    // 辅助函数：加载单个 batch
    Batch load_batch_at_index(size_t batch_idx);
    
    // 辅助函数：创建 pin_memory tensor
    torch::Tensor create_pinned_tensor(const torch::IntArrayRef& shape, torch::ScalarType dtype);
};

#endif // TRANSFORMER_MULTI_PROCESS_LOADER_H


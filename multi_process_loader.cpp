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
 * 多进程数据加载器实现 (Multi-Process Data Loader Implementation)
 * 
 * 实现多线程并行数据加载，提升 GPU 利用率
				   
				   
				   
				   
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

#include "multi_process_loader.h"
#include "logger.h"
#include <algorithm>
#include <chrono>

using namespace logging;

MultiProcessDataLoader::MultiProcessDataLoader(
    MTDataset& dataset,
    const std::vector<size_t>& indices,
    size_t batch_size,
    torch::Device device,
    const TransformerConfig& config,
    int num_workers,
    bool pin_memory,
    int prefetch_factor)
    : dataset_(dataset),
      indices_(indices),
      batch_size_(batch_size),
      device_(device),
      config_(config),
      num_workers_(num_workers > 0 ? num_workers : 1),
      pin_memory_(pin_memory && device.is_cuda()),  // 只有 GPU 才需要 pin_memory
      prefetch_factor_(prefetch_factor),
      total_batches_((indices.size() + batch_size - 1) / batch_size),
      current_batch_idx_(0),
      loaded_batch_count_(0),
      running_(true),
      should_stop_(false)
{
    // 如果 num_workers=0，使用单线程模式（不启动 worker 线程）
    if (num_workers_ > 1) {
        LOG_INFO("Multi-process data loader: num_workers=" + std::to_string(num_workers_) +
                 ", pin_memory=" + std::string(pin_memory_ ? "true" : "false") +
                 ", prefetch_factor=" + std::to_string(prefetch_factor_));
        
        // 启动 worker 线程
        worker_threads_.reserve(num_workers_);
        for (int i = 0; i < num_workers_; ++i) {
            worker_threads_.emplace_back(&MultiProcessDataLoader::worker_thread_func, this, i);
        }
    } else {
        LOG_INFO("Single-threaded data loader (num_workers=0)");
    }
}

MultiProcessDataLoader::~MultiProcessDataLoader() {
    // 停止所有 worker 线程
    should_stop_ = true;
    running_ = false;
    
    // 通知所有等待的线程
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        queue_cv_.notify_all();
    }
    
    // 等待所有 worker 线程退出
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void MultiProcessDataLoader::worker_thread_func(int worker_id) {
    // 每个 worker 线程持续加载 batch，直到所有 batch 加载完成
    while (running_ && !should_stop_) {
        // 获取下一个要加载的 batch 索引
        size_t batch_idx = loaded_batch_count_.fetch_add(1);
        
        if (batch_idx >= total_batches_) {
            // 所有 batch 已加载完成
            break;
        }
        
        // 加载 batch
        try {
            Batch batch = load_batch_at_index(batch_idx);
            
            // 将 batch 放入队列
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                batch_queue_.emplace(batch_idx, std::move(batch));
            }
            
            // 通知等待的线程
            queue_cv_.notify_one();
            
        } catch (const std::exception& e) {
            LOG_WARN("Worker " + std::to_string(worker_id) + " failed to load batch " +
                     std::to_string(batch_idx) + ": " + e.what());
            // 继续处理下一个 batch
        }
    }
}

Batch MultiProcessDataLoader::load_batch_at_index(size_t batch_idx) {
    // 计算当前 batch 的数据索引范围
    size_t start_idx = batch_idx * batch_size_;
    size_t end_idx = std::min(start_idx + batch_size_, indices_.size());
    
    std::vector<size_t> batch_indices(indices_.begin() + start_idx,
                                      indices_.begin() + end_idx);
    
    // 调用数据集的 collate_fn
    // 注意：这里使用 CPU 设备创建 batch，然后使用 pin_memory 传输到 GPU
    torch::Device cpu_device(torch::kCPU);
    Batch batch = dataset_.collate_fn(batch_indices, cpu_device,
                                       config_.padding_idx, config_.bos_idx, config_.eos_idx,
                                       config_.src_vocab_size, config_.tgt_vocab_size);
    
    // 如果启用 pin_memory 且目标设备是 GPU，将数据转移到固定内存
    if (pin_memory_ && device_.is_cuda()) {
        // 使用 non_blocking 异步传输到 GPU
        batch.src = batch.src.to(device_, /*non_blocking=*/true);
        if (batch.trg.defined()) {
            batch.trg = batch.trg.to(device_, /*non_blocking=*/true);
        }
        if (batch.trg_y.defined()) {
            batch.trg_y = batch.trg_y.to(device_, /*non_blocking=*/true);
        }
        if (batch.src_mask.defined()) {
            batch.src_mask = batch.src_mask.to(device_, /*non_blocking=*/true);
        }
        if (batch.trg_mask.defined()) {
            batch.trg_mask = batch.trg_mask.to(device_, /*non_blocking=*/true);
        }
    } else {
        // 同步传输到目标设备
        batch.src = batch.src.to(device_);
        if (batch.trg.defined()) {
            batch.trg = batch.trg.to(device_);
        }
        if (batch.trg_y.defined()) {
            batch.trg_y = batch.trg_y.to(device_);
        }
        if (batch.src_mask.defined()) {
            batch.src_mask = batch.src_mask.to(device_);
        }
        if (batch.trg_mask.defined()) {
            batch.trg_mask = batch.trg_mask.to(device_);
        }
    }
    
    return batch;
}

Batch MultiProcessDataLoader::next() {
    // 单线程模式：直接加载
    if (num_workers_ <= 1) {
        if (current_batch_idx_ >= total_batches_) {
            // 返回空的 batch
            return Batch({}, {}, torch::Tensor(), torch::Tensor(), config_.padding_idx, device_);
        }
        
        size_t batch_idx = current_batch_idx_++;
        return load_batch_at_index(batch_idx);
    }
    
    // 多线程模式：从队列中获取
    std::unique_lock<std::mutex> lock(queue_mutex_);
    
    // 等待队列中有数据或所有数据已加载完成
    while (batch_queue_.empty() && loaded_batch_count_ < total_batches_ && !should_stop_) {
        queue_cv_.wait(lock);
    }
    
    // 检查是否应该停止
    if (should_stop_) {
        return Batch({}, {}, torch::Tensor(), torch::Tensor(), config_.padding_idx, device_);
    }
    
    // 如果队列为空且所有数据已加载，返回空 batch
    if (batch_queue_.empty()) {
        return Batch({}, {}, torch::Tensor(), torch::Tensor(), config_.padding_idx, device_);
    }
    
    // 获取队列中的第一个 batch（按 batch_idx 顺序，使用 priority_queue 自动排序）
    BatchItem item = std::move(batch_queue_.top());
    batch_queue_.pop();
    lock.unlock();
    
    current_batch_idx_++;
    return std::move(item.batch);
}

void MultiProcessDataLoader::reset(const std::vector<size_t>* new_indices) {
    // 停止当前加载
    should_stop_ = true;
    running_ = false;
    
    // 清空队列
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!batch_queue_.empty()) {
            batch_queue_.pop();
        }
    }
    
    // 等待所有 worker 线程退出
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
    
    // 更新索引（如果提供）
    if (new_indices != nullptr) {
        indices_ = *new_indices;
        total_batches_ = (indices_.size() + batch_size_ - 1) / batch_size_;
    }
    
    // 重置状态
    current_batch_idx_ = 0;
    loaded_batch_count_ = 0;
    should_stop_ = false;
    running_ = true;
    
    // 重新启动 worker 线程
    if (num_workers_ > 1) {
        worker_threads_.reserve(num_workers_);
        for (int i = 0; i < num_workers_; ++i) {
            worker_threads_.emplace_back(&MultiProcessDataLoader::worker_thread_func, this, i);
        }
    }
}

torch::Tensor MultiProcessDataLoader::create_pinned_tensor(const torch::IntArrayRef& shape, torch::ScalarType dtype) {
    // 创建固定内存 tensor（pin_memory）
    // 注意：LibTorch 的 TensorOptions 支持 pinned_memory
    auto options = torch::TensorOptions()
        .dtype(dtype)
        .device(torch::kCPU)
        .pinned_memory(true);
    
    return torch::empty(shape, options);
}


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
 * GPU 数据缓存实现 (GPU Data Cache Implementation)
				   
				   
				   
				   
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

#include "data_cache.h"
#include "config.h"
#include "logger.h"


using namespace logging;

DataCache::DataCache(int cache_size, torch::Device device)
    : cache_size_(cache_size), device_(device), 
      running_(false), should_stop_(false), current_batch_idx_(0), total_batches_(0) {
    // 初始化完成
}

DataCache::~DataCache() {
    stop();
}

void DataCache::start_prefetch(MTDataset& dataset, 
                              const std::vector<size_t>& indices,
                              size_t batch_size,
                              const TransformerConfig& config) {
    if (running_) {
        return;  // 已经启动
    }
    
    total_batches_ = (indices.size() + batch_size - 1) / batch_size;
    current_batch_idx_ = 0;
    running_ = true;
    should_stop_ = false;
    
    // 启动预加载线程
    prefetch_thread_ = std::thread(&DataCache::prefetch_worker, this,
                                   std::ref(dataset), std::ref(indices), batch_size, std::ref(config));
}

void DataCache::prefetch_worker(MTDataset& dataset,
                                const std::vector<size_t>& indices,
                                size_t batch_size,
                                const TransformerConfig& config) {
    while (running_ && !should_stop_) {
        // 检查缓存是否已满
        {
            std::unique_lock<std::mutex> lock(this->cache_mutex_);
            if (this->cache_queue_.size() >= static_cast<size_t>(this->cache_size_)) {
                // 缓存已满，等待
                this->cache_cv_.wait(lock, [this] {
                    return this->cache_queue_.size() < static_cast<size_t>(this->cache_size_) || this->should_stop_;
                });
            }
        }
        
        if (this->should_stop_) {
            break;
        }
        
        // 检查是否还有数据需要加载
        size_t batch_idx = this->current_batch_idx_.fetch_add(1);
        if (batch_idx >= this->total_batches_) {
            break;  // 所有数据已加载完成
        }
        
        // 加载 batch
        Batch batch = this->load_batch(dataset, indices, batch_idx, batch_size, config);
        
        // 将 batch 添加到缓存
        {
            std::unique_lock<std::mutex> lock(this->cache_mutex_);
            this->cache_queue_.push(std::move(batch));
        }
        this->cache_cv_.notify_one();
    }
    
    this->running_ = false;
}

Batch DataCache::get_next() {
    std::unique_lock<std::mutex> lock(this->cache_mutex_);
    
    // 等待缓存中有数据
    this->cache_cv_.wait(lock, [this] {
        return !this->cache_queue_.empty() || (!this->running_ && this->cache_queue_.empty());
    });
    
    if (this->cache_queue_.empty()) {
        // 缓存为空且预加载线程已停止，返回空 batch
        return Batch();
    }
    
    Batch batch = std::move(this->cache_queue_.front());
    this->cache_queue_.pop();
    this->cache_cv_.notify_one();  // 通知预加载线程可以继续加载
    
    return batch;
}

void DataCache::stop() {
    if (!this->running_) {
        return;
    }
    
    this->should_stop_ = true;
    this->cache_cv_.notify_all();  // 唤醒所有等待的线程
    
    if (this->prefetch_thread_.joinable()) {
        this->prefetch_thread_.join();
    }
    
    this->running_ = false;
    
    // 清空缓存
    std::unique_lock<std::mutex> lock(this->cache_mutex_);
    while (!this->cache_queue_.empty()) {
        this->cache_queue_.pop();
    }
}

bool DataCache::empty() {
    std::lock_guard<std::mutex> lock(this->cache_mutex_);
    return this->cache_queue_.empty() && !this->running_;
}

size_t DataCache::size() {
    std::lock_guard<std::mutex> lock(this->cache_mutex_);
    return this->cache_queue_.size();
}

Batch DataCache::load_batch(MTDataset& dataset,
                            const std::vector<size_t>& indices,
                            size_t batch_idx,
                            size_t batch_size,
                            const TransformerConfig& config) {
    // 计算当前 batch 的索引范围
    size_t start_idx = batch_idx * batch_size;
    size_t end_idx = std::min(start_idx + batch_size, indices.size());
    
    // 收集当前 batch 的数据
    std::vector<size_t> batch_indices;
    batch_indices.reserve(end_idx - start_idx);
    for (size_t i = start_idx; i < end_idx; ++i) {
        batch_indices.push_back(indices[i]);
    }
    
    // 调用 collate_fn 组装 batch（在 CPU 上创建，然后传输到 GPU）
    torch::Device cpu_device(torch::kCPU);
    Batch batch = dataset.collate_fn(batch_indices, cpu_device,
                                     config.padding_idx, config.bos_idx, config.eos_idx,
                                     config.src_vocab_size, config.tgt_vocab_size);
    
    // 将 batch 传输到 GPU（使用 non_blocking 异步传输）
    if (this->device_.is_cuda()) {
        c10::cuda::CUDAGuard guard(this->device_);
        batch.src = batch.src.to(this->device_, true);  // non_blocking=true
        batch.trg = batch.trg.to(this->device_, true);
        batch.trg_y = batch.trg_y.to(this->device_, true);
        batch.src_mask = batch.src_mask.to(this->device_, true);
        batch.trg_mask = batch.trg_mask.to(this->device_, true);
    }
    
    return batch;
}


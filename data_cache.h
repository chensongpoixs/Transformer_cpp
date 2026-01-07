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
  GPU 数据缓存 (GPU Data Cache)
  
  阶段 3：数据缓存优化
  - 预加载多个 batch 到 GPU 内存
  - 使用 CUDA pinned memory 和异步传输
  - 减少数据加载等待时间，提高 GPU 利用率
				   
				   
				   
				   
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

#ifndef TRANSFORMER_DATA_CACHE_H
#define TRANSFORMER_DATA_CACHE_H


#include <torch/torch.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <memory>
#include <vector>
#include "data_loader.h"
 
#include "config.h"

#include <c10/cuda/CUDAGuard.h>


/**
 * GPU 数据缓存类
 * 预加载多个 batch 到 GPU 内存，减少数据加载等待时间
 */
class DataCache {
public:
    /**
     * 构造函数
     * @param cache_size 缓存大小（预加载的 batch 数量）
     * @param device 目标设备（GPU）
     */
    DataCache(int cache_size, torch::Device device);
    
    /**
     * 析构函数
     */
    ~DataCache();
    
    /**
     * 启动缓存预加载线程
     * @param dataset 数据集引用
     * @param indices 数据索引列表
     * @param batch_size 批次大小
     * @param config 训练配置
     */
    void start_prefetch(MTDataset& dataset, const std::vector<size_t>& indices, size_t batch_size, const TransformerConfig& config);
    
    /**
     * 获取下一个缓存的 batch
     * @return Batch 对象，如果缓存为空返回空的 Batch
     */
    Batch get_next();
    
    /**
     * 停止预加载线程
     */
    void stop();
    
    /**
     * 检查缓存是否为空
     */
    bool empty()  ;
    
    /**
     * 获取缓存大小
     */
    size_t size();

private:
    // 预加载函数
    void prefetch_worker(MTDataset& dataset,
        const std::vector<size_t>& indices,
        size_t batch_size,
        const TransformerConfig& config);

    // 辅助函数：加载单个 batch
    Batch load_batch(MTDataset& dataset,
        const std::vector<size_t>& indices,
        size_t batch_idx,
        size_t batch_size,
        const TransformerConfig& config);
    
    // 缓存配置
    int cache_size_;
    torch::Device device_;
    
    // 缓存队列（线程安全）
    std::queue<Batch> cache_queue_;
    mutable std::mutex cache_mutex_;  // mutable：允许在 const 方法中锁定
    std::condition_variable cache_cv_;
    
    // 预加载线程
    std::thread prefetch_thread_;
    std::atomic<bool> running_;
    std::atomic<bool> should_stop_;
    
    // 预加载状态
    std::atomic<size_t> current_batch_idx_;
    size_t total_batches_;
    
    
};

#endif // TRANSFORMER_DATA_CACHE_H


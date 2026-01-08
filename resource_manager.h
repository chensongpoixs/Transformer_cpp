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
 * 资源管理 RAII 包装类 (Resource Management RAII Wrappers)
 * 
 * 提供 RAII 模式的资源管理类，确保资源在作用域结束时自动释放：
 * - DataCacheRAII: 管理 DataCache 的生命周期，确保 stop() 被调用
 * - TensorScopeGuard: 管理张量的生命周期，确保在作用域结束时释放
 * - BatchScopeGuard: 管理 Batch 对象的生命周期，确保所有张量被释放
				   
				   
				   
				   
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

#ifndef TRANSFORMER_RESOURCE_MANAGER_H
#define TRANSFORMER_RESOURCE_MANAGER_H

#include <torch/torch.h>
#include <memory>
#include "data_cache.h"
#include "data_loader.h"

/**
 * DataCache RAII 包装类
 * 确保 DataCache 的 stop() 方法在作用域结束时被调用
 */
class DataCacheRAII {
public:
    /**
     * 构造函数
     * @param cache DataCache 指针（可以是 nullptr）
     */
    explicit DataCacheRAII(DataCache* cache) : cache_(cache) {}
    
    /**
     * 禁止拷贝构造和赋值
     */
    DataCacheRAII(const DataCacheRAII&) = delete;
    DataCacheRAII& operator=(const DataCacheRAII&) = delete;
    
    /**
     * 移动构造
     */
    DataCacheRAII(DataCacheRAII&& other) noexcept : cache_(other.cache_) {
        other.cache_ = nullptr;
    }
    
    /**
     * 移动赋值
     */
    DataCacheRAII& operator=(DataCacheRAII&& other) noexcept {
        if (this != &other) {
            release();
            cache_ = other.cache_;
            other.cache_ = nullptr;
        }
        return *this;
    }
    
    /**
     * 析构函数：自动调用 stop()
     */
    ~DataCacheRAII() {
        release();
    }
    
    /**
     * 获取底层 DataCache 指针
     */
    DataCache* get() const { return cache_; }
    
    /**
     * 重载 -> 操作符
     */
    DataCache* operator->() const { return cache_; }
    
    /**
     * 重载 * 操作符
     */
    DataCache& operator*() const { return *cache_; }
    
    /**
     * 显式释放资源（调用 stop()）
     */
    void release() {
        if (cache_) {
            cache_->stop();
            cache_ = nullptr;
        }
    }
    
    /**
     * 重置为新的 DataCache（会先释放旧的）
     */
    void reset(DataCache* cache = nullptr) {
        release();
        cache_ = cache;
    }
    
    /**
     * 检查是否有效
     */
    explicit operator bool() const { return cache_ != nullptr; }

private:
    DataCache* cache_;
};

/**
 * 张量作用域保护类
 * 确保张量在作用域结束时被释放
 */
class TensorScopeGuard {
public:
    /**
     * 构造函数：接受一个或多个张量的引用
     */
    TensorScopeGuard() = default;
    
    /**
     * 添加张量到管理列表
     */
    void add(torch::Tensor& tensor) {
        tensors_.push_back(&tensor);
    }
    
    /**
     * 析构函数：释放所有管理的张量
     */
    ~TensorScopeGuard() {
        release_all();
    }
    
    /**
     * 禁止拷贝
     */
    TensorScopeGuard(const TensorScopeGuard&) = delete;
    TensorScopeGuard& operator=(const TensorScopeGuard&) = delete;
    
    /**
     * 显式释放所有张量
     */
    void release_all() {
        for (auto* tensor : tensors_) {
            if (tensor) {
                *tensor = torch::Tensor();
            }
        }
        tensors_.clear();
    }

private:
    std::vector<torch::Tensor*> tensors_;
};

/**
 * Batch 作用域保护类
 * 确保 Batch 对象中的所有张量在作用域结束时被释放
 */
class BatchScopeGuard {
public:
    /**
     * 构造函数：接受 Batch 引用
     */
    explicit BatchScopeGuard(Batch& batch) : batch_(&batch) {}
    
    /**
     * 析构函数：释放 Batch 中的所有张量
     */
    ~BatchScopeGuard() {
        release();
    }
    
    /**
     * 禁止拷贝
     */
    BatchScopeGuard(const BatchScopeGuard&) = delete;
    BatchScopeGuard& operator=(const BatchScopeGuard&) = delete;
    
    /**
     * 显式释放 Batch 中的所有张量
     */
    void release() {
        if (batch_) {
            batch_->src = torch::Tensor();
            batch_->trg = torch::Tensor();
            batch_->trg_y = torch::Tensor();
            batch_->src_mask = torch::Tensor();
            batch_->trg_mask = torch::Tensor();
            batch_ = nullptr;
        }
    }
    
    /**
     * 获取 Batch 引用
     */
    Batch& get() const { return *batch_; }
    
    /**
     * 重载 -> 操作符
     */
    Batch* operator->() const { return batch_; }
    
    /**
     * 重载 * 操作符
     */
    Batch& operator*() const { return *batch_; }

private:
    Batch* batch_;
};

/**
 * 辅助宏：自动管理 Batch 生命周期
 * 使用示例：
 *   BATCH_GUARD(batch) {
 *       // 使用 batch
 *   }  // 自动释放
 */
#define BATCH_GUARD(batch_var) \
    BatchScopeGuard _batch_guard(batch_var)

/**
 * 辅助宏：自动管理多个张量
 * 使用示例：
 *   TENSOR_GUARD(out, batch.src, batch.trg) {
 *       // 使用张量
 *   }  // 自动释放
 */
#define TENSOR_GUARD(...) \
    TensorScopeGuard _tensor_guard; \
    [&]() { \
        torch::Tensor* _tensors[] = {&__VA_ARGS__}; \
        for (auto* t : _tensors) _tensor_guard.add(*t); \
    }()

#endif // TRANSFORMER_RESOURCE_MANAGER_H


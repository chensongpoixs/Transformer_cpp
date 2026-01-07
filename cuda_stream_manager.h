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
 * CUDA Stream 管理器 (CUDA Stream Manager)
 * 
 * 用于管理多个 CUDA Stream，实现流水线并行：
 * - 数据传输 Stream：用于 CPU->GPU 数据传输
 * - 计算 Stream：用于 GPU 计算（前向传播、反向传播）
 * 
 * 流水线并行原理：
 * 1. 在 Stream 0 上传输 batch N 的数据
 * 2. 在 Stream 1 上计算 batch N-1 的前向/反向传播
 * 3. 让数据传输和计算重叠，提高 GPU 利用率
				   
				   
				   
				   
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

#ifndef TRANSFORMER_CUDA_STREAM_MANAGER_H
#define TRANSFORMER_CUDA_STREAM_MANAGER_H

#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <memory>

/**
 * CUDA Stream 管理器
 * 用于实现流水线并行，提高 GPU 利用率
 */
class CudaStreamManager {
public:
    /**
     * 构造函数
     * @param device GPU 设备
     * @param num_streams Stream 数量（默认 2：1 个用于数据传输，1 个用于计算）
     */
    explicit CudaStreamManager(torch::Device device, int num_streams = 2);
    
    /**
     * 析构函数
     */
    ~CudaStreamManager();
    
    /**
     * 获取数据传输 Stream
     * 用于 CPU->GPU 数据传输
     */
    at::cuda::CUDAStream& get_transfer_stream() {
        return *transfer_stream_;
    }
    
    /**
     * 获取计算 Stream
     * 用于 GPU 计算（前向传播、反向传播）
     */
    at::cuda::CUDAStream& get_compute_stream() {
        return *compute_stream_;
    }
    
    /**
     * 获取指定索引的 Stream
     * @param index Stream 索引（0 为数据传输，1 为计算）
     */
    at::cuda::CUDAStream& get_stream(int index) {
        return *streams_[index];
    }
    
    /**
     * 获取 Stream 数量
     */
    int num_streams() const {
        return static_cast<int>(streams_.size());
    }
    
    /**
     * 同步所有 Stream
     * 等待所有 Stream 上的操作完成
     */
    void synchronize_all();
    
    /**
     * 同步指定 Stream
     * @param index Stream 索引
     */
    void synchronize(int index);
    
    /**
     * 设置当前 Stream（用于后续操作）
     * @param index Stream 索引
     */
    void set_current_stream(int index);
    
    /**
     * 获取设备
     */
    torch::Device device() const {
        return device_;
    }

private:
    torch::Device device_;
    // 使用 at::cuda::CUDAStream（等价于旧版 torch::cuda::Stream）
    std::vector<std::unique_ptr<at::cuda::CUDAStream>> streams_;
    at::cuda::CUDAStream* transfer_stream_ = nullptr;  // 数据传输 Stream（指向 streams_ 中的元素）
    at::cuda::CUDAStream* compute_stream_ = nullptr;   // 计算 Stream（指向 streams_ 中的元素）
};

#endif // TRANSFORMER_CUDA_STREAM_MANAGER_H


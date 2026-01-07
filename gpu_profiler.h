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
 * GPU 性能分析器 (GPU Profiler)
 * 
 * 提供 GPU 性能分析和监控功能：
 * - 测量各个操作（collate_fn, forward, loss_compute）的执行时间
 * - 监控 GPU 内存使用情况
 * - 检查 GPU 利用率和设备属性
 * 
 * 用于诊断训练性能瓶颈和 GPU 资源使用情况
				   
				   
				   
				   
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

#ifndef TRANSFORMER_GPU_PROFILER_H
#define TRANSFORMER_GPU_PROFILER_H

#include <torch/torch.h>
#include <string>
#include <chrono>
#include <map>
#include <vector>
#include <c10/cuda/CUDACachingAllocator.h>

/**
 * GPU性能分析工具
 * 用于排查GPU使用不足问题
 */
class GPUProfiler {
public:
    struct TimingInfo {
        std::string name;
        double total_time_ms = 0.0;
        int count = 0;
        double min_time_ms = std::numeric_limits<double>::max();
        double max_time_ms = 0.0;
    };

    static void start_timer(const std::string& name);
    static void end_timer(const std::string& name);
    static void reset();
    static void print_summary();
    
    // GPU利用率检查
    static void check_gpu_utilization(torch::Device device);
    
    // 检查张量是否在GPU上
    static bool is_on_gpu(const torch::Tensor& tensor);
    
    // 获取GPU内存使用情况
    static void print_gpu_memory(torch::Device device);
    
    // 获取GPU内存使用情况（返回格式化字符串，用于进度条显示）
    static std::string get_gpu_memory_str(torch::Device device);
    
    // 内存统计结构体（兼容 torch::cuda::memory_stats 的返回格式）
    struct MemoryStats {
        size_t allocated_bytes_current;  // 当前已分配的字节数
        size_t reserved_bytes_current;   // 当前已保留的字节数
        size_t allocated_bytes_peak;    // 峰值已分配的字节数
        size_t reserved_bytes_peak;     // 峰值已保留的字节数
    };
    
    // 获取GPU内存统计信息（使用 c10::cuda::CUDACachingAllocator::getMemoryStats）
    // 替代 torch::cuda::memory_stats，提供更底层和准确的内存信息
    static MemoryStats get_memory_stats(torch::Device device);

private:
    static std::map<std::string, std::chrono::steady_clock::time_point> start_times_;
    static std::map<std::string, TimingInfo> timings_;
    static std::mutex mutex_;
};

#endif // TRANSFORMER_GPU_PROFILER_H


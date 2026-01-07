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
 * GPU 性能分析器实现 (GPU Profiler Implementation)
 * 
 * 实现 GPU 性能分析和监控功能：
 * - 测量各个操作的执行时间
 * - 监控 GPU 内存使用情况
 * - 检查 GPU 利用率和设备属性
				   
				   
				   
				   
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

#include "gpu_profiler.h"
#include "logger.h"
#include <iomanip>
#include <sstream>
#include <mutex>
#include <limits>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#ifdef _WIN32
#include <windows.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace logging;

std::map<std::string, std::chrono::steady_clock::time_point> GPUProfiler::start_times_;
std::map<std::string, GPUProfiler::TimingInfo> GPUProfiler::timings_;
std::mutex GPUProfiler::mutex_;

void GPUProfiler::start_timer(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    start_times_[name] = std::chrono::steady_clock::now();
}

void GPUProfiler::end_timer(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = start_times_.find(name);
    if (it == start_times_.end()) {
        return;
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - it->second).count() / 1000.0; // 转换为毫秒
    
    auto& timing = timings_[name];
    timing.name = name;
    timing.total_time_ms += duration;
    timing.count++;
    timing.min_time_ms = std::min(timing.min_time_ms, duration);
    timing.max_time_ms = std::max(timing.max_time_ms, duration);
    
    start_times_.erase(it);
}

void GPUProfiler::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    start_times_.clear();
    timings_.clear();
}

void GPUProfiler::print_summary() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (timings_.empty()) {
        LOG_INFO("GPU性能分析：无计时数据");
        return;
    }
    
    LOG_INFO("========== GPU性能分析 ==========");
    LOG_INFO("操作名称 | 总时间(ms) | 次数 | 平均(ms) | 最小(ms) | 最大(ms) | 占比(%)");
    LOG_INFO("-----------------------------------");
    
    double total_time = 0.0;
    for (const auto& [name, timing] : timings_) {
        total_time += timing.total_time_ms;
    }
    
    for (const auto& [name, timing] : timings_) {
        double avg_time = timing.total_time_ms / timing.count;
        double percentage = (timing.total_time_ms / total_time) * 100.0;
        
        std::ostringstream oss;
        oss << std::left << std::setw(20) << timing.name << " | "
            << std::right << std::setw(10) << std::fixed << std::setprecision(2) << timing.total_time_ms << " | "
            << std::setw(6) << timing.count << " | "
            << std::setw(8) << std::fixed << std::setprecision(2) << avg_time << " | "
            << std::setw(8) << std::fixed << std::setprecision(2) << timing.min_time_ms << " | "
            << std::setw(8) << std::fixed << std::setprecision(2) << timing.max_time_ms << " | "
            << std::setw(6) << std::fixed << std::setprecision(1) << percentage;
        LOG_INFO(oss.str());
    }
    
    LOG_INFO("-----------------------------------");
    {
        std::ostringstream oss;
        oss << "总时间: " << std::fixed << std::setprecision(2) << total_time << " ms";
        LOG_INFO(oss.str());
    }
}

void GPUProfiler::check_gpu_utilization(torch::Device device) {
    if (!device.is_cuda()) {
        LOG_WARN("当前使用CPU，无法检查GPU利用率");
        return;
    }
    
    LOG_INFO("========== GPU状态检查 ==========");
    
    // 检查CUDA是否可用
    if (!torch::cuda::is_available()) {
        LOG_ERROR("CUDA不可用");
        return;
    }
    
    // 获取GPU数量
    int num_gpus = torch::cuda::device_count();
    LOG_INFO("GPU数量: " + std::to_string(num_gpus));
    
    // 检查当前设备
    // 为了兼容不同版本的 LibTorch，避免直接调用 device.index()
    // 使用 torch::cuda::current_device() 获取当前 GPU 索引
    int current_device = at::cuda::current_device(); //torch::cuda::current_device();
    LOG_INFO("当前使用GPU: " + std::to_string(current_device));
    
    // 获取GPU属性
#ifdef USE_CUDA
    c10::cuda::CUDAGuard guard(device);
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, current_device) == cudaSuccess) {
        {
            std::ostringstream oss;
            oss << "GPU名称: " << prop.name;
            LOG_INFO(oss.str());
        }
        {
            std::ostringstream oss;
            oss << "显存大小: " << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB";
            LOG_INFO(oss.str());
        }
    }
#endif
    
    // 打印显存使用情况
    print_gpu_memory(device);
    
    LOG_INFO("提示: 使用 'nvidia-smi' 命令实时监控GPU利用率");
    LOG_INFO("提示: 如果GPU利用率低，可能原因:");
    LOG_INFO("  1. Batch size太小，增加 --batch-size 参数");
    LOG_INFO("  2. 数据预处理在CPU上，导致GPU等待");
    LOG_INFO("  3. 同步操作过多，考虑异步处理");
    LOG_INFO("  4. 模型太小，无法充分利用GPU");
}

bool GPUProfiler::is_on_gpu(const torch::Tensor& tensor) {
    return tensor.is_cuda();
}

void GPUProfiler::print_gpu_memory(torch::Device device) {
    if (!device.is_cuda()) {
        return;
    }
    
    try {
        c10::cuda::CUDAGuard guard(device);
        
        size_t allocated = 0;
        size_t cached = 0;
        
        // 使用 c10::cuda::CUDACachingAllocator::getDeviceStats 获取内存统计
        try {
            auto stats = get_memory_stats(device);
            allocated = stats.allocated_bytes_current;
            cached = stats.reserved_bytes_current;
        } catch (...) {
#ifdef USE_CUDA
            // 如果获取失败，尝试使用CUDA API
            size_t free = 0;
            if (cudaMemGetInfo(&free, &cached) == cudaSuccess) {
                allocated = cached - free;
            } else {
                LOG_WARN("无法获取GPU显存信息");
                return;
            }
#else
            LOG_WARN("CUDA未启用，无法获取GPU显存信息");
            return;
#endif
        }
        
        {
            std::ostringstream oss;
            oss << "已分配显存: " << std::fixed << std::setprecision(2) 
                << (allocated / (1024.0 * 1024.0)) << " MB";
            LOG_INFO(oss.str());
        }
        {
            std::ostringstream oss;
            oss << "缓存显存: " << std::fixed << std::setprecision(2) 
                << (cached / (1024.0 * 1024.0)) << " MB";
            LOG_INFO(oss.str());
        }
        {
            std::ostringstream oss;
            oss << "总使用显存: " << std::fixed << std::setprecision(2) 
                << ((allocated + cached) / (1024.0 * 1024.0)) << " MB";
            LOG_INFO(oss.str());
        }
    } catch (const std::exception& e) {
        LOG_WARN(std::string("获取GPU显存信息失败: ") + e.what());
    }
}

std::string GPUProfiler::get_gpu_memory_str(torch::Device device) {
    if (!device.is_cuda()) {
        return "CPU";
    }
    
    try {
        c10::cuda::CUDAGuard guard(device);
        
        size_t allocated = 0;
        size_t total = 0;
        
#ifdef USE_CUDA
        size_t free = 0;
        if (cudaMemGetInfo(&free, &total) == cudaSuccess) {
            allocated = total - free;
        } else {
            return "N/A";
        }
#else
        return "N/A";
#endif
        
        // 格式化：显示已使用/总显存 (GB)
        double allocated_gb = allocated / (1024.0 * 1024.0 * 1024.0);
        double total_gb = total / (1024.0 * 1024.0 * 1024.0);
        int usage_percent = static_cast<int>((allocated / static_cast<double>(total)) * 100.0 + 0.5);
        
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << allocated_gb 
            << "/" << std::fixed << std::setprecision(2) << total_gb 
            << "GB (" << usage_percent << "%)";
        return oss.str();
    } catch (const std::exception& e) {
        return "N/A";
    }
}

GPUProfiler::MemoryStats GPUProfiler::get_memory_stats(torch::Device device) {
    MemoryStats stats = {0, 0, 0, 0};
    
    if (!device.is_cuda()) {
        return stats;
    }
    
    try {
        c10::cuda::CUDAGuard guard(device);
        
        // 使用 c10::cuda::CUDACachingAllocator::getDeviceStats 获取内存统计
        // 这是 LibTorch C++ 中获取 CUDA 内存统计的底层 API
        // 对应 Python 中的 torch.cuda.memory_stats()
        // 为了避免对 Device.index() 的依赖，这里使用当前 CUDA 设备索引
        int dev_index = c10::cuda::current_device();
        auto device_stats = c10::cuda::CUDACachingAllocator::getDeviceStats(dev_index);
        
        // DeviceStats 结构体包含多个统计项
        // 使用 StatType::AGGREGATE 获取聚合统计（对应 Python 中的 "all"）
        using StatType = c10::CachingAllocator::StatType;
        size_t aggregate_idx = static_cast<size_t>(StatType::AGGREGATE);
        
        // allocated_bytes.all.current - 当前已分配的字节数
        stats.allocated_bytes_current = static_cast<size_t>(
            device_stats.allocated_bytes[aggregate_idx].current);
        
        // reserved_bytes.all.current - 当前已保留的字节数
        stats.reserved_bytes_current = static_cast<size_t>(
            device_stats.reserved_bytes[aggregate_idx].current);
        
        // allocated_bytes.all.peak - 峰值已分配的字节数
        stats.allocated_bytes_peak = static_cast<size_t>(
            device_stats.allocated_bytes[aggregate_idx].peak);
        
        // reserved_bytes.all.peak - 峰值已保留的字节数
        stats.reserved_bytes_peak = static_cast<size_t>(
            device_stats.reserved_bytes[aggregate_idx].peak);
        
    } catch (const std::exception& e) {
        // 如果获取失败，返回全零的统计信息
        // 调用者应该检查返回值
        std::stringstream cmd;
        cmd << "Get c10::cuda::CUDACachingAllocator::getDeviceStats function :"<< e.what();
        LOG_WARN(cmd.str());
    }
    
    return stats;
}

GPUProfiler::UtilizationReport GPUProfiler::analyze_utilization(torch::Device device,
                                                                 double collate_time_ms,
                                                                 double forward_time_ms,
                                                                 double backward_time_ms,
                                                                 double loss_time_ms) {
    GPUProfiler::UtilizationReport report;
    report.collate_time_ms = collate_time_ms;
    report.forward_time_ms = forward_time_ms;
    report.backward_time_ms = backward_time_ms;
    report.loss_time_ms = loss_time_ms;
    report.total_time_ms = collate_time_ms + forward_time_ms + backward_time_ms + loss_time_ms;
    
    // 计算 GPU 计算时间（forward + backward + loss）
    double gpu_compute_time = forward_time_ms + backward_time_ms + loss_time_ms;
    
    // GPU 利用率 = GPU 计算时间 / 总时间
    if (report.total_time_ms > 0.0) {
        report.gpu_utilization = (gpu_compute_time / report.total_time_ms) * 100.0;
    }
    
    // CPU/GPU 时间比（>1 表示 CPU 是瓶颈）
    if (gpu_compute_time > 0.0) {
        report.cpu_gpu_ratio = collate_time_ms / gpu_compute_time;
    }
    
    return report;
}

void GPUProfiler::print_utilization_report(const GPUProfiler::UtilizationReport& report) {
    using namespace logging;
    
    LOG_INFO("========== GPU 利用率分析 ==========");
    LOG_INFO("时间分布:");
    {
        std::ostringstream oss;
        oss << "  数据加载 (CPU): " << std::fixed << std::setprecision(2) 
            << report.collate_time_ms << " ms (" 
            << (report.total_time_ms > 0 ? (report.collate_time_ms / report.total_time_ms * 100.0) : 0.0) << "%)";
        LOG_INFO(oss.str());
    }
    {
        std::ostringstream oss;
        oss << "  前向传播 (GPU): " << std::fixed << std::setprecision(2) 
            << report.forward_time_ms << " ms (" 
            << (report.total_time_ms > 0 ? (report.forward_time_ms / report.total_time_ms * 100.0) : 0.0) << "%)";
        LOG_INFO(oss.str());
    }
    {
        std::ostringstream oss;
        oss << "  反向传播 (GPU): " << std::fixed << std::setprecision(2) 
            << report.backward_time_ms << " ms (" 
            << (report.total_time_ms > 0 ? (report.backward_time_ms / report.total_time_ms * 100.0) : 0.0) << "%)";
        LOG_INFO(oss.str());
    }
    {
        std::ostringstream oss;
        oss << "  损失计算 (GPU): " << std::fixed << std::setprecision(2) 
            << report.loss_time_ms << " ms (" 
            << (report.total_time_ms > 0 ? (report.loss_time_ms / report.total_time_ms * 100.0) : 0.0) << "%)";
        LOG_INFO(oss.str());
    }
    {
        std::ostringstream oss;
        oss << "  总时间: " << std::fixed << std::setprecision(2) 
            << report.total_time_ms << " ms";
        LOG_INFO(oss.str());
    }
    
    LOG_INFO("性能指标:");
    {
        std::ostringstream oss;
        oss << "  GPU 利用率: " << std::fixed << std::setprecision(1) 
            << report.gpu_utilization << "%";
        LOG_INFO(oss.str());
    }
    {
        std::ostringstream oss;
        oss << "  CPU/GPU 时间比: " << std::fixed << std::setprecision(2) 
            << report.cpu_gpu_ratio;
        if (report.cpu_gpu_ratio > 1.0) {
            oss << " ⚠️ CPU 是瓶颈（数据加载太慢）";
        } else if (report.cpu_gpu_ratio > 0.5) {
            oss << " ⚠️ CPU 可能成为瓶颈";
        } else {
            oss << " ✅ CPU 不是瓶颈";
        }
        LOG_INFO(oss.str());
    }
    
    LOG_INFO("优化建议:");
    if (report.cpu_gpu_ratio > 1.0) {
        LOG_WARN("  1. 数据加载是主要瓶颈，建议：");
        LOG_WARN("     - 使用非阻塞数据传输（已实现）");
        LOG_WARN("     - 增加数据加载线程数（--workers）");
        LOG_WARN("     - 考虑数据预取");
    }
    if (report.gpu_utilization < 50.0) {
        LOG_WARN("  2. GPU 利用率低，建议：");
        LOG_WARN("     - 增加 batch size（--batch-size）");
        LOG_WARN("     - 减少同步操作");
        LOG_WARN("     - 检查是否有其他进程占用 GPU");
    }
    if (report.forward_time_ms < report.collate_time_ms) {
        LOG_WARN("  3. GPU 计算时间 < 数据加载时间，建议：");
        LOG_WARN("     - 增加 batch size 以增加 GPU 计算时间");
        LOG_WARN("     - 优化数据加载流程");
    }
    
    LOG_INFO("====================================");
}


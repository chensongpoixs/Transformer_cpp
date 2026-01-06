#include "gpu_profiler.h"
#include "logger.h"
#include <iomanip>
#include <sstream>
#include <mutex>
#include <limits>
#include <c10/cuda/CUDAGuard.h>
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
    int current_device = device.index();
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
        
      /*  try {
            auto stats = torch::cuda::memory_stats(device.index());
            allocated = stats.allocated_bytes.all.current;
            cached = stats.reserved_bytes.all.current;
        } catch (...) */
        {
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


#ifndef TRANSFORMER_GPU_PROFILER_H
#define TRANSFORMER_GPU_PROFILER_H

#include <torch/torch.h>
#include <string>
#include <chrono>
#include <map>
#include <vector>

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

private:
    static std::map<std::string, std::chrono::steady_clock::time_point> start_times_;
    static std::map<std::string, TimingInfo> timings_;
    static std::mutex mutex_;
};

#endif // TRANSFORMER_GPU_PROFILER_H


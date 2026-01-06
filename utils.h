#ifndef TRANSFORMER_UTILS_H
#define TRANSFORMER_UTILS_H

#include <torch/torch.h>
#include <vector>
#include <memory>

/**
 * 工具函数集合
 */

/**
 * 生成subsequent mask，防止解码时看到未来的词
 * @param size 序列长度
 * @return mask张量
 */
torch::Tensor subsequent_mask(int size);

/**
 * 生成subsequent mask（指定设备）
 * @param size 序列长度
 * @param device 设备
 * @return mask张量
 */
torch::Tensor subsequent_mask(int size, torch::Device device);

/**
 * 克隆模块N次
 * @param module 要克隆的模块
 * @param N 克隆次数
 * @return 模块列表
 */
template<typename T>
std::vector<std::shared_ptr<T>> clones(const std::shared_ptr<T>& module, int N) {
    std::vector<std::shared_ptr<T>> result;
    for (int i = 0; i < N; ++i) {
        result.push_back(std::make_shared<T>(*module));
    }
    return result;
}

/**
 * Xavier均匀初始化
 * @param tensor 要初始化的张量
 */
void xavier_uniform_init(torch::Tensor& tensor);

#endif // TRANSFORMER_UTILS_H


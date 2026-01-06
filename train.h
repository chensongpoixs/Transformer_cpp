#ifndef TRANSFORMER_TRAIN_H
#define TRANSFORMER_TRAIN_H

#include <torch/torch.h>
#include "transformer.h"
#include "train_utils.h"
#include "data_loader.h"
#include "config.h"
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <tuple>

/**
 * 运行一个epoch的训练或验证
 * @param dataset 数据集
 * @param model Transformer模型
 * @param loss_compute 损失计算器
 * @param batch_size 批次大小
 * @param device 设备
 * @param config 配置
 * @return (平均损失, 总tokens数, 批次数量)
 */
std::tuple<float, long long, size_t> run_epoch(MTDataset& dataset,
                                                Transformer model,
                                                LossCompute& loss_compute,
                                                int batch_size,
                                                torch::Device device,
                                                const TransformerConfig& config,
                                                bool is_training = true,
                                                int epoch = 0,
                                                int total_epochs = 0);

/**
 * 训练函数
 * @param train_dataset 训练数据集
 * @param dev_dataset 验证数据集
 * @param model Transformer模型
 * @param criterion 损失函数
 * @param optimizer 优化器
 * @param config 配置
 * @param device 设备
 */
void train(MTDataset& train_dataset,
           MTDataset& dev_dataset,
           Transformer model,
           torch::nn::CrossEntropyLoss criterion,
           std::shared_ptr<NoamOpt> optimizer,
           const TransformerConfig& config,
           torch::Device device);

/**
 * 评估函数（计算BLEU分数）
 * 注意：完整实现需要集成SentencePiece和BLEU计算库
 * @param dataset 数据集
 * @param model Transformer模型
 * @param config 配置
 * @param device 设备
 * @return BLEU分数
 */
float evaluate(MTDataset& dataset,
                Transformer model,
                const TransformerConfig& config,
                torch::Device device);

#endif // TRANSFORMER_TRAIN_H


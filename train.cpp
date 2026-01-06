#include "train.h"
#include "beam_search.h"
#include "bleu.h"
#include "tokenizer_wrapper.h"
#include "logger.h"
#include <iomanip>
#include <algorithm>
#include <random>
#include <filesystem>
#include <numeric>
#include <limits>
#include <sstream>

namespace fs = std::filesystem;
using namespace logging;

float run_epoch(MTDataset& dataset,
                Transformer model,
                LossCompute& loss_compute,
                int batch_size,
                torch::Device device,
                const TransformerConfig& config,
                bool is_training) {
    
    float total_tokens = 0.0f;
    float total_loss = 0.0f;
    
    // 创建索引列表
    std::vector<size_t> indices(dataset.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    // 如果是训练模式，打乱数据
    if (is_training) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        LOG_DEBUG("训练模式：已对样本索引进行随机打乱");
    }
    
    // 按批次处理数据
    size_t num_batches = (dataset.size() + batch_size - 1) / batch_size;
    {
        std::ostringstream oss;
        oss << (is_training ? "[Train] " : "[Eval] ")
            << "共 " << num_batches << " 个 batch, batch_size=" << batch_size
            << ", 样本数=" << dataset.size();
        LOG_INFO(oss.str());
    }
    
    for (size_t i = 0; i < num_batches; ++i) {
        size_t start_idx = i * batch_size;
        size_t end_idx = std::min(start_idx + batch_size, dataset.size());
        
        // 获取当前批次的索引
        std::vector<size_t> batch_indices(indices.begin() + start_idx, 
                                         indices.begin() + end_idx);
        
        // 创建batch
        auto batch = dataset.collate_fn(batch_indices, device,
                                       config.padding_idx, config.bos_idx, config.eos_idx,
                                       config.src_vocab_size, config.tgt_vocab_size);
        
        // 前向传播
        auto out = model->forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask);
        {
            //std::ostringstream oss;
            //oss << ""
            //LOG_DEBUG("");
        }
        
        // 计算损失
        float loss = loss_compute(out, batch.trg_y, static_cast<float>(batch.ntokens));
        
        // 累加
        total_loss += loss * batch.ntokens;
        total_tokens += batch.ntokens;
        
        // 周期性输出进度
        if ((i + 1) % 10 == 0 || i == num_batches - 1) {
            std::ostringstream oss;
            oss << (is_training ? "[Train] " : "[Eval] ")
                << "Batch " << (i + 1) << "/" << num_batches
                << ", 当前损失=" << std::fixed << std::setprecision(3) << loss
                << ", 累积token数=" << static_cast<long long>(total_tokens);
            LOG_INFO(oss.str());
        }
    }
    
    float avg_loss = (total_tokens > 0.0f) ? (total_loss / total_tokens) : 0.0f;
    {
        std::ostringstream oss;
        oss << (is_training ? "[Train] " : "[Eval] ")
            << "Epoch结束, 平均损失=" << std::fixed << std::setprecision(4) << avg_loss
            << ", 总token数=" << static_cast<long long>(total_tokens);
        LOG_INFO(oss.str());
    }
    return avg_loss;
}

void train(MTDataset& train_dataset,
           MTDataset& dev_dataset,
           Transformer model,
           torch::nn::CrossEntropyLoss criterion,
           std::shared_ptr<NoamOpt> optimizer,
           const TransformerConfig& config,
           torch::Device device) {
    
    // 创建实验文件夹
    std::string exp_folder = "./run/train/exp_cpp";
    std::string weights_folder = exp_folder + "/weights";
    
    try {
        fs::create_directories(weights_folder);
        LOG_INFO("实验目录创建/存在: " + weights_folder);
    } catch (...) {
        LOG_WARN("无法创建实验目录: " + weights_folder + "，将继续尝试训练（可能无法保存模型）");
    }
    
    float best_bleu_score = -std::numeric_limits<float>::infinity();
    
    // 创建损失计算器
    auto loss_compute_train = LossCompute(model->get_generator(), criterion, optimizer);
    auto loss_compute_eval = LossCompute(model->get_generator(), criterion, nullptr);
    LOG_INFO("LossCompute 对象创建完成（train & eval）");
    
    // 训练循环
    for (int epoch = 1; epoch <= config.epoch_num; ++epoch) {
        {
            std::ostringstream oss;
            oss << "========== Epoch " << epoch << "/" << config.epoch_num << " ==========";
            LOG_INFO(oss.str());
        }
        
        // 训练阶段
        model->train();
        LOG_INFO("开始训练阶段...");
        float train_loss = run_epoch(train_dataset, model, loss_compute_train,
                                    config.batch_size, device, config, true);
        
        // 验证阶段
        model->eval();
        LOG_INFO("开始验证阶段...");
        float dev_loss = run_epoch(dev_dataset, model, loss_compute_eval,
                                  config.batch_size, device, config, false);
        
        // 计算BLEU分数（简化版，实际需要完整实现）
        float bleu_score = evaluate(dev_dataset, model, config, device);
        
        {
            std::ostringstream oss;
            oss << "Epoch " << epoch
                << " - TrainLoss=" << std::fixed << std::setprecision(3) << train_loss
                << ", ValLoss=" << std::fixed << std::setprecision(3) << dev_loss
                << ", BLEU=" << std::fixed << std::setprecision(2) << bleu_score;
            LOG_INFO(oss.str());
        }
        
        // 保存最佳模型
        if (bleu_score > best_bleu_score) {
            if (best_bleu_score != -std::numeric_limits<float>::infinity()) {
                std::string old_path = weights_folder + "/best_bleu_" + 
                                     std::to_string(best_bleu_score) + ".pth";
                try {
                    fs::remove(old_path);
                } catch (...) {
                    LOG_WARN("删除旧最佳模型失败: " + old_path);
                }
            }
            
            std::string best_path = weights_folder + "/best_bleu_" + 
                                   std::to_string(bleu_score) + ".pth";
            try {
                torch::save(model, best_path);
                LOG_INFO("保存新的最佳模型: " + best_path);
            } catch (const std::exception& e) {
                LOG_ERROR(std::string("保存最佳模型失败: ") + best_path + ", 错误: " + e.what());
            }
            best_bleu_score = bleu_score;
        }
        
        // 保存最后一个epoch的模型
        if (epoch == config.epoch_num) {
            std::string last_path = weights_folder + "/last_bleu_" + 
                                   std::to_string(bleu_score) + ".pth";
            try {
                torch::save(model, last_path);
                LOG_INFO("保存最后模型: " + last_path);
            } catch (const std::exception& e) {
                LOG_ERROR(std::string("保存最后模型失败: ") + last_path + ", 错误: " + e.what());
            }
        }
    }
}

float evaluate(MTDataset& dataset,
               Transformer model,
               const TransformerConfig& config,
               torch::Device device) {
    // 加载中文分词器用于解码
    auto sp_chn = chinese_tokenizer_load();
    
    model->eval();
    torch::NoGradGuard no_grad;
    
    std::vector<std::vector<std::string>> all_candidates;
    std::vector<std::vector<std::vector<std::string>>> all_references;
    
    // 评估所有数据（或限制数量）
    size_t eval_size = dataset.size();
    std::vector<size_t> indices(eval_size);
    std::iota(indices.begin(), indices.end(), 0);
    
    for (size_t i = 0; i < indices.size(); i += config.batch_size) {
        size_t end = std::min(i + config.batch_size, indices.size());
        std::vector<size_t> batch_indices(indices.begin() + i, indices.begin() + end);
        
        // 获取batch数据
        auto batch = dataset.collate_fn(batch_indices, device,
                                       config.padding_idx, config.bos_idx, config.eos_idx,
                                       config.src_vocab_size, config.tgt_vocab_size);
        
        // 创建src_mask
        auto src_mask = (batch.src != config.padding_idx).unsqueeze(-2);
        
        // 使用beam search解码
        auto [decode_results, scores] = beam_search(
            model,
            batch.src,
            src_mask,
            config.max_len,
            config.padding_idx,
            config.bos_idx,
            config.eos_idx,
            config.beam_size,
            device
        );
        
        // 处理解码结果
        for (size_t j = 0; j < decode_results.size(); ++j) {
            // 取最佳结果（第一个）
            if (!decode_results[j].empty()) {
                // 将token ID转换为字符串
                std::string translation = sp_chn->decode_ids(decode_results[j][0]);
                all_candidates.push_back(tokenize_chinese(translation));
            } else {
                all_candidates.push_back({});
            }
            
            // 参考句子（真实目标文本）
            std::vector<std::vector<std::string>> refs;
            refs.push_back(tokenize_chinese(batch.trg_text[j]));
            all_references.push_back(refs);
        }
    }
    
    // 计算BLEU分数
    float bleu_score = corpus_bleu(all_candidates, all_references, 4);
    return bleu_score;
}


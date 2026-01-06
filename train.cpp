#include "train.h"
#include "beam_search.h"
#include "bleu.h"
#include "tokenizer_wrapper.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <filesystem>
#include <numeric>
#include <limits>

namespace fs = std::filesystem;

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
    }
    
    // 按批次处理数据
    size_t num_batches = (dataset.size() + batch_size - 1) / batch_size;
    
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
        
        // 计算损失
        float loss = loss_compute(out, batch.trg_y, static_cast<float>(batch.ntokens));
        
        // 累加
        total_loss += loss * batch.ntokens;
        total_tokens += batch.ntokens;
        
        // 显示进度
        if ((i + 1) % 10 == 0 || i == num_batches - 1) {
            std::cout << "\r进度: " << (i + 1) << "/" << num_batches 
                      << " batches, 当前损失: " << std::fixed << std::setprecision(3) << loss;
            std::cout.flush();
        }
    }
    
    std::cout << std::endl;
    return total_loss / total_tokens;
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
    } catch (...) {
        std::cerr << "警告: 无法创建文件夹 " << weights_folder << std::endl;
    }
    
    float best_bleu_score = -std::numeric_limits<float>::infinity();
    
    // 创建损失计算器
    auto loss_compute_train = LossCompute(model->get_generator(), criterion, optimizer);
    auto loss_compute_eval = LossCompute(model->get_generator(), criterion, nullptr);
    
    // 训练循环
    for (int epoch = 1; epoch <= config.epoch_num; ++epoch) {
        std::cout << "\n========== Epoch " << epoch << "/" << config.epoch_num << " ==========" << std::endl;
        
        // 训练阶段
        model->train();
        std::cout << "训练中..." << std::endl;
        float train_loss = run_epoch(train_dataset, model, loss_compute_train,
                                    config.batch_size, device, config, true);
        
        // 验证阶段
        model->eval();
        std::cout << "验证中..." << std::endl;
        float dev_loss = run_epoch(dev_dataset, model, loss_compute_eval,
                                  config.batch_size, device, config, false);
        
        // 计算BLEU分数（简化版，实际需要完整实现）
        float bleu_score = evaluate(dev_dataset, model, config, device);
        
        std::cout << "Epoch " << epoch << " - Train Loss: " << std::fixed << std::setprecision(3) 
                  << train_loss << ", Val Loss: " << dev_loss 
                  << ", BLEU: " << std::setprecision(2) << bleu_score << std::endl;
        
        // 保存最佳模型
        if (bleu_score > best_bleu_score) {
            if (best_bleu_score != -std::numeric_limits<float>::infinity()) {
                std::string old_path = weights_folder + "/best_bleu_" + 
                                     std::to_string(best_bleu_score) + ".pth";
                try {
                    fs::remove(old_path);
                } catch (...) {
                    // 忽略删除错误
                }
            }
            
            std::string best_path = weights_folder + "/best_bleu_" + 
                                   std::to_string(bleu_score) + ".pth";
            torch::save(model, best_path);
            best_bleu_score = bleu_score;
            std::cout << "保存最佳模型: " << best_path << std::endl;
        }
        
        // 保存最后一个epoch的模型
        if (epoch == config.epoch_num) {
            std::string last_path = weights_folder + "/last_bleu_" + 
                                   std::to_string(bleu_score) + ".pth";
            torch::save(model, last_path);
            std::cout << "保存最后模型: " << last_path << std::endl;
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


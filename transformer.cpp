#include "transformer.h"
#include "utils.h"
#include <torch/nn/init.h>

// Transformer实现
TransformerImpl::TransformerImpl(
    Encoder encoder,
    Decoder decoder,
    Embeddings src_embed,
    Embeddings tgt_embed,
    Generator generator,
    PositionalEncoding src_position,
    PositionalEncoding tgt_position)
    : encoder(encoder),
      decoder(decoder),
      src_embed(src_embed),
      tgt_embed(tgt_embed),
      generator(generator),
      src_position(src_position),
      tgt_position(tgt_position) {
    
    register_module("encoder", encoder);
    register_module("decoder", decoder);
    register_module("src_embed", src_embed);
    register_module("tgt_embed", tgt_embed);
    register_module("generator", generator);
    register_module("src_position", src_position);
    register_module("tgt_position", tgt_position);
}

torch::Tensor TransformerImpl::encode(torch::Tensor src, torch::Tensor src_mask) {
    auto x = src_embed->forward(src);
    x = src_position->forward(x);
    return encoder->forward(x, src_mask);
}

torch::Tensor TransformerImpl::decode(
    torch::Tensor memory,
    torch::Tensor src_mask,
    torch::Tensor tgt,
    torch::Tensor tgt_mask) {
    auto x = tgt_embed->forward(tgt);
    x = tgt_position->forward(x);
    return decoder->forward(x, memory, src_mask, tgt_mask);
}

torch::Tensor TransformerImpl::forward(
    torch::Tensor src,
    torch::Tensor tgt,
    torch::Tensor src_mask,
    torch::Tensor tgt_mask) {
    // encoder的结果作为decoder的memory参数传入
    return decode(encode(src, src_mask), src_mask, tgt, tgt_mask);
}

// make_model实现
Transformer make_model(
    int src_vocab_size,
    int tgt_vocab_size,
    int N,
    int d_model,
    int d_ff,
    int h,
    float dropout,
    torch::Device device) {
    
    // 创建源语言和目标语言的嵌入层
    auto src_embed = Embeddings(d_model, src_vocab_size);
    auto tgt_embed = Embeddings(d_model, tgt_vocab_size);
    
    // 创建第一个编码器层作为模板（Encoder会创建N个独立的层）
    auto encoder_self_attn = MultiHeadedAttention(h, d_model, dropout);
    auto encoder_ff = PositionwiseFeedForward(d_model, d_ff, dropout);
    auto encoder_layer = EncoderLayer(d_model, encoder_self_attn, encoder_ff, dropout);
    
    // 创建第一个解码器层作为模板（Decoder会创建N个独立的层）
    auto decoder_self_attn = MultiHeadedAttention(h, d_model, dropout);
    auto decoder_src_attn = MultiHeadedAttention(h, d_model, dropout);
    auto decoder_ff = PositionwiseFeedForward(d_model, d_ff, dropout);
    auto decoder_layer = DecoderLayer(d_model, decoder_self_attn, decoder_src_attn, decoder_ff, dropout);
    
    // 创建编码器和解码器（内部会创建N个独立的层）
    auto encoder = Encoder(encoder_layer, N);
    auto decoder = Decoder(decoder_layer, N);
    
    // 创建位置编码（在指定设备上创建，确保pe buffer在GPU上）
    auto src_position = PositionalEncoding(d_model, dropout, 5000, device);
    auto tgt_position = PositionalEncoding(d_model, dropout, 5000, device);
    
    // 创建Transformer模型
    auto model = Transformer(
        encoder,
        decoder,
        src_embed,
        tgt_embed,
        Generator(d_model, tgt_vocab_size),
        src_position,
        tgt_position
    );
    
    // 初始化模型参数（Xavier均匀初始化）
    for (auto& param : model->parameters()) {
        if (param.dim() > 1) {
            torch::nn::init::xavier_uniform_(param);
        }
    }
    
    // 将整个模型移动到指定设备（包括所有参数和buffer）
    // 这确保所有参数、buffer和子模块都在GPU上
    model->to(device);
    
    return model;
}


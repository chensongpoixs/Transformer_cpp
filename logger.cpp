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
 * 日志系统实现 (Logging System Implementation)
 * 
 * 实现多级别日志输出功能，支持颜色显示和线程安全
				   
				   
				   
				   
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

#include "logger.h"

namespace logging {

std::mutex Logger::mutex_;
Level Logger::current_level_ = Level::Info;
bool Logger::color_enabled_ = true;
bool Logger::ansi_enabled_ = false;

// ANSI 颜色代码
namespace colors {
    const char* RESET = "\033[0m";
    const char* BOLD = "\033[1m";
    
    // 前景色
    const char* BLACK = "\033[30m";
    const char* RED = "\033[31m";
    const char* GREEN = "\033[32m";
    const char* YELLOW = "\033[33m";
    const char* BLUE = "\033[34m";
    const char* MAGENTA = "\033[35m";
    const char* CYAN = "\033[36m";
    const char* WHITE = "\033[37m";
    
    // 亮色
    const char* BRIGHT_BLACK = "\033[90m";
    const char* BRIGHT_RED = "\033[91m";
    const char* BRIGHT_GREEN = "\033[92m";
    const char* BRIGHT_YELLOW = "\033[93m";
    const char* BRIGHT_BLUE = "\033[94m";
    const char* BRIGHT_MAGENTA = "\033[95m";
    const char* BRIGHT_CYAN = "\033[96m";
    const char* BRIGHT_WHITE = "\033[97m";
}

void Logger::enable_ansi_on_windows() {
#ifdef _WIN32
    if (ansi_enabled_) return;
    
    // Windows 10 1607+ 支持 ANSI 转义码
    // 如果 ENABLE_VIRTUAL_TERMINAL_PROCESSING 未定义，使用数值 0x0004
    #ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
    #define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
    #endif
    
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE) {
        DWORD dwMode = 0;
        if (GetConsoleMode(hOut, &dwMode)) {
            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            if (SetConsoleMode(hOut, dwMode)) {
                ansi_enabled_ = true;
            }
        }
    }
#else
    // Linux/Mac 默认支持 ANSI 转义码
    ansi_enabled_ = true;
#endif
}

std::string Logger::get_color_code(Level level) {
    if (!color_enabled_ || !ansi_enabled_) {
        return "";
    }
    
    switch (level) {
        case Level::Debug:
            return colors::BRIGHT_BLACK;  // 灰色
        case Level::Info:
            return colors::BRIGHT_CYAN;   // 青色
        case Level::Warn:
            return colors::BRIGHT_YELLOW; // 黄色
        case Level::Error:
            return colors::BRIGHT_RED;    // 红色
        default:
            return "";
    }
}

std::string Logger::get_reset_code() {
    if (!color_enabled_ || !ansi_enabled_) {
        return "";
    }
    return colors::RESET;
}

static const char* level_to_string(Level level) {
    switch (level) {
        case Level::Debug: return "DEBUG";
        case Level::Info:  return "INFO";
        case Level::Warn:  return "WARN";
        case Level::Error: return "ERROR";
        default:           return "UNKNOWN";
    }
}

void Logger::init(Level level) {
    current_level_ = level;
    enable_ansi_on_windows();
}

void Logger::set_level(Level level) {
    current_level_ = level;
}

void Logger::enable_color(bool enable) {
    color_enabled_ = enable;
    if (enable) {
        enable_ansi_on_windows();
    }
}

void Logger::debug(const std::string& msg) {
    log(Level::Debug, msg);
}

void Logger::info(const std::string& msg) {
    log(Level::Info, msg);
}

void Logger::warn(const std::string& msg) {
    log(Level::Warn, msg);
}

void Logger::error(const std::string& msg) {
    log(Level::Error, msg);
}

void Logger::log(Level level, const std::string& msg) {
    if (static_cast<int>(level) < static_cast<int>(current_level_)) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // 获取当前时间
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#if defined(_WIN32) || defined(_WIN64)
    localtime_s(&tm_buf, &now_c);
#else
    localtime_r(&now_c, &tm_buf);
#endif

    char time_buf[32];
    std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", &tm_buf);

    // 获取颜色代码
    std::string color_code = get_color_code(level);
    std::string reset_code = get_reset_code();
    
    std::ostringstream oss;
    oss << color_code
        << "[" << time_buf << "]"
        << "[" << level_to_string(level) << "] "
        << reset_code
        << msg;

    std::cout << oss.str() << std::endl;
}

} // namespace logging
//
//{
//  "cells": [],
//  "metadata": {
//    "language_info": {
//      "name": "python"
//    }
//  },
//  "nbformat": 4,
//  "nbformat_minor": 2
//}
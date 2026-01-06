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
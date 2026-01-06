#include "logger.h"

namespace logging {

std::mutex Logger::mutex_;
Level Logger::current_level_ = Level::Info;

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
}

void Logger::set_level(Level level) {
    current_level_ = level;
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

    std::ostringstream oss;
    oss << "[" << time_buf << "]"
        << "[" << level_to_string(level) << "] "
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
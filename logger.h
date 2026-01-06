#ifndef TRANSFORMER_LOGGER_H
#define TRANSFORMER_LOGGER_H

#include <string>
#include <mutex>
#include <iostream>
#include <chrono>
#include <ctime>
#include <sstream>

// 简单日志系统：输出到标准输出，带时间戳和级别
namespace logging {

enum class Level {
    Debug = 0,
    Info  = 1,
    Warn  = 2,
    Error = 3
};

class Logger {
public:
    // 初始化日志系统，默认级别为 Info
    static void init(Level level = Level::Info);
    static void set_level(Level level);

    static void debug(const std::string& msg);
    static void info(const std::string& msg);
    static void warn(const std::string& msg);
    static void error(const std::string& msg);

private:
    static void log(Level level, const std::string& msg);

    static std::mutex mutex_;
    static Level current_level_;
};

// 便捷宏
#define LOG_DEBUG(msg) ::logging::Logger::debug(msg)
#define LOG_INFO(msg)  ::logging::Logger::info(msg)
#define LOG_WARN(msg)  ::logging::Logger::warn(msg)
#define LOG_ERROR(msg) ::logging::Logger::error(msg)

} // namespace logging

#endif // TRANSFORMER_LOGGER_H

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
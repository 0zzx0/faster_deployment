#pragma once

#include <string>

#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdarg.h>

namespace FasterNCNN {

/*
logger
*/
enum class LogLevel : int{
    Debug   = 5,
    Verbose = 4,
    Info    = 3,
    Warning = 2,
    Error   = 1,
    Fatal   = 0
};


static const char* level_string(LogLevel level);
static void __log_func(const char* file, int line, LogLevel level, const char* fmt, ...);
static std::string file_name(const std::string& path, bool include_suffix);


/* 修改这个level来实现修改日志输出级别 */
#define CURRENT_LOG_LEVEL       LogLevel::Info
#define INFOD(...)			__log_func(__FILE__, __LINE__, LogLevel::Debug, __VA_ARGS__)
#define INFOV(...)			__log_func(__FILE__, __LINE__, LogLevel::Verbose, __VA_ARGS__)
#define INFO(...)			__log_func(__FILE__, __LINE__, LogLevel::Info, __VA_ARGS__)
#define INFOW(...)			__log_func(__FILE__, __LINE__, LogLevel::Warning, __VA_ARGS__)
#define INFOE(...)			__log_func(__FILE__, __LINE__, LogLevel::Error, __VA_ARGS__)
#define INFOF(...)			__log_func(__FILE__, __LINE__, LogLevel::Fatal, __VA_ARGS__)

static const char* level_string(LogLevel level){
    switch (level){
        case LogLevel::Debug: return "debug";
        case LogLevel::Verbose: return "verbo";
        case LogLevel::Info: return "info";
        case LogLevel::Warning: return "warn";
        case LogLevel::Error: return "error";
        case LogLevel::Fatal: return "fatal";
        default: return "unknow";
    }
}

static void __log_func(const char* file, int line, LogLevel level, const char* fmt, ...){

    if(level > CURRENT_LOG_LEVEL)
        return;

    va_list vl;
    va_start(vl, fmt);
    
    char buffer[2048];
    std::string filename = file_name(file, true);
    int n = snprintf(buffer, sizeof(buffer), "[%s][%s:%d]:", level_string(level), filename.c_str(), line);
    vsnprintf(buffer + n, sizeof(buffer) - n, fmt, vl);

    fprintf(stdout, "%s\n", buffer);
    if (level == LogLevel::Fatal) {
        fflush(stdout);
        abort();
    }
}

static std::string file_name(const std::string& path, bool include_suffix){

    if (path.empty()) return "";
    int p = path.rfind('/');
    p += 1;

    //include suffix
    if (include_suffix)
        return path.substr(p);

    int u = path.rfind('.');
    if (u == -1)
        return path.substr(p);

    if (u <= p) u = path.size();
    return path.substr(p, u - p);
}


}
#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdarg.h>

#include <cuda_runtime.h>

/*
一些工具函数 包括CUDA检查 输出
文件保存读取等函数
全部在tools里面定义并直接实现
*/

namespace YOLO{
    using namespace std;

    enum class LogLevel : int{
        Debug   = 5,
        Verbose = 4,
        Info    = 3,
        Warning = 2,
        Error   = 1,
        Fatal   = 0
    };

    #define CURRENT_DEVICE_ID   -1  // 当前设备
    static bool check_runtime(cudaError_t e, const char* call, int line, const char *file);
    static const char* level_string(LogLevel level);
    static string file_name(const string& path, bool include_suffix);
    static void __log_func(const char* file, int line, LogLevel level, const char* fmt, ...);


    
    /* 修改这个level来实现修改日志输出级别 */
    #define CURRENT_LOG_LEVEL       LogLevel::Info
    #define INFOD(...)			__log_func(__FILE__, __LINE__, LogLevel::Debug, __VA_ARGS__)
    #define INFOV(...)			__log_func(__FILE__, __LINE__, LogLevel::Verbose, __VA_ARGS__)
    #define INFO(...)			__log_func(__FILE__, __LINE__, LogLevel::Info, __VA_ARGS__)
    #define INFOW(...)			__log_func(__FILE__, __LINE__, LogLevel::Warning, __VA_ARGS__)
    #define INFOE(...)			__log_func(__FILE__, __LINE__, LogLevel::Error, __VA_ARGS__)
    #define INFOF(...)			__log_func(__FILE__, __LINE__, LogLevel::Fatal, __VA_ARGS__)


    #define KernelPositionBlock											\
        int position = (blockDim.x * blockIdx.x + threadIdx.x);		    \
        if (position >= (edge)) return;

    #define checkCudaRuntime(call) check_runtime(call, #call, __LINE__, __FILE__)

    #define checkCudaKernel(...)                                                                         \
        __VA_ARGS__;                                                                                     \
        do{cudaError_t cudaStatus = cudaPeekAtLastError();                                               \
        if (cudaStatus != cudaSuccess){                                                                  \
            INFOE("launch failed: %s", cudaGetErrorString(cudaStatus));                                  \
        }} while(0);

    #define Assert(op)					 \
        do{                              \
            bool cond = !(!(op));        \
            if(!cond){                   \
                INFOF("Assert failed, " #op);  \
            }                                  \
        }while(false)

    

    static bool check_runtime(cudaError_t e, const char* call, int line, const char *file){
        if (e != cudaSuccess) {
            INFOE("CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d", call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
            return false;
        }
        return true;
    }

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

    static string file_name(const string& path, bool include_suffix){

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

    static void __log_func(const char* file, int line, LogLevel level, const char* fmt, ...){

        if(level > CURRENT_LOG_LEVEL)
            return;

        va_list vl;
        va_start(vl, fmt);
        
        char buffer[2048];
        string filename = file_name(file, true);
        int n = snprintf(buffer, sizeof(buffer), "[%s][%s:%d]:", level_string(level), filename.c_str(), line);
        vsnprintf(buffer + n, sizeof(buffer) - n, fmt, vl);

        fprintf(stdout, "%s\n", buffer);
        if (level == LogLevel::Fatal) {
            fflush(stdout);
            abort();
        }
    }


    static bool exists(const string& path){
        return access(path.c_str(), R_OK) == 0;
    }

    static bool save_file(const string& file, const void* data, size_t length){

        FILE* f = fopen(file.c_str(), "wb");
        if (!f) return false;

        if (data && length > 0){
            if (fwrite(data, 1, length, f) != length){
                fclose(f);
                return false;
            }
        }
        fclose(f);
        return true;
    }

    static bool save_file(const string& file, const vector<uint8_t>& data){
        return save_file(file, data.data(), data.size());
    }


    /* 构造时设置当前gpuid，析构时修改为原来的gpuid */
    class AutoDevice{
    public:
        AutoDevice(int device_id = 0){
            cudaGetDevice(&old_);
            checkCudaRuntime(cudaSetDevice(device_id));
        }

        virtual ~AutoDevice(){
            checkCudaRuntime(cudaSetDevice(old_));
        }
    
    private:
        int old_ = -1;
    };
    

    static bool check_device_id(int device_id){
        int device_count = -1;
        checkCudaRuntime(cudaGetDeviceCount(&device_count));
        if(device_id < 0 || device_id >= device_count){
            INFOE("Invalid device id: %d, count = %d", device_id, device_count);
            return false;
        }
        return true;
    }

    static int get_device(int device_id){
        if(device_id != CURRENT_DEVICE_ID){
            check_device_id(device_id);
            return device_id;
        }
        checkCudaRuntime(cudaGetDevice(&device_id));
        return device_id;
    }



    static std::vector<uint8_t> load_file(const string& file){

        ifstream in(file, ios::in | ios::binary);
        if (!in.is_open())
            return {};

        in.seekg(0, ios::end);
        size_t length = in.tellg();

        std::vector<uint8_t> data;
        if (length > 0){
            in.seekg(0, ios::beg);
            data.resize(length);

            in.read((char*)&data[0], length);
        }
        in.close();
        return data;
    }


    inline int upbound(int n, int align = 32){return (n + align - 1) / align * align;}

    template<typename _T>
    static string join_dims(const vector<_T>& dims){
        stringstream output;
        char buf[64];
        const char* fmts[] = {"%d", " x %d"};
        for(int i = 0; i < dims.size(); ++i){
            snprintf(buf, sizeof(buf), fmts[i != 0], dims[i]);
            output << buf;
        }
        return output.str();
    }

    

}; // end namespace YOLO

#endif
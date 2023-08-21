/**
 * @file monopoly_accocator.hpp
 * @author 0zzx0
 * @brief 独占分配器
 * @version 0.1
 * @date 2023-6-11 2023-8-21
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef MONOPOLY_ALLOCATOR_HPP
#define MONOPOLY_ALLOCATOR_HPP

#include <vector>
#include <mutex>
#include <memory>
#include <condition_variable>

namespace FasterTRT {
///////////////////////////class MonopolyAllocator///////////////////////////
/* 独占分配器
    通过对tensor做独占管理，具有max_batch * 2个tensor，通过query获取一个tensor
    当推理结束后，该tensor释放使用权，即可交给下一个图像使用，内存实现复用

 * 1. tensor复用
 * 2. tensor的预处理和推理并行
 *
 * 输入图像时，具有2倍batch的空间进行预处理用于缓存
 * 引擎推理时，每次拿1个batch的数据进行推理
 * 当引擎推理速度慢而预处理速度快时，输入图像需要进行等候。
 **/
template <class _ItemType>
class MonopolyAllocator {
public:
    /* MonopolyData是数据容器类
    允许query获取的item执行item->release释放自身所有权，该对象可以被复用
    通过item->data()获取储存的对象的指针
    */
    class MonopolyData {
    public:
        std::shared_ptr<_ItemType>& data() { return data_; }
        void release() { manager_->release_one(this); }

    private:
        MonopolyData(MonopolyAllocator* pmanager) { manager_ = pmanager; }

    private:
        friend class MonopolyAllocator;
        MonopolyAllocator* manager_ = nullptr;
        std::shared_ptr<_ItemType> data_;
        bool available_ = true;
    };
    typedef std::shared_ptr<MonopolyData> MonopolyDataPointer;

    // 构造函数 初始化尺寸
    MonopolyAllocator(int size) {
        capacity_ = size;
        num_available_ = size;
        datas_.resize(size);

        for(int i = 0; i < size; ++i)
            datas_[i] = std::shared_ptr<MonopolyData>(new MonopolyData(this));
    }

    // 析构
    virtual ~MonopolyAllocator() {
        run_ = false;
        cv_.notify_all();

        std::unique_lock<std::mutex> l(lock_);
        cv_exit_.wait(l, [&]() { return num_wait_thread_ == 0; });
    }
    /* 获取一个可用的对象
    timeout：超时时间，如果没有可用的对象，将会进入阻塞等待，如果等待超时则返回空指针
    请求得到一个对象后，该对象被占用，除非他执行了release释放该对象所有权
    */
    MonopolyDataPointer query(int timeout = 10000) {
        std::unique_lock<std::mutex> l(lock_);
        if(!run_) return nullptr;

        if(num_available_ == 0) {
            num_wait_thread_++;

            auto state = cv_.wait_for(l, std::chrono::milliseconds(timeout),
                                      [&]() { return num_available_ > 0 || !run_; });

            num_wait_thread_--;
            cv_exit_.notify_one();

            // timeout, no available, exit program
            if(!state || num_available_ == 0 || !run_) return nullptr;
        }

        auto item = std::find_if(datas_.begin(), datas_.end(),
                                 [](MonopolyDataPointer& item) { return item->available_; });
        if(item == datas_.end()) return nullptr;

        (*item)->available_ = false;
        num_available_--;
        return *item;
    }

    // 有效数量
    int num_available() { return num_available_; }

    // 空间大小
    int capacity() { return capacity_; }

private:
    // 释放一个对象的所有权
    void release_one(MonopolyData* prq) {
        std::unique_lock<std::mutex> l(lock_);
        if(!prq->available_) {
            prq->available_ = true;
            num_available_++;
            cv_.notify_one();
        }
    }

private:
    std::mutex lock_;
    std::condition_variable cv_;
    std::condition_variable cv_exit_;
    std::vector<MonopolyDataPointer> datas_;
    int capacity_ = 0;
    volatile int num_available_ = 0;
    volatile int num_wait_thread_ = 0;
    volatile bool run_ = true;
};

};  // namespace FasterTRT

#endif
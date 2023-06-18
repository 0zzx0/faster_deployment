#include "infer_base.hpp"

namespace YOLO{
///////////////////////////////////////////////////////////////////////
/////////////////////////// TRTInferImpl //////////////////////////////
///////////////////////////////////////////////////////////////////////

TRTInferImpl::~TRTInferImpl(){
    destroy();
}

// 销毁对象(析构默认调用)
void TRTInferImpl::destroy() {
    int old_device = 0;
    checkCudaRuntime(cudaGetDevice(&old_device));
    checkCudaRuntime(cudaSetDevice(device_));
    this->context_.reset();
    this->blobsNameMapper_.clear();
    this->outputs_.clear();
    this->inputs_.clear();
    this->inputs_name_.clear();
    this->outputs_name_.clear();
    checkCudaRuntime(cudaSetDevice(old_device));
}

// 打印信息 输入输出信息
void TRTInferImpl::print(){
    if(!context_){
        INFOW("Infer print, nullptr.");
        return;
    }

    INFO("Infer %p detail", this);
    INFO("\tMax Batch Size: %d", this->get_max_batch_size());
    INFO("\tInputs: %d", inputs_.size());
    for(int i = 0; i < inputs_.size(); ++i){
        auto& tensor = inputs_[i];
        auto& name = inputs_name_[i];
        INFO("\t\t%d.%s : shape {%s}", i, name.c_str(), tensor->shape_string());
    }

    INFO("\tOutputs: %d", outputs_.size());
    for(int i = 0; i < outputs_.size(); ++i){
        auto& tensor = outputs_[i];
        auto& name = outputs_name_[i];
        INFO("\t\t%d.%s : shape {%s}", i, name.c_str(), tensor->shape_string());
    }
}

// 序列化engine
std::shared_ptr<std::vector<uint8_t>> TRTInferImpl::serial_engine() {
    auto memory = this->context_->engine_->serialize();
    auto output = make_shared<std::vector<uint8_t>>((uint8_t*)memory->data(), (uint8_t*)memory->data()+memory->size());
    memory->destroy();
    return output;
}

// 从内存加载
bool TRTInferImpl::load_from_memory(const void* pdata, size_t size) {
    if (pdata == nullptr || size == 0)
        return false;

    context_.reset(new EngineContext());

    //build model
    if (!context_->build_model(pdata, size)) {
        context_.reset();
        return false;
    }

    workspace_.reset(new MixMemory());
    cudaGetDevice(&device_);
    build_engine_input_and_outputs_mapper();
    return true;
}

// 从文件加载
bool TRTInferImpl::load(const std::string& file, int batch_size) {

    auto data = load_file(file);
    if (data.empty())
        return false;

    context_.reset(new EngineContext());

    //build model
    if (!context_->build_model(data.data(), data.size())) {
        context_.reset();
        return false;
    }
    batch_max_size_ = batch_size;

    workspace_.reset(new MixMemory());
    cudaGetDevice(&device_);
    build_engine_input_and_outputs_mapper();
    return true;
}

// 获取设备的内存大小
size_t TRTInferImpl::get_device_memory_size() {
    EngineContext* context = (EngineContext*)this->context_.get();
    return context->context_->getEngine().getDeviceMemorySize();
}

// 获取输入输出等信息
void TRTInferImpl::build_engine_input_and_outputs_mapper() {
        
    EngineContext* context = (EngineContext*)this->context_.get();
    int nbBindings = context->engine_->getNbBindings();
    // int max_batchsize = context->engine_->getMaxBatchSize();
    int max_batchsize = batch_max_size_;

    inputs_.clear();
    inputs_name_.clear();
    outputs_.clear();
    outputs_name_.clear();
    orderdBlobs_.clear();
    bindingsPtr_.clear();
    blobsNameMapper_.clear();
    for (int i = 0; i < nbBindings; ++i) {

        auto dims = context->engine_->getBindingDimensions(i);
        auto type = context->engine_->getBindingDataType(i);
        const char* bindingName = context->engine_->getBindingName(i);
        dims.d[0] = max_batchsize;
        auto newTensor = make_shared<Tensor>(dims.nbDims, dims.d);
        newTensor->set_stream(this->context_->stream_);
        newTensor->set_workspace(this->workspace_);
        if (context->engine_->bindingIsInput(i)) {
            //if is input
            inputs_.push_back(newTensor);
            inputs_name_.push_back(bindingName);
            inputs_map_to_ordered_index_.push_back(orderdBlobs_.size());
        }
        else {
            //if is output
            outputs_.push_back(newTensor);
            outputs_name_.push_back(bindingName);
            outputs_map_to_ordered_index_.push_back(orderdBlobs_.size());
        }
        blobsNameMapper_[bindingName] = i;
        orderdBlobs_.push_back(newTensor);
    }
    bindingsPtr_.resize(orderdBlobs_.size());
}

// 数据和推理引擎设置cuda流
void TRTInferImpl::set_stream(cudaStream_t stream){
    this->context_->set_stream(stream);

    for(auto& t : orderdBlobs_)
        t->set_stream(stream);
}

// 获取当前cuda流
cudaStream_t TRTInferImpl::get_stream() {
    return this->context_->stream_;
}

// 获取当前设备
int TRTInferImpl::device() {
    return device_;
}

// 等待同步
void TRTInferImpl::synchronize() {
    checkCudaRuntime(cudaStreamSynchronize(context_->stream_));
}

// 判断是否属于输出
bool TRTInferImpl::is_output_name(const std::string& name){
    return std::find(outputs_name_.begin(), outputs_name_.end(), name) != outputs_name_.end();
}

// 判断是否属于输入
bool TRTInferImpl::is_input_name(const std::string& name){
    return std::find(inputs_name_.begin(), inputs_name_.end(), name) != inputs_name_.end();
}

// 推理
void TRTInferImpl::forward(bool sync) {

    EngineContext* context = (EngineContext*)context_.get();
    int inputBatchSize = inputs_[0]->size(0);
    for(int i = 0; i < context->engine_->getNbBindings(); ++i){
        auto dims = context->engine_->getBindingDimensions(i);
        auto type = context->engine_->getBindingDataType(i);
        dims.d[0] = inputBatchSize;
        if(context->engine_->bindingIsInput(i)){
            context->context_->setBindingDimensions(i, dims);
        }
    }

    for (int i = 0; i < outputs_.size(); ++i) {
        outputs_[i]->resize_single_dim(0, inputBatchSize);
        outputs_[i]->to_gpu(false);
    }

    for (int i = 0; i < orderdBlobs_.size(); ++i)
        bindingsPtr_[i] = orderdBlobs_[i]->gpu();

    void** bindingsptr = bindingsPtr_.data();
    //bool execute_result = context->context_->enqueue(inputBatchSize, bindingsptr, context->stream_, nullptr);
    bool execute_result = context->context_->enqueueV2(bindingsptr, context->stream_, nullptr);
    if(!execute_result){
        auto code = cudaGetLastError();
        INFOF("execute fail, code %d[%s], message %s", code, cudaGetErrorName(code), cudaGetErrorString(code));
    }

    if (sync) {
        synchronize();
    }
}

// 获取workspace_(这是一个内存管理类的指针)
std::shared_ptr<MixMemory> TRTInferImpl::get_workspace() {
    return workspace_;
}

// 返回输入数量
int TRTInferImpl::num_input() {
    return this->inputs_.size();
}

// 返回输出数量
int TRTInferImpl::num_output() {
    return this->outputs_.size();
}

// 设置第index的输入
void TRTInferImpl::set_input (int index, std::shared_ptr<Tensor> tensor){
    Assert(index >= 0 && index < inputs_.size());
    this->inputs_[index] = tensor;

    int order_index = inputs_map_to_ordered_index_[index];
    this->orderdBlobs_[order_index] = tensor;
}

// 设置第index的输出
void TRTInferImpl::set_output(int index, std::shared_ptr<Tensor> tensor){
    Assert(index >= 0 && index < outputs_.size());
    this->outputs_[index] = tensor;

    int order_index = outputs_map_to_ordered_index_[index];
    this->orderdBlobs_[order_index] = tensor;
}

// 返回第index输入tensor
std::shared_ptr<Tensor> TRTInferImpl::input(int index) {
    Assert(index >= 0 && index < inputs_name_.size());
    return this->inputs_[index];
}

// 返回第index输入tensor名字
std::string TRTInferImpl::get_input_name(int index){
    Assert(index >= 0 && index < inputs_name_.size());
    return inputs_name_[index];
}

// 返回第index输输出tensor
std::shared_ptr<Tensor> TRTInferImpl::output(int index) {
    Assert(index >= 0 && index < outputs_.size());
    return outputs_[index];
}

// 返回第index输出tensor名字
std::string TRTInferImpl::get_output_name(int index){
    Assert(index >= 0 && index < outputs_name_.size());
    return outputs_name_[index];
}

// 获取最大batchsize
int TRTInferImpl::get_max_batch_size() {
    Assert(this->context_ != nullptr);
    // return this->context_->engine_->getMaxBatchSize();
    return batch_max_size_;
}

// 根据名字查找tensor
std::shared_ptr<Tensor> TRTInferImpl::tensor(const std::string& name) {
    Assert(this->blobsNameMapper_.find(name) != this->blobsNameMapper_.end());
    return orderdBlobs_[blobsNameMapper_[name]];
}




///////////////////////////////////////////////////////////////////////
/////////////////////////加载文件初始化对象 /////////////////////////////
/////////////////////////////////////////////////////////////////////// 
std::shared_ptr<TRTInferImpl> load_infer(const string& file, int batch_size) {
    
    std::shared_ptr<TRTInferImpl> infer(new TRTInferImpl());
    if (!infer->load(file, batch_size))
        infer.reset();
    return infer;
}


};


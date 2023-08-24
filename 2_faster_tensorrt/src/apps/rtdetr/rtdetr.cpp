#include "rtdetr.h"

namespace RTDETR {

RtDetrTRTInferImpl::~RtDetrTRTInferImpl() {
    stop();
}

// 启动 但不是重写基类的startup 参数不一样 里面会去调用基类
bool RtDetrTRTInferImpl::startup(const std::string& file, int gpuid, int batch_size,
                                 float confidence_threshold) {
    // const float mean_norm[3] = {103.53, 116.28, 123.675};
    // const float std_norm[3] = {57.375, 57.12, 58.395};
    // normalize_ = Norm::mean_std(mean_norm, std_norm, 1 / 255.0f, ChannelType::SwapRB);
    normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
    confidence_threshold_ = confidence_threshold;
    batch_size_ = batch_size;
    return ThreadSafedAsyncInferImpl::startup(std::make_tuple(file, gpuid));
}

// 重写基类worker 工作线程
void RtDetrTRTInferImpl::worker(std::promise<bool>& result) {
    std::string file = std::get<0>(start_param_);
    int gpuid = std::get<1>(start_param_);

    set_device(gpuid);
    auto engine = load_infer(file, batch_size_);
    if(engine == nullptr) {
        INFOE("Engine %s load failed", file.c_str());
        result.set_value(false);
        return;
    }

    engine->print();

    const int MAX_IMAGE_BBOX = 100;
    const int NUM_BOX_ELEMENT = 7;  // left, top, right, bottom, confidence, class, keepflag
    Tensor affin_matrix_device(FasterTRT::DataType::Float);
    Tensor output_array_device(FasterTRT::DataType::Float);

    // 输入输出
    int max_batch_size = engine->get_max_batch_size();
    auto input = engine->tensor("image");
    auto output = engine->tensor("output");

    // decode数据
    int num_classes, output_num_bboxes, output_fm_area;
    output_num_bboxes = output->size(0) * output->size(1);
    output_fm_area = output->size(2);
    num_classes = output->size(2) - 4;

    // 输入
    input_width_ = input->size(3);
    input_height_ = input->size(2);
    tensor_allocator_ = std::make_shared<MonopolyAllocator<Tensor>>(max_batch_size * 2);
    stream_ = engine->get_stream();
    gpu_ = gpuid;
    result.set_value(true);  // 初始化完成 返回给startup函数结束

    input->resize_single_dim(0, max_batch_size).to_gpu();
    affin_matrix_device.set_stream(stream_);

    // 这里8个值的目的是保证 8 * sizeof(float) % 32 == 0
    affin_matrix_device.resize(max_batch_size, 8).to_gpu();

    // 输出数据
    output_array_device.set_stream(stream_);
    output_array_device.resize(max_batch_size, 1 + MAX_IMAGE_BBOX * NUM_BOX_ELEMENT).to_gpu();

    auto decode_kernel_invoker = rtdetr_decode_kernel_invoker;

    // 循环等待&检测
    std::vector<Job> fetch_jobs;
    while(get_jobs_and_wait(fetch_jobs, max_batch_size)) {
        int infer_batch_size = fetch_jobs.size();
        input->resize_single_dim(0, infer_batch_size);

        for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch) {
            auto& job = fetch_jobs[ibatch];
            auto& mono = job.mono_tensor->data();
            affin_matrix_device.copy_from_gpu(affin_matrix_device.offset(ibatch),
                                              mono->get_workspace()->gpu(), 6);
            input->copy_from_gpu(input->offset(ibatch), mono->gpu(), mono->count());
            job.mono_tensor->release();
        }

        engine->forward(false);

        output_array_device.to_gpu(false);
        for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch) {
            // auto& job = fetch_jobs[ibatch];
            float* image_based_output = output->gpu<float>(ibatch);
            float* output_array_ptr = output_array_device.gpu<float>(ibatch);
            auto affine_matrix = affin_matrix_device.gpu<float>(ibatch);
            checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(int), stream_));
            decode_kernel_invoker(image_based_output, output_num_bboxes, output_fm_area,
                                  num_classes, confidence_threshold_, affine_matrix,
                                  output_array_ptr, MAX_IMAGE_BBOX, input_width_, stream_);
        }

        output_array_device.to_cpu();
        for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch) {
            float* parray = output_array_device.cpu<float>(ibatch);
            int count = std::min(MAX_IMAGE_BBOX, (int)*parray);
            auto& job = fetch_jobs[ibatch];
            auto& image_based_boxes = job.output;
            for(int i = 0; i < count; ++i) {
                float* pbox = parray + 1 + i * NUM_BOX_ELEMENT;
                int label = pbox[5];
                int keepflag = pbox[6];
                if(keepflag == 1) {
                    image_based_boxes.emplace_back(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4],
                                                   label);
                }
            }
            job.pro->set_value(image_based_boxes);
        }
        fetch_jobs.clear();
    }
    stream_ = nullptr;
    // TODO 这个流是否要考虑换个地方释放？
    checkCudaRuntime(cudaStreamDestroy(stream_pro_));
    stream_pro_ = nullptr;
    tensor_allocator_.reset();
    INFO("Engine destroy.");
}

// 预处理
bool RtDetrTRTInferImpl::preprocess(Job& job, const cv::Mat& image) {
    if(tensor_allocator_ == nullptr) {
        INFOE("tensor_allocator_ is nullptr");
        return false;
    }

    job.mono_tensor = tensor_allocator_->query();
    if(job.mono_tensor == nullptr) {
        INFOE("Tensor allocator query failed.");
        return false;
    }

    if(stream_pro_ == nullptr) {
        checkCudaRuntime(cudaStreamCreate(&stream_pro_));
    }

    AutoDevice auto_device(gpu_);
    auto& tensor = job.mono_tensor->data();
    if(tensor == nullptr) {
        // not init
        tensor = std::make_shared<Tensor>();
        tensor->set_workspace(std::make_shared<MixMemory>());
    }

    cv::Size input_size(input_width_, input_height_);
    job.additional.compute(image.size(), input_size);

    tensor->set_stream(stream_pro_);
    tensor->resize(1, 3, input_height_, input_width_);

    size_t size_image = image.cols * image.rows * 3;
    size_t size_matrix = upbound(sizeof(job.additional.d2i), 32);
    auto workspace = tensor->get_workspace();
    uint8_t* gpu_workspace = (uint8_t*)workspace->gpu(size_matrix + size_image);
    float* affine_matrix_device = (float*)gpu_workspace;
    uint8_t* image_device = size_matrix + gpu_workspace;

    uint8_t* cpu_workspace = (uint8_t*)workspace->cpu(size_matrix + size_image);
    float* affine_matrix_host = (float*)cpu_workspace;
    uint8_t* image_host = size_matrix + cpu_workspace;

    memcpy(image_host, image.data, size_image);
    memcpy(affine_matrix_host, job.additional.d2i, sizeof(job.additional.d2i));
    checkCudaRuntime(
        cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_pro_));
    checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host,
                                     sizeof(job.additional.d2i), cudaMemcpyHostToDevice,
                                     stream_pro_));

    warp_affine_bilinear_and_normalize_plane(image_device, image.cols * 3, image.cols, image.rows,
                                             tensor->gpu<float>(), input_width_, input_height_,
                                             affine_matrix_device, .0, normalize_, stream_pro_);
    // 这个地方需要同步，确保数据放到gpu后才可以吧任务提交到队列中。
    cudaStreamSynchronize(stream_pro_);

    return true;
}

// 提交<vector>任务
std::vector<std::shared_future<BoxArray>> RtDetrTRTInferImpl::commits(
    const std::vector<cv::Mat>& images) {
    return ThreadSafedAsyncInferImpl::commits(images);
}

// 提交cv::Mat任务
std::shared_future<BoxArray> RtDetrTRTInferImpl::commit(const cv::Mat& image) {
    return ThreadSafedAsyncInferImpl::commit(image);
}

// 创建推理器
std::shared_ptr<Infer> create_infer(const std::string& engine_file, int gpuid, int batch_size,
                                    float confidence_threshold) {
    std::shared_ptr<RtDetrTRTInferImpl> instance(new RtDetrTRTInferImpl());
    if(!instance->startup(engine_file, gpuid, batch_size, confidence_threshold)) {
        instance.reset();
    }
    return instance;
}
}  // namespace RTDETR

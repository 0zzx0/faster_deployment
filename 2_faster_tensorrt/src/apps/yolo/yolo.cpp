#include "yolo.h"

namespace YOLO {

// 返回type的名字
const char* type_name(YoloType type) {
    switch(type) {
        // case YoloType::V5: return "YoloV5";
        case YoloType::X:
            return "YoloX";
        default:
            return "Unknow";
    }
}

// yolox decode数据
DecodeMeta DecodeMeta::x_default_meta() {
    DecodeMeta meta;
    meta.num_anchor = 1;
    meta.num_level = 3;

    const int strides[] = {8, 16, 32};
    memcpy(meta.strides, strides, sizeof(meta.strides));
    return meta;
}

// yolov8 decode数据 同yolox
DecodeMeta DecodeMeta::v8_default_meta() {
    DecodeMeta meta;
    meta.num_anchor = 1;
    meta.num_level = 3;

    const int strides[] = {8, 16, 32};
    memcpy(meta.strides, strides, sizeof(meta.strides));
    return meta;
}

// yolov5 p5 decode数据
// DecodeMeta DecodeMeta::v5_p5_default_meta(){
//     DecodeMeta meta;
//     meta.num_anchor = 3;
//     meta.num_level = 3;

//     float anchors[] = {
//         10.000000, 13.000000, 16.000000, 30.000000, 33.000000, 23.000000,
//         30.000000, 61.000000, 62.000000, 45.000000, 59.000000, 119.000000,
//         116.000000, 90.000000, 156.000000, 198.000000, 373.000000, 326.000000
//     };

//     int abs_index = 0;
//     for(int i = 0; i < meta.num_level; ++i){
//         for(int j = 0; j < meta.num_anchor; ++j){
//             int aidx = i * meta.num_anchor + j;
//             meta.w[aidx] = anchors[abs_index++];
//             meta.h[aidx] = anchors[abs_index++];
//         }
//     }

//     const int strides[] = {8, 16, 32};
//     memcpy(meta.strides, strides, sizeof(meta.strides));
//     return meta;
// }

//////////////////////////////////////////////////////////////////////
//////////////////////// YoloTRTInferImpl ////////////////////////////
//////////////////////////////////////////////////////////////////////
YoloTRTInferImpl::~YoloTRTInferImpl() {
    stop();
}

// 启动 但不是重写基类的startup 参数不一样 里面会去调用基类
bool YoloTRTInferImpl::startup(const std::string& file, YoloType type, int gpuid, int batch_size,
                               float confidence_threshold, float nms_threshold,
                               bool is_use_trtnNMSPlugin) {
    if(type == YoloType::X) {
        normalize_ = Norm::None();
        meta_ = DecodeMeta::x_default_meta();
    } else if(type == YoloType::V8) {
        normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
        meta_ = DecodeMeta::v8_default_meta();
    }
    // else if(type == YoloType::V5) {
    //     normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
    //     meta_ = DecodeMeta::v5_p5_default_meta();
    // }
    else {
        INFOE("Now unsupport type %d!!", type);
    }

    confidence_threshold_ = confidence_threshold;
    nms_threshold_ = nms_threshold;
    type_ = type;
    batch_size_ = batch_size;
    is_use_trtnNMSPlugin_ = is_use_trtnNMSPlugin;
    return ThreadSafedAsyncInferImpl::startup(std::make_tuple(file, gpuid));
}

// 重写基类worker 工作线程
void YoloTRTInferImpl::worker(std::promise<bool>& result) {
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

    const int MAX_IMAGE_BBOX = 1024;
    const int NUM_BOX_ELEMENT = 7;  // left, top, right, bottom, confidence, class, keepflag
    Tensor affin_matrix_device(FasterTRT::DataType::Float);
    Tensor output_array_device(FasterTRT::DataType::Float);
    Tensor prior_box(FasterTRT::DataType::Float);
    int max_batch_size = engine->get_max_batch_size();
    auto input = engine->tensor("images");
    auto output = engine->tensor("output");
    // decode data
    int num_classes, output_num_bboxes, output_fm_area;
    output_num_bboxes = output->size(0) * output->size(1);
    output_fm_area = output->size(2);
    if(type_ == YoloType::X) {
        num_classes = output->size(2) - 5;
    } else if(type_ == YoloType::V8) {
        num_classes = output->size(2) - 4;
    }

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

    // 这里的 1 + MAX_IMAGE_BBOX结构是，counter + bboxes ...
    output_array_device.set_stream(stream_);
    output_array_device.resize(max_batch_size, 1 + MAX_IMAGE_BBOX * NUM_BOX_ELEMENT).to_gpu();

    auto decode_kernel_invoker = yolox_decode_kernel_invoker;

    if(type_ == YoloType::X) {
        prior_box.resize(output->size(0) * output->size(1), 3).to_cpu();
        init_yolox_prior_box(prior_box);
        decode_kernel_invoker = yolox_decode_kernel_invoker;
    } else if(type_ == YoloType::V8) {
        // prior_box.resize(output->size(0) * output->size(1), 3).to_cpu();
        // init_yolov8_prior_box(prior_box);
        decode_kernel_invoker = yolov8_decode_kernel_invoker;
    } else {
        INFOE("now support yolox & yolov8 only! ");
    }

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
            auto& job = fetch_jobs[ibatch];
            float* image_based_output = output->gpu<float>(ibatch);
            float* output_array_ptr = output_array_device.gpu<float>(ibatch);
            auto affine_matrix = affin_matrix_device.gpu<float>(ibatch);
            checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(int), stream_));
            decode_kernel_invoker(image_based_output, output_num_bboxes, output_fm_area,
                                  num_classes, confidence_threshold_, nms_threshold_, affine_matrix,
                                  output_array_ptr, prior_box.gpu<float>(), MAX_IMAGE_BBOX,
                                  stream_);
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

// 初始化yolox prior
void YoloTRTInferImpl::init_yolox_prior_box(Tensor& prior_box) {
    // 8400(lxaxhxw) x 3
    float* prior_ptr = prior_box.cpu<float>();
    for(int ianchor = 0; ianchor < meta_.num_anchor; ++ianchor) {
        for(int ilevel = 0; ilevel < meta_.num_level; ++ilevel) {
            int stride = meta_.strides[ilevel];
            int fm_width = input_width_ / stride;
            int fm_height = input_height_ / stride;
            int anchor_abs_index = ilevel * meta_.num_anchor + ianchor;
            for(int ih = 0; ih < fm_height; ++ih) {
                for(int iw = 0; iw < fm_width; ++iw) {
                    *prior_ptr++ = iw;
                    *prior_ptr++ = ih;
                    *prior_ptr++ = stride;
                }
            }
        }
    }
    prior_box.to_gpu();
}

// 初始化yolov8 prior
void YoloTRTInferImpl::init_yolov8_prior_box(Tensor& prior_box) {
    // 8400 这个咋网络里已经做了
    return;
}

// 初始化yolov5 prior
void YoloTRTInferImpl::init_yolov5_prior_box(Tensor& prior_box) {
    // 25200(lxaxhxw) x 5
    float* prior_ptr = prior_box.cpu<float>();
    for(int ianchor = 0; ianchor < meta_.num_anchor; ++ianchor) {
        for(int ilevel = 0; ilevel < meta_.num_level; ++ilevel) {
            int stride = meta_.strides[ilevel];
            int fm_width = input_width_ / stride;
            int fm_height = input_height_ / stride;
            int anchor_abs_index = ilevel * meta_.num_anchor + ianchor;
            for(int ih = 0; ih < fm_height; ++ih) {
                for(int iw = 0; iw < fm_width; ++iw) {
                    *prior_ptr++ = iw;
                    *prior_ptr++ = ih;
                    *prior_ptr++ = meta_.w[anchor_abs_index];
                    *prior_ptr++ = meta_.h[anchor_abs_index];
                    *prior_ptr++ = stride;
                }
            }
        }
    }
    prior_box.to_gpu();
}

// 预处理
bool YoloTRTInferImpl::preprocess(Job& job, const cv::Mat& image) {
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

    // resize_bilinear_and_normalize(
    //     image_device, image.cols*3, image.cols, image.rows,
    //     tensor->gpu<float>(), input_width_, input_width_,
    //     normalize_, stream_
    // );
    warp_affine_bilinear_and_normalize_plane(image_device, image.cols * 3, image.cols, image.rows,
                                             tensor->gpu<float>(), input_width_, input_height_,
                                             affine_matrix_device, 114, normalize_, stream_pro_);
    // 这个地方需要同步，确保数据放到gpu后才可以吧任务提交到队列中。
    cudaStreamSynchronize(stream_pro_);

    return true;
}

// 提交<vector>任务
std::vector<std::shared_future<BoxArray>> YoloTRTInferImpl::commits(
    const std::vector<cv::Mat>& images) {
    return ThreadSafedAsyncInferImpl::commits(images);
}

// 提交cv::Mat任务
std::shared_future<BoxArray> YoloTRTInferImpl::commit(const cv::Mat& image) {
    return ThreadSafedAsyncInferImpl::commit(image);
}

bool compile(Mode mode, YoloType type, unsigned int max_batch_size,
             const std::string& source_onnx_file, const std::string& save_engine_file,
             size_t max_workspace_size, const std::string& int8_images_folder,
             const std::string& int8_entropy_calibrator_cache_file) {
    bool hasEntropyCalibrator = false;
    std::vector<uint8_t> entropyCalibratorData;
    std::vector<std::string> entropyCalibratorFiles;

    auto int8process = [=](int current, int count, const std::vector<std::string>& files,
                           std::shared_ptr<Tensor>& tensor) {
        for(int i = 0; i < files.size(); ++i) {
            auto& file = files[i];
            INFO("Int8 load %d / %d, %s", current + i + 1, count, file.c_str());

            auto image = cv::imread(file);
            if(image.empty()) {
                INFOE("Load image failed, %s", file.c_str());
                continue;
            }
            image_to_tensor(image, tensor, type, i);
        }
        tensor->synchronize();
    };

    if(mode == Mode::INT8) {
        if(!int8_entropy_calibrator_cache_file.empty()) {
            if(exists(int8_entropy_calibrator_cache_file)) {
                entropyCalibratorData = load_file(int8_entropy_calibrator_cache_file);
                if(entropyCalibratorData.empty()) {
                    INFOE("entropyCalibratorFile is set as: %s, but we read is empty.",
                          int8_entropy_calibrator_cache_file.c_str());
                    return false;
                }
                hasEntropyCalibrator = true;
            }
        }

        if(hasEntropyCalibrator) {
            if(!int8_images_folder.empty()) {
                INFOW(
                    "int8_images_folder is ignore, when int8_entropy_calibrator_cache_file is set");
            }
        } else {
            entropyCalibratorFiles = glob_image_files(int8_images_folder);
            if(entropyCalibratorFiles.empty()) {
                INFOE("Can not find any images(jpg/png/bmp/jpeg/tiff) from directory: %s",
                      int8_images_folder.c_str());
                return false;
            }

            if(entropyCalibratorFiles.size() < max_batch_size) {
                INFOW(
                    "Too few images provided, %d[provided] < %d[max batch size], image copy will "
                    "be performed",
                    entropyCalibratorFiles.size(), max_batch_size);

                int old_size = entropyCalibratorFiles.size();
                for(int i = old_size; i < max_batch_size; ++i)
                    entropyCalibratorFiles.push_back(entropyCalibratorFiles[i % old_size]);
            }
        }
    } else {
        if(hasEntropyCalibrator) {
            INFOW("int8_entropy_calibrator_cache_file is ignore, when Mode is '%s'",
                  mode_string(mode));
        }
    }

    INFO("Compile %s %s.", mode_string(mode), source_onnx_file.c_str());
    std::shared_ptr<IBuilder> builder(createInferBuilder(gLogger),
                                      destroy_nvidia_pointer<IBuilder>);
    if(builder == nullptr) {
        INFOE("Can not create builder.");
        return false;
    }

    std::shared_ptr<IBuilderConfig> config(builder->createBuilderConfig(),
                                           destroy_nvidia_pointer<IBuilderConfig>);
    if(mode == Mode::FP16) {
        if(!builder->platformHasFastFp16()) {
            INFOW("Platform not have fast fp16 support");
        }
        config->setFlag(BuilderFlag::kFP16);
    } else if(mode == Mode::INT8) {
        if(!builder->platformHasFastInt8()) {
            INFOW("Platform not have fast int8 support");
        }
        config->setFlag(BuilderFlag::kINT8);
    }

    std::shared_ptr<INetworkDefinition> network;
    std::shared_ptr<nvonnxparser::IParser> onnxParser;
    const auto explicitBatch =
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    network = std::shared_ptr<INetworkDefinition>(builder->createNetworkV2(explicitBatch),
                                                  destroy_nvidia_pointer<INetworkDefinition>);

    // from onnx is not markOutput
    onnxParser.reset(nvonnxparser::createParser(*network, gLogger),
                     destroy_nvidia_pointer<nvonnxparser::IParser>);
    if(onnxParser == nullptr) {
        INFOE("Can not create parser.");
        return false;
    }

    if(!onnxParser->parseFromFile(source_onnx_file.c_str(), 1)) {
        INFOE("Can not parse OnnX file: %s", source_onnx_file.c_str());
        return false;
    }

    auto inputTensor = network->getInput(0);
    auto inputDims = inputTensor->getDimensions();

    std::shared_ptr<Int8EntropyCalibrator> int8Calibrator;
    if(mode == Mode::INT8) {
        auto calibratorDims = inputDims;
        calibratorDims.d[0] = max_batch_size;

        if(hasEntropyCalibrator) {
            INFO("Using exist entropy calibrator data[%d bytes]: %s", entropyCalibratorData.size(),
                 int8_entropy_calibrator_cache_file.c_str());
            int8Calibrator.reset(
                new Int8EntropyCalibrator(entropyCalibratorData, calibratorDims, int8process));
        } else {
            INFO("Using image list[%d files]: %s", entropyCalibratorFiles.size(),
                 int8_images_folder.c_str());
            int8Calibrator.reset(
                new Int8EntropyCalibrator(entropyCalibratorFiles, calibratorDims, int8process));
        }
        config->setInt8Calibrator(int8Calibrator.get());
    }

    INFO("Input shape is %s",
         join_dims(std::vector<int>(inputDims.d, inputDims.d + inputDims.nbDims)).c_str());
    INFO("Set max batch size = %d", max_batch_size);
    INFO("Set max workspace size = %.2f MB", max_workspace_size / 1024.0f / 1024.0f);

    int net_num_input = network->getNbInputs();
    INFO("Network has %d inputs:", net_num_input);
    std::vector<std::string> input_names(net_num_input);
    for(int i = 0; i < net_num_input; ++i) {
        auto tensor = network->getInput(i);
        auto dims = tensor->getDimensions();
        auto dims_str = join_dims(std::vector<int>(dims.d, dims.d + dims.nbDims));
        INFO("      %d.[%s] shape is %s", i, tensor->getName(), dims_str.c_str());

        input_names[i] = tensor->getName();
    }

    int net_num_output = network->getNbOutputs();
    INFO("Network has %d outputs:", net_num_output);
    for(int i = 0; i < net_num_output; ++i) {
        auto tensor = network->getOutput(i);
        auto dims = tensor->getDimensions();
        auto dims_str = join_dims(std::vector<int>(dims.d, dims.d + dims.nbDims));
        INFO("      %d.[%s] shape is %s", i, tensor->getName(), dims_str.c_str());
    }

    int net_num_layers = network->getNbLayers();
    INFO("Network has %d layers", net_num_layers);
    builder->setMaxBatchSize(max_batch_size);
    config->setMaxWorkspaceSize(max_workspace_size);

    auto profile = builder->createOptimizationProfile();
    for(int i = 0; i < net_num_input; ++i) {
        auto input = network->getInput(i);
        auto input_dims = input->getDimensions();
        input_dims.d[0] = 1;
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
        input_dims.d[0] = max_batch_size;
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    }
    config->addOptimizationProfile(profile);

    INFO("Building engine...");
    auto time_start = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
    std::shared_ptr<ICudaEngine> engine(builder->buildEngineWithConfig(*network, *config),
                                        destroy_nvidia_pointer<ICudaEngine>);
    if(engine == nullptr) {
        INFOE("engine is nullptr");
        return false;
    }

    if(mode == Mode::INT8) {
        if(!hasEntropyCalibrator) {
            if(!int8_entropy_calibrator_cache_file.empty()) {
                INFO("Save calibrator to: %s", int8_entropy_calibrator_cache_file.c_str());
                save_file(int8_entropy_calibrator_cache_file,
                          int8Calibrator->getEntropyCalibratorData());
            } else {
                INFO("No set entropyCalibratorFile, and entropyCalibrator will not save.");
            }
        }
    }

    auto time_end = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
    INFO("Build done %lld ms !", time_end - time_start);

    // serialize the engine, then close everything down
    std::shared_ptr<IHostMemory> seridata(engine->serialize(), destroy_nvidia_pointer<IHostMemory>);
    return save_file(save_engine_file, seridata->data(), seridata->size());
}

void image_to_tensor(const cv::Mat& image, std::shared_ptr<Tensor>& tensor, YoloType type,
                     int ibatch) {
    Norm normalize;
    if(type == YoloType::V5) {
        normalize = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
    } else if(type == YoloType::X) {
        normalize = Norm::None();
    } else {
        INFOE("Unsupport type %d", type);
    }

    cv::Size input_size(tensor->size(3), tensor->size(2));
    AffineMatrix affine;
    affine.compute(image.size(), input_size);

    size_t size_image = image.cols * image.rows * 3;
    size_t size_matrix = upbound(sizeof(affine.d2i), 32);
    auto workspace = tensor->get_workspace();
    uint8_t* gpu_workspace = (uint8_t*)workspace->gpu(size_matrix + size_image);
    float* affine_matrix_device = (float*)gpu_workspace;
    uint8_t* image_device = size_matrix + gpu_workspace;

    uint8_t* cpu_workspace = (uint8_t*)workspace->cpu(size_matrix + size_image);
    float* affine_matrix_host = (float*)cpu_workspace;
    uint8_t* image_host = size_matrix + cpu_workspace;
    auto stream = tensor->get_stream();

    memcpy(image_host, image.data, size_image);
    memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
    checkCudaRuntime(
        cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream));
    checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i),
                                     cudaMemcpyHostToDevice, stream));

    warp_affine_bilinear_and_normalize_plane(
        image_device, image.cols * 3, image.cols, image.rows, tensor->gpu<float>(ibatch),
        input_size.width, input_size.height, affine_matrix_device, 114, normalize, stream);
}

// 创建推理器
std::shared_ptr<Infer> create_infer(const std::string& engine_file, YoloType type, int gpuid,
                                    int batch_size, float confidence_threshold,
                                    float nms_threshold) {
    std::shared_ptr<YoloTRTInferImpl> instance(new YoloTRTInferImpl());
    if(!instance->startup(engine_file, type, gpuid, batch_size, confidence_threshold,
                          nms_threshold)) {
        instance.reset();
    }
    return instance;
}

};  // namespace YOLO

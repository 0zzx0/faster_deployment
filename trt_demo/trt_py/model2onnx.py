import tensorrt as trt 
import torch
import torch.nn as nn
import onnx

class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=1)
        self.bn = nn.BatchNorm2d(3)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.conv(x)))
        return self.pool(x)


device = torch.device('cuda:0')
onnx_model_name = '../files/model.onnx'
torch.onnx.export(MyModule(), 
                  torch.randn(1, 3, 224, 224), 
                  onnx_model_name, 
                  input_names=['input'],
                  output_names=['output'], 
                  opset_version=11)


def ger_engine():
    torch.onnx.export(MyModule(), torch.randn(1, 3, 112, 112), onnx_model_name, input_names=['input'],
                    output_names=['output'], opset_version=11)

    onnx_model = onnx.load(onnx_model_name)

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    # EXPLICIT_BATCH 显式batch
    EXPLICIT_BATCH = 1 << (int)( 
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) 
    network = builder.create_network(EXPLICIT_BATCH)    # 创建network

    parser = trt.OnnxParser(network, logger)            # 解析onnx

    if not parser.parse(onnx_model.SerializePartialToString()):
        error_mags = ' '
        for error in range(parser.num_errors):
            error_mags += error
        raise RuntimeError(f"解析失败辣: {error_mags}")

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 20
    profile = builder.create_optimization_profile() 

    profile.set_shape('input', [1,3 ,112 ,112],[1,3 ,112 ,112],[1,3 ,112 ,112]) 
    config.add_optimization_profile(profile) 
    # create engine 
    with torch.cuda.device(device): 
        engine = builder.build_engine(network, config) 
    
    with open('model.engine', mode='wb') as f: 
        f.write(bytearray(engine.serialize())) 
        print("generating file done!") 



import os
import torch # 1.9.0+cpu
import torchvision.transforms as transforms
# import onnx # 1.10.1
import onnxruntime #1.8.1
from PIL import Image

#导入模型
# from model.MobilenetV3 import mobilenetv3_small as create_model
from model.shufflenetv2 import shufflenet_v2_x0_5 as create_model

# ----------------------------------------------------------------------------------------------------------- #
root_path = '/home/shanshan/Desktop/image_Clas_pytorch/pytorch2onnx/'  # 根目录路径
img_path = root_path + 'test_img/' + "test_demo.jpg"   # 测试图片路径
input_model_path = root_path + 'orign_model/' + 'MaskedFace_shufflenetv2_x0_5_400_coloraug_affine.pth'  # 源pytorch模型路径
out_onnx_model_path = root_path + 'result_model/' + 'MaskedFace_shufflenetv2_x0_5_400_coloraug_affine.onnx'  # 保存的onnx模型路径
# ----------------------------------------------------------------------------------------------------------- #

#生成onnx时指定输入shape, NCHW
input_shape = [1, 3, 224, 224]
#生成onnx模型时指定输入的名称，一般单输入不需要修改
input_name = "input"

#请按真实前处理方式填写
img_size = [input_shape[2], input_shape[3]]
data_transform = transforms.Compose(
        [transforms.Resize(img_size),
         transforms.ToTensor(),
         ]) 


#前处理, 请按真实前处理方式填写
def pre_processing(image_path):
    # load image
    assert os.path.exists(image_path), "file: '{}' dose not exist.".format(image_path)
    img = Image.open(image_path)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    print('*'* 80)
    print(">>> image shape :",img.shape)
    return img

#后处理, 请按实际后处理（解析数据、绘图等）改写代码
def post_processing(output, model_name):
    print(model_name, " \n>>> output =", output)

# 1.pytroch转onnx
def pth_2_onnx_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=3).to(device)
    model.load_state_dict(torch.load(input_model_path))
    model.eval()
    dummy_input = torch.randn(input_shape, device = device)
    input_names = [input_name]
    torch.onnx._export(model, dummy_input, out_onnx_model_path, export_params=True, verbose=False, do_constant_folding=True, opset_version=9, input_names=input_names)

# 2.测试pytorch模型前向推理
def run_pth_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    #前处理
    img = pre_processing(img_path)

    #初始化模型
    # 创建模型实例
    model = create_model(num_classes=3).to(device)
    # 加载模型权重
    model_weight_path = input_model_path
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        #前向推理
        output = model(img.to(device)).cpu()
        #后处理
        post_processing(output, model_weight_path)

# 3.测试onnx模型前向推理
def run_onnx_model():
    #前处理
    img = pre_processing(img_path)
    img = img.numpy()

    #onnxruntime 初始化
    ort_session = onnxruntime.InferenceSession(out_onnx_model_path)
    #onnxruntime 前向推理
    output = ort_session.run(None, {input_name:  img})

    #后处理
    post_processing(output, out_onnx_model_path)


if __name__ == '__main__':
    # 1.pytroch转onnx
    pth_2_onnx_model()

    # 2.测试pytorch模型输出
    run_pth_model()

    # 3.测试onnx模型输出
    run_onnx_model()
    print('*'* 80)


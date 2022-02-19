import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import time

# from model.MobilenetV3 import mobilenetv3_small as predict_model
from model.shufflenetv2 import shufflenet_v2_x0_5 as predict_model
# from model.pplcnet import PPLCNet_x0_5 as predict_model

# ----------------------------------------------------------------------------------------------------------- #
img_path = "./img_data/TestDataset/Mask_test/365807_Mask.jpg"    # 预测图片路径
weights_path = "./pretrain_model/MaskedFace_shufflenetv2_x0_5_400_coloraug_affine_SGD.pth"  # 预训练模型路径
json_path = './class_label/class_indices.json'  # 类别标签文件路径
# ----------------------------------------------------------------------------------------------------------- #

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 加载图像
    # img_path = "./img_data/Mask/365807_Mask.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # 拓展图像维度
    img = torch.unsqueeze(img, dim=0)

    # 读取类别标签文件
    # json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # 创建模型实例
    model = predict_model(num_classes=3).to(device)

    # 加载模型权重
    # weights_path = "./pretrain_model/MaskedFace_shufflenetv2_x0_5_400_coloraug_affine_SGD.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # 预测
    model.eval()
    start = time.time()
    with torch.no_grad():
        # 预测类别
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()
    end = time.time()
    print('预测总耗时：%.4f s' % (end-start))


if __name__ == '__main__':
    main()

import os
import json

import torch
from PIL import Image
import glob
from torchvision import transforms
import matplotlib.pyplot as plt
import logging
import time

# from model.MobilenetV3 import mobilenetv3_small as bt_predict_model
# from model.shufflenetv2 import shufflenet_v2_x0_5 as bt_predict_model
# from model.ghostnet import ghostnet as bt_predict_model
from model.shufflenetv2 import shufflenet_v2_x0_25 as bt_predict_model
# from model.pplcnet import PPLCNet_x0_5 as bt_predict_model

# ----------------------------------------------------------------------------------------------------------- #
weights_path = "./pretrain_model/MaskedFace_shufflenet_v2_x0_25_400_coloraug_affine.pth"  # 预测模型文件路径
json_path = './class_label/class_indices.json'   # 数据标签路径
log_path = './log/batch_predict.log'   # 日志保存路径
# ----------------------------------------------------------------------------------------------------------- #

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 输出日志到终端
    # sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    # logger.addHandler(sh)

    return logger

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 加载批量图像路径
    # img_path_list = ["./test_image/02116_Mask.jpg", "./test_image/04634.png","./test_image/05077_Mask_Mouth_Chin.jpg"]
    img_path_list = './img_data/TestDataset/IMFD_test'
    img_path_list = glob.glob(os.path.join(img_path_list, '*.jpg'))


    img_list = []
    for img_path in img_path_list:
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        plt.imshow(img)
        img = data_transform(img)
        img_list.append(img)

    # 图像批处理
    batch_img = torch.stack(img_list, dim=0)

    # 读取类别标签
    # json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # 创建模型实例
    model = bt_predict_model(num_classes=3).to(device)

    # 加载模型权重
    # weights_path = "./pretrain_model/MaskedFace_shufflenetv2_x0_5_400_coloraug_affine_SGD.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # 预测
    model.eval()
    # 启动日志记录
    logger = get_logger(log_path)
    print('日志记录启动!')
    start = time.time()
    with torch.no_grad():
        # 预测类别
        output = model(batch_img.to(device)).cpu()
        predict = torch.softmax(output, dim=1)
        probs, classes = torch.max(predict, dim=1)

        for idx, (pro, cla) in enumerate(zip(probs, classes)):
            print_res = "image: {}  class: {}  prob: {:.3}".format(img_path_list[idx],
                                                             class_indict[str(cla.numpy())],
                                                             pro.numpy())
            # 日志记录
            logging.info(print_res)
            print(print_res)
            # plt.title(print_res)
            # plt.show()
    print('日志记录已完成!')
    end = time.time()
    print('预测总耗时：%.04f s' % (end-start))

if __name__ == '__main__':
    main()

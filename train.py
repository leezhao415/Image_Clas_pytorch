import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import logging

# from model.MobilenetV3 import mobilenetv3_small as pre_model
# from model.shufflenetv2 import shufflenet_v2_x0_5 as pre_model
from model.shufflenetv2 import shufflenet_v2_x0_25 as pre_model
# from model.ghostnet import ghostnet as pre_model
# from model.pplcnet import PPLCNet_x0_5 as pre_model

# ----------------------------------------------------------------------------------------------------------- #
model_weight_path = "./pretrain_model/shufflenetv2_x0.5-f707e7126e.pth"   # 预训练模型路径
save_path = './pretrain_model/MaskedFace_shufflenet_v2_x0_25_400_coloraug_affine.pth'   # 保存路径
class_label_path = './class_label/class_indices.json'  # 类别标签文件路径
epochs = 400
log_path = './log/train_shufflenet_v2_x0_25_400.log'
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
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ColorJitter(brightness=0.5, hue=0.5),
                                     transforms.RandomAffine(degrees=(10, 30), translate=(0.25, 0.5), scale=(1.2, 2.0)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    # transforms.RandomAffine(degrees=(10, 30), translate=(0.25, 0.5), scale=(1.2, 2.0)),
    data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))  # 获取数据根目录
    image_path = os.path.join(data_root, "dataset", "MaskedFaceData")  # 数据路径设置
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'类别1':0, '类别2':1, '类别3':2, '类别4':3, '类别5':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # 类别json文件操作
    json_str = json.dumps(cla_dict, indent=4)
    with open(class_label_path, 'w') as json_file:
        json_file.write(json_str)

    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = pre_model(num_classes = 3)
    # 加载预训练权重
    # model_weight_path = "./pretrain_model/shufflenetv2_x0.5-f707e7126e.pth"   # 预训练模型路径
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    # # 改变fc层结构
    # net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # input_channels = net.fc.output_channels
    # net.fc = nn.Linear(input_channels, 3)

    # 删除classifier权重
    pre_weights = torch.load(model_weight_path, map_location=device)
    pre_dict = {k:v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys =net.load_state_dict(pre_dict, strict=False)

    # # 冻结特征权重
    # for param in net.parameters():
    #     param.requires_grad = False

    net.to(device)

    # 定义损失函数
    loss_function = nn.CrossEntropyLoss()
    # 定义优化器
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)
    # optimizer = optim.SGD(params, lr=0.0001, momentum=0.8, weight_decay=1e-4)

    ## 学习率衰减
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)         # 指数衰减
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # 等间隔衰减
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 100], 0.5)      # 按需衰减

    # epochs = 400
    best_acc = 0.0
    # save_path = './pretrain_model/MaskedFace_shufflenetv2_x0_5_400_coloraug_affine.pth'   # 保存路径
    train_steps = len(train_loader)

    logger = get_logger(log_path)
    print('日志记录启动!')
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))

            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            ## 学习率更新
            # scheduler.step()
            ## 打印状态量
            running_loss += loss.item()

            train_bar.desc = "Train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)
        # 模型验证
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "Valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        # 日志记录
        logging.info('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    print('日志记录已完成!')
    print('训练已完成！')


if __name__ == '__main__':
    main()

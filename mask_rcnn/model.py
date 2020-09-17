import torchvision_learn
from torchvision_learn.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision_learn.models.detection.mask_rcnn import MaskRCNNPredictor
import torch.nn as nn

def get_model_instance_segmentation(num_classes):
    # 加载在coco上预训练好的实例分割模型
    model = torchvision_learn.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # 获得分类器的输入特征数
    in_feature = model.roi_heads.box_predictor.cls_score.in_features
    # 用新的头部替换预先训练好的头部
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feature, num_classes=num_classes)

    # 现在获取掩码分类器的输入特征数
    in_feature_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # 用新的掩码预测器替换掩码预测期
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feature_mask, hidden_layer, num_classes)
    return model

import mask_rcnn.detection.transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

from mask_rcnn.detection.engine import train_one_epoch, evaluate
import mask_rcnn.detection.utils as utils
import mask_rcnn.dataset as mydataset
import torch.utils.data
import os
import torch

def main():
    # cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes=2
    #  使用我们的数据集和定义的转换
    dataset = mydataset.PennFudanDataset('../PennFudanPed', get_transform(train=True))
    dataset_test = mydataset.PennFudanDataset('../PennFudanPed', get_transform(train=False))

    # 在训练和测试集中拆分数据集
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # 定义训练和验证数据加载器
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn
    )

    # 使用我们的辅助函数获得模型
    model = get_model_instance_segmentation(num_classes=num_classes)

    # cuda
    # model.cuda()
    model = nn.DataParallel(model, device_ids=[0,2])

    # 构建一个优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # 学习率调度程序
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 训练10个epoch
    num_epochs = 10
    for epoch in range(num_epochs):
        # 训练一个epoch，每10个迭代打印一次
        train_one_epoch(model, optimizer, data_loader, device=device,
                        epoch = epoch, print_freq=10)
        # 更新学习率
        lr_scheduler.step()
        # 在测试集上评价
        evaluate(model, data_loader_test, device=device)

    print("That's is!")

if __name__ == '__main__':
    main()
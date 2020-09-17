###############################################################################
###################### 场景1 微调已经预训练的模型 ################################
###############################################################################
# 我们想要预训练好的模型，只微调最后一层
import torchvision_learn
from torchvision_learn.models.detection.faster_rcnn import FastRCNNPredictor

# 在COCO上加载经过预训练的模型
model = torchvision_learn.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# 将分类器替换为用户定义的num_classes的新分类器
num_classes = 2 # 1 class(person) + backgroud
# 获得分类器的输入参数数量
in_features = model.roi_heads.box_predictor.cls_score.in_features
# 用新的头部替换预先训练好的头部
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

###############################################################################
###################### 场景2 修改模型以添加不同的主干 ############################
###############################################################################
# 我们想要用不同的模型替换主干
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# 加载预先训练的模型进行分类和返回
# 只有功能
backbone = torchvision.models.mobilenet_v2(pretrained=True).features

# # print(backbone)
# backbone1 = torchvision.models.resnet50(pretrained=True)
# # print(backbone1)
# # print(type(backbone1))
# model_ft = torchvision.models.resnet18(pretrained=True) #加载预训练网络

# FasterRCNN需要知道骨干网中的输出通道数量。对于mobilenet_v2，他是1280，所以我们需要在这里添加它
backbone.out_channels = 1280

# 我们让RPN在每个空间位置生成5x3个锚点
# 具有5种不同大小和3种不用的宽高比
# 我们有一个元祖[元祖[int]]
# 因为每个特征映射可能具有不同的大小和宽高比
anchor_generotor = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0), ))

# 定义一下我们将用于执行感兴趣区域裁剪的特征映射，以及重新缩放后裁剪的大小
# 如果你的主干返回Tensor，则featmap_names应为[0]
# 更一般地，主干应该返回OrderedDict[Tensor]
# 并且在featmap_names中，你可以选择要使用的功能映射
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                      output_size=7,
                                                      sampling_ratio=2)
# 将这些pieces放在fasterrcnn模型中
model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generotor, box_roi_pool=roi_pooler)
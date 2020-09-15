from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)
print(dir(torchvision.models))

# os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'

# 顶级数据目录。 这里我们假设目录的格式符合ImageFolder结构
data_dir = "./hymenoptera_data"
# 从[resnet, alexnet, vgg, squeezenet, densenet, inception]中选择模型
model_name = "squeezenet"
# 数据集中类别数量
num_classes = 2
# 训练的批量大小（根据您的内存量而变化）
batch_size = 8
# 你要训练的epoch数
num_epochs = 15
# 用于特征提取的标志。当为FALSE时，我们微调整个模型,更新所有模型参数
# 当为TRUE时，我们只更新最后一层的参数
feature_extract = True

# train_model函数处理给定模型的训练和验证，
# 作为输入，它需要pytorch模型，数据加载器字典，损失函数，优化器，用于训练和验证epoch数，以及当模型是初始模型时的布尔标志
# 这个函数训练指定数量的epoch,并且在每个epoch之后运行完整的验证步骤。 它还跟踪最佳性能的模型(从验证准确率方面），并在训练结束时返回性能最好的模型。 在每个epoch之后，打印训练和验证正确率。
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
      since = time.time()

      val_acc_history = []

      best_model_wts = copy.deepcopy(model.state_dict())
      best_acc = 0.0

      for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                  if phase == 'train':
                        model.train()
                  else:
                        model.eval()

                  running_loss = 0.0
                  running_corrects = 0

                  # Iterate over data
                  for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        # inputs = inputs.cuda()
                        # labels = labels.cuda()


                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                              # Get model outputs and calculate loss
                              # Special case for inception beacause in training it has an auxiliary output.
                              # In train mode, we calculate the loss by summing the final output and the auxiliary output
                              # but in testing we only consider the final output
                              if is_inception and phase == 'train':
                                    # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                                    outputs, aux_outputs = model(inputs)
                                    loss1 = criterion(outputs, labels)
                                    loss2 = criterion(aux_outputs, labels)
                                    loss = loss1 + 0.4 * loss2
                              else:
                                    outputs = model(inputs)
                                    loss = criterion(outputs, labels)

                              _, preds = torch.max(outputs, 1)

                              # backward + optimize only if in training phase
                              if phase == 'train':
                                    loss.backward()
                                    optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                  epoch_loss = running_loss / len(dataloaders[phase].dataset)
                  epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                  print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                  # deep copy the model
                  if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                  if phase == 'val':
                        val_acc_history.append(epoch_acc)

            print()

      time_elapsed = time.time() - since
      print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
      print('Best val Acc: {:4f}'.format(best_acc))

      # load best model weights
      model.load_state_dict(best_model_wts)
      return model, val_acc_history


# 如果只是想要特征提取并且只想为新初始化的层计算梯度
def set_parameter_requires_grad(model, feature_extracting):
      if feature_extracting:
            for param in model.parameters():
                  param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
          """ Resnet18"""
          model_ft = models.resnet18(pretrained=use_pretrained)
          set_parameter_requires_grad(model_ft, feature_extract)
          num_ftrs = model_ft.fc.in_features
          model_ft.fc = nn.Linear(num_ftrs, num_classes)
          input_size = 224

    elif model_name == "alexnet":
        """ Alexnet"""
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn"""
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
 """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet"""
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
print(model_ft)
# print(model_ft.classifier[2])


# 数据加载
data_transforms = {
    'train' : transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val' : transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

print("Initializing Datasets and Dataloaders...")

# 创建训练和验证数据集
image_datasets = {
    x : datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in  ['train', 'val']
}

# 创建训练和验证数据加载器
dataloaders_dict = {
    x : torch.utils.data.DataLoader(image_datasets[x],
                                    batch_size= batch_size,
                                    shuffle = True,
                                    num_workers = 4) for x in ['train', 'val']
}
# 检测我们是否有可用的GPU
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")


# 创建优化器
model_ft = model_ft.to(device)
# model_ft = model_ft.cuda()
# model_ft = nn.DataParallel(model_ft, device_ids=[0,2])

# 在此运行中收集要优化/更新的参数
# 如果我们正在进行微调，我们将更新所有参数
# 但如果我们正在提取特征，我们只会更新刚刚初始化的参数，即‘requires_grad’的参数为TRUE
params_to_update = model_ft.parameters()
print("Parameters to learn:")
if feature_extract:
    params_to_update=[]
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print('\t', name)
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# 运行训练和验证
criterion = nn.CrossEntropyLoss()
# train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name == "inception"))

# Initialize the non-pretrained version of the model used for this run
scratch_model,_ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
scratch_model = scratch_model.to(device)
scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
scratch_criterion = nn.CrossEntropyLoss()
_,scratch_hist = train_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer, num_epochs=num_epochs, is_inception=(model_name=="inception"))

# Plot the training curves of validation accuracy vs. number
#  of training epochs for the transfer learning method and
#  the model trained from scratch
ohist = []
shist = []

ohist = [h.cpu().numpy() for h in hist]
shist = [h.cpu().numpy() for h in scratch_hist]

plt.title("Validation Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1,num_epochs+1),ohist,label="Pretrained")
plt.plot(range(1,num_epochs+1),shist,label="Scratch")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.show()
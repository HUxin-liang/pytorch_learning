import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

# 定义模型
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化模型
model = TheModelClass()
# 初始化优化器
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# 打印模型的状态字典
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
# 打印优化器的状态字典
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

###### 随意设 ######
PATH = ''
args = ''
kwargs = ''
epoch = ''
loss = ''
TheOptimizerClass=''
modelA = ''
TheModelBClass = ''



####################### 保存和读取state_dict ###############################
# 保存state_dict
torch.save(model.state_dict(), PATH)
# 加载state_dict
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()

####################### 保存和读取完整模型 ###############################
# 保存完整模型
torch.save(model, PATH)
# 加载完整模型
# 模型类必须在此之前被定义
model = torch.load(PATH)
model.eval()

####################### 保存和读取checkpoint ###############################
#当保存成 Checkpoint 的时候，可用于推理或者是继续训练，保存的不仅仅是模型的 state_dict
# 。保存优化器的 state_dict 也很重要, 因为它包含作为模型训练更新的缓冲区和参数。你也许
# 想保存其他项目，比如最新记录的训练损失，外部的 torch.nn.Embedding 层等等。
# 保存checkpoint-->.tar格式
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    # ...
    }, PATH)
# 加载checkpoint
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.eval()
# - or -
model.train()

####################### 保存和读取-热启动 ###############################
# 保存
torch.save(modelA.state_dict(), PATH)
# 加载
modelB = TheModelBClass(*args, **kwargs)
# 无论是从缺少某些键的 state_dict 加载还是从键的数目多于加载模型的 state_dict , 都可以通过在
# load_state_dict() 函数中将 strict 参数设置为 False 来忽略非匹配键的函数。
modelB.load_state_dict(torch.load(PATH), strict=False)

import torch.nn as nn
import torch.nn.functional as F

class CNet(nn.Module):  # 继承父类
    # def __init__(self):  # 构造方法
    #     super(FirstNet, self).__init__()  # 调用父类的构造方法
    #     # super().__init__()
    #     #  定义属性方法等
    #     self.conv1 = nn.Conv2d(1, 11, 5)  # 1灰度图片的通道，  10：输出通道，5：kernel 5*5
    #     self.conv2 = nn.Conv2d(11, 22, 3)  # 10输入通道，  20：输出通道，3：kernel 3*3
    #     self.fc1 = nn.Linear(22 * 10 * 10, 550)  # 全连接层为线性层 20*10*10：输入通道， 500：输出通道
    #     self.fc2 = nn.Linear(550, 11)  # 500：输入通道， 10：输出通道
    #
    # def forward(self, x):  # 定义前向传播
    #     input_size = x.size(0)  # batch_size(16){ * 1(灰度) * 28(像素) * 28}
    #     x = self.conv1(x)  # 输入：batch*1*28*28, 输出：batch*10*24*24 (28 - 5 + 1)
    #     x = F.relu(x)  # 保持shape不变, 输出：batch*10*24*24
    #     x = F.max_pool2d(x, 2, 2)  # (将采样)输入：batch*10*24*24， 输出：batch*10*12*12
    #
    #     x = self.conv2(x)  # 输入：batch*10*12*12 输出：batch*20*10*10
    #     x = F.relu(x)
    #
    #     x = x.view(input_size, -1)  # 拉平， -1自动计算维度，20*10*10=2000
    #
    #     x = self.fc1(x)  # 输入：batch*2000 输出：batch*500
    #     x = F.relu(x)  # 保持shape不变
    #
    #     x = self.fc2(x)  # 输入：batch*500 输出：batch*10
    #
    #     output = F.log_softmax(x, dim=1)  # （损失函数）计算分类后，每个数字的概率值
    #
    #     return output
    def __init__(self):  # 构造方法
        super(CNet, self).__init__()  # 调用父类的构造方法
        # super().__init__()
        #  定义属性方法等
        self.conv1 = nn.Conv2d(1, 11, 5)  # 1灰度图片的通道，  10：输出通道，5：kernel 5*5

        self.fc1 = nn.Linear(11 * 12 * 12, 11)  # 全连接层为线性层 20*10*10：输入通道， 500：输出通道
        self.fc2 = nn.Linear(396, 11)  # 500：输入通道， 10：输出通道

    def forward(self, x):  # 定义前向传播
        input_size = x.size(0)  # batch_size(16){ * 1(灰度) * 28(像素) * 28}
        x = self.conv1(x)  # 输入：batch*1*28*28, 输出：batch*11*24*24 (28 - 5 + 1)
        x = F.relu(x)  # 保持shape不变, 输出：batch*11*24*24
        x = F.max_pool2d(x, 2, 2)  # (将采样)输入：batch*11*24*24， 输出：batch*10*12*12

        x = x.view(input_size, -1)  # 拉平， -1自动计算维度，11*12*12=1584

        x = self.fc1(x)  # 输入：batch*1584 输出：batch*396
        # x = F.relu(x)  # 保持shape不变
        #
        # x = self.fc2(x)  # 输入：batch*396 输出：batch*11

        output = F.log_softmax(x, dim=1)  # （损失函数）计算分类后，每个数字的概率值

        return output
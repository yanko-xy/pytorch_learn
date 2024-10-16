import torch
import torch.nn as nn


# 生成数据
sample_nums = 100
mean_value = 1.7
bias = 1
n_data = torch.ones(sample_nums, 2)
x0 = torch.normal(mean_value * n_data, 1) + bias
y0 = torch.zeros(sample_nums)
x1 = torch.normal(-mean_value * n_data, 1) + bias
y1 = torch.ones(sample_nums)
train_x = torch.cat((x0, x1), 0)
train_y = torch.cat((y0, y1), 0)



# 选择模型
class LR(nn.models):
    def __init__(self):
        super(LR, self).__init__()
        self.features = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)
        return x


lr_net = LR()


# 选择损失函数
loss_fn = nn.BCELoss()


# 选择优化器
lr = 0.01 # 学习率
optimizer = torch.optim.SGD(lr_net.parameters(), lr=lr, momentum=0.9)

# 训练模型
for iteration in range(1000):

    # 向前传播
    y_pred= lr_net.forward(train_x)

    # 计算loss
    loss = loss_fn(y_pred.squeeze(), train_y)

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    # 绘图
    if iteration % 20 == 0:
        mask = y_pred.ge(0.5).float().squeeze()
        connect = (mask == train_y).sum()
        acc = connect.item() / train_y.size(0)

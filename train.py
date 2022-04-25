# 环境准备
import numpy as np  # numpy数组库
import time as time

import torch  # torch基础库
import torch.nn as nn  # torch神经网络库

import torchvision.datasets as dataset  # 公开数据集的下载和管理
import torchvision.transforms as transforms  # 公开数据集的预处理库,格式转换

import torch.utils.data as data_utils  # 对数据集进行分批加载的工具集
import torchvision.models as models
import gc

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.backends.cudnn.version())
from cutout import Cutout
import mixup
import cutmix
from torch.utils.tensorboard import SummaryWriter

# 数据增强方法

print('''数据增强方法：
                    无 ：0
                    cutout： 1
                    mixup： 2 
                    cutmix：3''')
method = int(input())

# 2-1 准备数据集
# 数据集格式转换


transform_train = transforms.Compose(
    [transforms.Resize(256),  # transforms.Scale(256)
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_test = transforms.Compose(
    [transforms.Resize(256),  # transforms.Scale(256)
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

if method == 1:
    transform_train.transforms.append(Cutout(length=8))

# 训练数据集
train_data = dataset.CIFAR100(root="../datasets/cifar100",
                              train=True,
                              transform=transform_train,
                              download=True)

# 测试数据集
test_data = dataset.CIFAR100(root="../datasets/cifar100",
                             train=False,
                             transform=transform_test,
                             download=True)

# 批量数据读取
batch_size = 32


train_loader = data_utils.DataLoader(dataset=train_data,  # 训练数据
                                     batch_size=batch_size,  # 每个批次读取的图片数量
                                     shuffle=True)  # 读取到的数据，是否需要随机打乱顺序

test_loader = data_utils.DataLoader(dataset=test_data,  # 测试数据集
                                    batch_size=batch_size,
                                    shuffle=True)

# 2-3 定义网络
# 直接使用框架提供的预定义模型，设定输出分类的种类数100，默认为1000分类
net = models.resnet18(num_classes=100)

# 定义优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

# 3-3 训练前准备
# Assume that we are on a CUDA machine, then this should print a CUDA device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 把网络转移到GPU
net.to(device)
loss_fn=nn.CrossEntropyLoss()
# 把loss计算转移到GPU
loss_fn = loss_fn.to(device)






def train(Learning_rate=0.1, loss_fn=nn.CrossEntropyLoss(), reg=0, epochs=20, train_loader=train_loader,
          test_loader=test_loader,fre=0):
    print(loss_fn)

    # optimizer = SGD： 基本梯度下降法
    # parameters：指明要优化的参数列表
    # optimizer = torch.optim.Adam(model.parameters(), lr = Learning_rate)
    # 定义优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=Learning_rate, momentum=0.9, weight_decay=reg)

    # 3-3 训练前准备
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 把网络转移到GPU
    net.to(device)

    # 把loss计算转移到GPU
    loss_fn = loss_fn.to(device)

    # 定义迭代次数

    train_loss_history = []  # 训练过程中的loss数据
    train_accuracy_history = []  # 中间的预测结果

    accuracy_batch = 0.0

    # 3-4 模型训练
    train_start = time.time()
    print('train start at {}'.format(train_start))

    correct_dataset = 0
    total_dataset = 0
    accuracy_dataset = 0.0

    writer = SummaryWriter('runs/train')
    for i in range(0, epochs):

        epoch_start = time.time()
        net.train()
        for j, (x_train, y_train) in enumerate(train_loader):

            # 指定数据处理的硬件单元
            x_train = x_train.to(device)
            # x_train = x_train.cuda()
            y_train = y_train.to(device)
            # y_train = y_train.cuda()

            if method == 0 or method == 1:
                optimizer.zero_grad()
                y_pred = net(x_train)
                train_loss = loss_fn(y_pred, y_train)
            if method == 2:
                x_train, y_a, y_b, lam = mixup.mixup_data(x_train, y_train)
                optimizer.zero_grad()
                y_pred = net(x_train)
                train_loss = mixup.mixup_criterion(loss_fn, y_pred, y_a, y_b, lam)
            if method == 3:
                r = np.random.rand(1)
                if r < 0.5:  # 做cutmix的概率为0.5
                    x_train, y_a, y_b, lam = cutmix.cutmix_data(x_train, y_train)
                    optimizer.zero_grad()
                    y_pred = net(x_train)
                    train_loss = cutmix.cutmix_criterion(loss_fn, y_pred, y_a, y_b, lam)
                else:
                    y_pred = net(x_train)
                    train_loss = loss_fn(y_pred, y_train)

            # (3) 反向求导
            train_loss.backward()

            # (4) 反向迭代
            optimizer.step()

            # 记录训练过程中的损失值
            train_loss_history.append(train_loss.item())  # loss for a batch

            # 记录训练过程中的准确率
            number_batch = y_train.size()[0]  # 图片的个数
            _, predicted = torch.max(y_pred.data, dim=1)
            correct_batch = (predicted == y_train).sum().item()  # 预测正确的数目
            accuracy_batch = 100 * correct_batch / number_batch
            train_accuracy_history.append(accuracy_batch)

            writer.add_scalar('train_loss', train_loss.item(), global_step=int((i+fre) * len(train_data) / batch_size) + j)
            writer.add_scalar('train_acc', accuracy_batch, global_step=int((i+fre) * len(train_data) / batch_size) + j)

            if (j % 100 == 0):
                print('train: epoch {} batch {} In {} loss = {:.4f} accuracy = {:.4f}%'.format(i+fre, j,
                                                                                               len(train_data) / batch_size,
                                                                                               train_loss.item(),
                                                                                               accuracy_batch))
        net.eval()
        # 进行评测的时候网络不更新梯度
        with torch.no_grad():
            for k, data in enumerate(test_loader):
                # 获取一个batch样本"
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                # 对batch中所有样本进行预测
                outputs = net(images)
                test_loss = loss_fn(outputs, labels)

                # 对batch中每个样本的预测结果，选择最可能的分类
                _, predicted = torch.max(outputs.data, 1)

                # 对batch中的样本数进行累计
                total_dataset += labels.size()[0]

                # 对batch中的所有结果进行比较"
                bool_results = (predicted == labels)

                # 统计预测正确样本的个数
                correct_dataset += bool_results.sum().item()

                # 统计预测正确样本的精度
                accuracy_dataset = 100 * correct_dataset / total_dataset
                writer.add_scalar('test_loss', test_loss.item(), global_step=int((i+fre) * len(test_data) / batch_size) + k)
                writer.add_scalar('test_acc', accuracy_dataset, global_step=int((i+fre) * len(test_data) / batch_size) + k)

                if (k % 100 == 0):
                    print(
                        'test: epoch {} batch {} In {} loss= {:.4f} accuracy = {:.4f}%'.format(i+fre, k,
                                                                                               len(test_data) / 32,
                                                                                               test_loss.item(),
                                                                                               accuracy_dataset))
        for p in optimizer.param_groups:
            p['lr'] *= 0.95     # 学习率下降
        epoch_end = time.time()
        epoch_cost = epoch_end - epoch_start
        print('epoch {} cost {}s '.format(i, epoch_cost))
        gc.collect()
        torch.cuda.empty_cache()

    writer.close()


train(Learning_rate=0.01, reg=1e-4, epochs=15, train_loader=train_loader, test_loader=test_loader)

# 在baseline方法下进一步增大准确率并保存模型
if method==0:
    torch.save(net.state_dict(), "resnet_model_cifar100.pkl")
    net.load_state_dict(torch.load('resnet_model_cifar100.pkl'))  # 装载上传训练的参数

    models = net.modules()

    for p in models:
        if p._get_name() != 'Linear':
            p.requires_grad_ = False

    train(Learning_rate=0.001, reg=1e-4, epochs=3, train_loader=train_loader, test_loader=test_loader, fre=12)

    torch.save(net.state_dict(), "resnet_model_cifar100.pkl")



import matplotlib.pyplot as plt  # 画图库
import torch  # torch基础库
import torchvision.datasets as dataset  # 公开数据集的下载和管理
import torchvision.transforms as transforms  # 公开数据集的预处理库,格式转换
import torchvision.utils as utils
import torch.utils.data as data_utils  # 对数据集进行分批加载的工具集
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.backends.cudnn.version())
from cutout import Cutout
import mixup
import cutmix


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
batch_size = 64

train_loader = data_utils.DataLoader(dataset=train_data,  # 训练数据
                                     batch_size=batch_size,  # 每个批次读取的图片数量
                                     shuffle=True)  # 读取到的数据，是否需要随机打乱顺序

test_loader = data_utils.DataLoader(dataset=test_data,  # 测试数据集
                                    batch_size=batch_size,
                                    shuffle=True)



print("获取3张图片")
imgs, labels = next(iter(train_loader))
imgs=imgs[0:3]
print(imgs.shape)


images = utils.make_grid(imgs)

images = images.numpy().transpose(1, 2, 0)


plt.imshow(images)
plt.show()

cut=Cutout(n_holes=3)
imgs1=[]
for i in range(3):
    out=cut(imgs[i])
    imgs1.append(out)


images = utils.make_grid(imgs1)
images = images.numpy().transpose(1, 2, 0)
plt.imshow(images)
plt.show()





imgs2 , y_a, y_b, lam =mixup.mixup_data(imgs,labels)
images = utils.make_grid(imgs2)
images = images.numpy().transpose(1, 2, 0)
plt.imshow(images)
plt.show()


imgs3, y_a, y_b, lam = cutmix.cutmix_data(imgs, labels)
images = utils.make_grid(imgs3)
images = images.numpy().transpose(1, 2, 0)
plt.imshow(images)
plt.show()
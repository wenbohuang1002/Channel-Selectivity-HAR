import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import *
from selective_convolution import SelectiveConv2d
from thop import profile
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from torchstat import stat
import sklearn.metrics as sm
# torch.cuda.set_device(0)
n_gpu = torch.cuda.device_count()
print(n_gpu)

train_x = np.load('experiments/unimib/train_x.npy')
train_y = np.load('experiments/unimib/train_y_p.npy')
test_x = np.load('experiments/unimib/test_x.npy')
test_y = np.load('experiments/unimib/test_y_p.npy')

print("\nShape of train_x:",train_x.shape,
      "\nShape of train_y:",train_y.shape,
      "\nShape of test_x:",test_x.shape,
      "\nShape of test_y:",test_y.shape,)

train_x = np.reshape(train_x, [-1, 1, 151, 3])
test_x = np.reshape(test_x, [-1, 1, 151, 3])
train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)
test_x = torch.from_numpy(test_x)
test_y = torch.from_numpy(test_y)
print("\nShape of train_x:",train_x.shape,
      "\nShape of train_y:",train_y.shape,
      "\nShape of test_x:",test_x.shape,
      "\nShape of test_y:",test_y.shape,)

# use_gpu = torch.cuda.is_available()
batchSize = 64
# num_batches = math.ceil((np.size(train_x,0)/batchSize))
# print(num_batches)
torch_dataset = Data.TensorDataset(train_x,train_y)
train_loader = Data.DataLoader(dataset=torch_dataset,batch_size=batchSize,shuffle=True,num_workers=0)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

class Net_SC(nn.Module):
    def __init__(self):
            super(Net_SC, self).__init__()
        # message = input("Use Selective_Convolution?[Y or N]")
        # if message in ['y', 'Y']:
            self.Block1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(6, 1), stride=(2, 1), padding=(1, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            )
            self.shortcut1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(6, 1), stride=(2, 1), padding=(1, 0)),
                nn.BatchNorm2d(128),
            )
            self.ca1 = ChannelAttention(128)
            self.Block2 = nn.Sequential(
                SelectiveConv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0),
                                dropout_rate=0.01, gamma=0.0005, K=16, N_max=32),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                SelectiveConv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0),
                                dropout_rate=0.01, gamma=0.0005, K=32, N_max=32),
                nn.BatchNorm2d(256),
                nn.ReLU(True)
            )
            self.shortcut2 = nn.Sequential(
                SelectiveConv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0),
                                dropout_rate=0.1, gamma=0.0005, K=16, N_max=32),
                nn.BatchNorm2d(256),
            )
            self.ca2 = ChannelAttention(256)
            # self.sa1 = SpatialAttention()
            self.Block3 = nn.Sequential(
                SelectiveConv2d(in_channels=256, out_channels=384, kernel_size=(6, 2), stride=(2, 1), padding=(1, 0),
                                dropout_rate=0.1, gamma=0.0005, K=32, N_max=64),
                nn.BatchNorm2d(384),
                nn.ReLU(True),
                SelectiveConv2d(in_channels=384, out_channels=384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0),
                                dropout_rate=0.1, gamma=0.0005, K=64, N_max=64),
                nn.BatchNorm2d(384),
                nn.ReLU(True)
            )
            self.shortcut3 = nn.Sequential(
                SelectiveConv2d(in_channels=256, out_channels=384, kernel_size=(6, 2), stride=(2, 1), padding=(1, 0),
                                dropout_rate=0.1, gamma=0.0005, K=32, N_max=64),
                nn.BatchNorm2d(384),
            )
            self.ca3 = ChannelAttention(384)
            # self.Block4 = nn.Sequential(
            #     SelectiveConv2d(in_channels=256, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(2, 1),
            #                     dropout_rate=0.1, gamma=0.0005, K=16, N_max=32),
            #     nn.BatchNorm2d(256),
            #     nn.ReLU(True),
            #     SelectiveConv2d(in_channels=256, out_channels=256, kernel_size=(6, 1), stride=(1, 1), padding=(2, 0),
            #                     dropout_rate=0.1, gamma=0.0005, K=16, N_max=32),
            #     nn.BatchNorm2d(256),
            #     nn.ReLU(True)
            # )
            # self.shortcut4 = nn.Sequential(
            #     SelectiveConv2d(in_channels=256, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1),
            #                     dropout_rate=0.1, gamma=0.0005, K=16, N_max=32),
            #     nn.BatchNorm2d(256),
            # )
            # self.ca4 = ChannelAttention(256)
            # self.Block5 = nn.Sequential(
            #     SelectiveConv2d(in_channels=256, out_channels=384, kernel_size=(3, 2), stride=(2, 1), padding=(1, 2),
            #                     dropout_rate=0.1, gamma=0.0005, K=16, N_max=32),
            #     nn.BatchNorm2d(384),
            #     nn.ReLU(True),
            #     SelectiveConv2d(in_channels=384, out_channels=384, kernel_size=(3, 2), stride=(1, 2), padding=(1, 3),
            #                     dropout_rate=0.1, gamma=0.0005, K=16, N_max=32),
            #     nn.BatchNorm2d(384),
            #     nn.ReLU(True)
            # )
            # self.shortcut5 = nn.Sequential(
            #     SelectiveConv2d(in_channels=256, out_channels=384, kernel_size=(3, 2), stride=(2, 1), padding=(1, 0),
            #                     dropout_rate=0.1, gamma=0.0005, K=16, N_max=32),
            #     nn.BatchNorm2d(384),
            # )
            # self.ca5 = ChannelAttention(384)
            # self.Block6 = nn.Sequential(
            #     SelectiveConv2d(in_channels=384, out_channels=384, kernel_size=(3, 2), stride=(2, 1), padding=(1, 2),
            #                     dropout_rate=0.1, gamma=0.0005, K=32, N_max=64),
            #     nn.BatchNorm2d(384),
            #     nn.ReLU(True),
            #     SelectiveConv2d(in_channels=384, out_channels=384, kernel_size=(3, 2), stride=(1, 2), padding=(1, 3),
            #                     dropout_rate=0.1, gamma=0.0005, K=32, N_max=64),
            #     nn.BatchNorm2d(384),
            #     nn.ReLU(True)
            # )
            # self.shortcut6 = nn.Sequential(
            #     SelectiveConv2d(in_channels=384, out_channels=384, kernel_size=(3, 2), stride=(2, 1), padding=(1, 0),
            #                     dropout_rate=0.1, gamma=0.0005, K=32, N_max=64),
            #     nn.BatchNorm2d(384),
            # )
            # self.ca6 = ChannelAttention(384)
            # self.sa3 = SpatialAttention()
            self.fc = nn.Sequential(
                nn.Linear(8448, 17)
            )
        # elif message in ['n', 'N']:
        #     self.layer1 = nn.Sequential(
        #         nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(6,2), stride=(3,1), padding=1),
        #         nn.BatchNorm2d(64),
        #         nn.ReLU(True)
        #     )
        #     self.ca1 = ChannelAttention(64)
        #     self.sa1 = SpatialAttention()
        #     self.layer2 = nn.Sequential(
        #         nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6,2), stride=(3,1), padding=1),
        #         nn.BatchNorm2d(128),
        #         nn.ReLU(True)
        #     )
        #     self.ca2 = ChannelAttention(128)
        #     self.sa2 = SpatialAttention()
        #     self.layer3 = nn.Sequential(
        #         nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6,2), stride=(3,1), padding=1),
        #         nn.BatchNorm2d(256),
        #         nn.ReLU(True)
        #     )
        #     self.ca3 = ChannelAttention(256)
        #     self.sa3 = SpatialAttention()
        #     self.fc = nn.Sequential(
        #         nn.Linear(7680, 17)
        #     )
        # else:
        #     exit(1)

    def forward(self, x):
        # print(x.shape)
        h1 = self.Block1(x)
        # print(h1.shape)
        r = self.shortcut1(x)
        # print(r.shape)
        h1 = h1 + r
        h1 = self.ca1(h1) * h1
        # print(h1.shape)
        h2 = self.Block2(h1)
        # print(h2.shape)
        r = self.shortcut2(h1)
        # print(r.shape)
        h2 = h2 + r
        h2 = self.ca2(h2) * h2
        # print(h2.shape)
        h3 = self.Block3(h2)
        # print(h3.shape)
        r = self.shortcut3(h2)
        # print(r.shape)
        h3 = h3 + r
        h3 = self.ca3(h3) * h3
        # h4 = self.Block4(h3)
        # r = self.shortcut4(h3)
        # h4 = h4 + r
        # h4 = self.ca4(h4) * h4
        # # print(h4.shape)
        # h5 = self.Block5(h4)
        # # print(h5.shape)
        # r = self.shortcut5(h4)
        # # print(r.shape)
        # h5 = h5 + r
        # h5 = self.ca5(h5) * h5
        # h6 = self.Block6(h5)
        # r = self.shortcut6(h5)
        # h6 = h6 + r
        # h6 = self.ca6(h6) * h6
        x = h3.view(h3.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        x = nn.LayerNorm(x.size())(x.cpu())
        x = x.cuda()
        # x = F.normalize(x.cuda())
        # print(x.shape)
        return x
        #
        # try:
        #     output = Net_SC(x)
        # except RuntimeError as exception:
        #     if "out of memory" in str(exception):
        #         print("WARNING: out of memory")
        #         if hasattr(torch.cuda, 'empty_cache'):
        #             torch.cuda.empty_cache()
        #     else:
        #         raise exception

lr_list = []
LR = 0.001
net = Net_SC().cuda()
net = nn.DataParallel(net,device_ids=[0,1])
opt = torch.optim.Adam(net.parameters(),lr=LR,weight_decay=1e-5)
loss_func = nn.CrossEntropyLoss().cuda()
params = list(net.parameters())
scheduler = lr_scheduler.ExponentialLR(opt, gamma=0.9)
# flops, params = profile(net, input_size=(64,1,128,9))
k = 0
for i in params:
    l = 1
    print("该层的结构：" + str(list(i.size())))
    for j in i.size():
        l *= j
    print("该层参数和：" + str(l))
    k = k + l
print("总参数数量和：" + str(k))
epoch_list = []
accuracy_list = []
loss_list = []
def flat(data):
    data=np.argmax(data,axis=1)
    return  data
for epoch in range(2):
    net.train()
    for step,(x,y) in enumerate(train_loader):
        x = x.type(torch.FloatTensor)
        x,y=x.cuda(),y
        output = net(x)
        # y = to_one_hot(y)
        # y = to_one_hot(y).cpu()

        y = flat(y).cuda()

        # print(output.shape, y.shape)
        loss = loss_func(output,y.long())

        net.zero_grad()
        opt.zero_grad()
        loss.backward()
        opt.step()

    # opt.zero_grad()
    # loss.backward()
    # opt.step()
    # print('test')
    if epoch%1 ==0:
        # for data, target in tqdm(train_loader):
            net.eval()
            test_x = test_x.type(torch.FloatTensor)
            test_out = net(test_x.cuda())
        # print('test:',test_out.shape)
            pred_y = torch.max(test_out,1)[1].data.squeeze().cuda()

        # print('pred_y:',pred_y.shape,'test_y:',test_y.shape)
            scheduler.step()
            lr_list.append(opt.state_dict()['param_groups'][0]['lr'])
            accuracy = (torch.sum(pred_y == flat(test_y.float()).cuda()).type(torch.FloatTensor) / test_y.size(0)).cuda()
            print('Epoch: ', epoch,  '| test accuracy: %.6f' % accuracy,'|loss:%.6f'%loss,'| params:',str(k))
    epoch_list.append(epoch)
    accuracy_list.append(accuracy.item())
    loss_list.append(loss.item())
print('Epoch_list:',epoch_list,'Accuracy_list:',accuracy_list,'Loss_list:',loss_list)
cm = sm.confusion_matrix(pred_y.cpu().numpy(), flat(test_y.float()).cpu().numpy())
print(cm)
np.save('Store/Unimib_R/confusion_matrix_resc.npy',cm)
# np.save('Store/Unimib_R/epoch_resc.npy',epoch_list)
# np.save('Store/Unimib_R/accuracy_resc.npy',accuracy_list)
# np.save('Store/Unimib_R/loss_resc.npy',loss_list)

# x = epoch_list
# y1 = accuracy_list
# y2 = loss_list
# plt.plot(x,y1,label = 'Accuracy')
# plt.plot(x,y2,label = 'Loss')
# plt.title('Line')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()

model = Net_SC()
stat(model, (1, 151, 3))

#          module name  input shape output shape     params memory(MB)          MAdd         Flops  MemRead(B)  MemWrite(B) duration[%]   MemR+W(B)
# 0           layer1.0    1 151   3  128  74   3      896.0       0.11     340,992.0     198,912.0      5396.0     113664.0       9.24%    119060.0
# 1           layer1.1  128  74   3  128  74   3      256.0       0.11     113,664.0      56,832.0    114688.0     113664.0       1.72%    228352.0
# 2           layer1.2  128  74   3  128  74   3        0.0       0.11      28,416.0      28,416.0    113664.0     113664.0       0.48%    227328.0
# 3       ca1.avg_pool  128  74   3  128   1   1        0.0       0.00           0.0           0.0         0.0          0.0       1.17%         0.0
# 4       ca1.max_pool  128  74   3  128   1   1        0.0       0.00           0.0           0.0         0.0          0.0       1.35%         0.0
# 5            ca1.fc1  128   1   1    8   1   1     1024.0       0.00       2,040.0       1,024.0      4608.0         32.0       1.10%      4640.0
# 6          ca1.relu1    8   1   1    8   1   1        0.0       0.00           8.0           8.0        32.0         32.0       0.26%        64.0
# 7            ca1.fc2    8   1   1  128   1   1     1024.0       0.00       1,920.0       1,024.0      4128.0        512.0       0.94%      4640.0
# 8        ca1.sigmoid  128   1   1  128   1   1        0.0       0.00           0.0           0.0         0.0          0.0       0.62%         0.0
# 9      layer2.0.norm  128  74   3  128  74   3      256.0       0.11     113,664.0      56,832.0    114688.0     113664.0       1.36%    228352.0
# 10     layer2.0.drop  128  74   3  128  74   3        0.0       0.11           0.0           0.0         0.0          0.0       0.24%         0.0
# 11     layer2.0.conv  128  74   3  128  36   3    98304.0       0.05  21,219,840.0  10,616,832.0    506880.0      55296.0      11.03%    562176.0
# 12          layer2.1  128  36   3  128  36   3      256.0       0.05      55,296.0      27,648.0     56320.0      55296.0       0.75%    111616.0
# 13          layer2.2  128  36   3  128  36   3        0.0       0.05      13,824.0      13,824.0     55296.0      55296.0       0.29%    110592.0
# 14      ca2.avg_pool  128  36   3  128   1   1        0.0       0.00           0.0           0.0         0.0          0.0       0.72%         0.0
# 15      ca2.max_pool  128  36   3  128   1   1        0.0       0.00           0.0           0.0         0.0          0.0       0.89%         0.0
# 16           ca2.fc1  128   1   1    8   1   1     1024.0       0.00       2,040.0       1,024.0      4608.0         32.0       0.98%      4640.0
# 17         ca2.relu1    8   1   1    8   1   1        0.0       0.00           8.0           8.0        32.0         32.0       0.25%        64.0
# 18           ca2.fc2    8   1   1  128   1   1     1024.0       0.00       1,920.0       1,024.0      4128.0        512.0       0.94%      4640.0
# 19       ca2.sigmoid  128   1   1  128   1   1        0.0       0.00           0.0           0.0         0.0          0.0       0.24%         0.0
# 20     layer3.0.norm  128  36   3  128  36   3      256.0       0.05      55,296.0      27,648.0     56320.0      55296.0       0.91%    111616.0
# 21     layer3.0.drop  128  36   3  128  36   3        0.0       0.05           0.0           0.0         0.0          0.0       0.19%         0.0
# 22     layer3.0.conv  128  36   3  256  11   3   196608.0       0.03  12,967,680.0   6,488,064.0    841728.0      33792.0      10.72%    875520.0
# 23          layer3.1  256  11   3  256  11   3      512.0       0.03      33,792.0      16,896.0     35840.0      33792.0       0.71%     69632.0
# 24          layer3.2  256  11   3  256  11   3        0.0       0.03       8,448.0       8,448.0     33792.0      33792.0       0.28%     67584.0
# 25      ca3.avg_pool  256  11   3  256   1   1        0.0       0.00           0.0           0.0         0.0          0.0       0.69%         0.0
# 26      ca3.max_pool  256  11   3  256   1   1        0.0       0.00           0.0           0.0         0.0          0.0       0.86%         0.0
# 27           ca3.fc1  256   1   1   16   1   1     4096.0       0.00       8,176.0       4,096.0     17408.0         64.0       1.09%     17472.0
# 28         ca3.relu1   16   1   1   16   1   1        0.0       0.00          16.0          16.0        64.0         64.0       0.26%       128.0
# 29           ca3.fc2   16   1   1  256   1   1     4096.0       0.00       7,936.0       4,096.0     16448.0       1024.0       1.00%     17472.0
# 30       ca3.sigmoid  256   1   1  256   1   1        0.0       0.00           0.0           0.0         0.0          0.0       0.25%         0.0
# 31     layer4.0.norm  256  11   3  256  11   3      512.0       0.03      33,792.0      16,896.0     35840.0      33792.0       0.77%     69632.0
# 32     layer4.0.drop  256  11   3  256  11   3        0.0       0.03           0.0           0.0         0.0          0.0       0.19%         0.0
# 33     layer4.0.conv  256  11   3  256   3   3   393216.0       0.01   7,075,584.0   3,538,944.0   1606656.0       9216.0      10.11%   1615872.0
# 34          layer4.1  256   3   3  256   3   3      512.0       0.01       9,216.0       4,608.0     11264.0       9216.0       0.71%     20480.0
# 35          layer4.2  256   3   3  256   3   3        0.0       0.01       2,304.0       2,304.0      9216.0       9216.0       0.27%     18432.0
# 36      ca4.avg_pool  256   3   3  256   1   1        0.0       0.00           0.0           0.0         0.0          0.0       0.73%         0.0
# 37      ca4.max_pool  256   3   3  256   1   1        0.0       0.00           0.0           0.0         0.0          0.0       0.68%         0.0
# 38           ca4.fc1  256   1   1   16   1   1     4096.0       0.00       8,176.0       4,096.0     17408.0         64.0       1.00%     17472.0
# 39         ca4.relu1   16   1   1   16   1   1        0.0       0.00          16.0          16.0        64.0         64.0       0.24%       128.0
# 40           ca4.fc2   16   1   1  256   1   1     4096.0       0.00       7,936.0       4,096.0     16448.0       1024.0       0.96%     17472.0
# 41       ca4.sigmoid  256   1   1  256   1   1        0.0       0.00           0.0           0.0         0.0          0.0       0.26%         0.0
# 42     layer5.0.norm  256   3   3  256   3   3      512.0       0.01       9,216.0       4,608.0     11264.0       9216.0       0.54%     20480.0
# 43     layer5.0.drop  256   3   3  256   3   3        0.0       0.01           0.0           0.0         0.0          0.0       0.19%         0.0
# 44     layer5.0.conv  256   3   3  384   2   2   589824.0       0.01   4,717,056.0   2,359,296.0   2368512.0       6144.0       9.35%   2374656.0
# 45          layer5.1  384   2   2  384   2   2      768.0       0.01       6,144.0       3,072.0      9216.0       6144.0       0.60%     15360.0
# 46          layer5.2  384   2   2  384   2   2        0.0       0.01       1,536.0       1,536.0      6144.0       6144.0       0.27%     12288.0
# 47      ca5.avg_pool  384   2   2  384   1   1        0.0       0.00           0.0           0.0         0.0          0.0       0.72%         0.0
# 48      ca5.max_pool  384   2   2  384   1   1        0.0       0.00           0.0           0.0         0.0          0.0       0.71%         0.0
# 49           ca5.fc1  384   1   1   24   1   1     9216.0       0.00      18,408.0       9,216.0     38400.0         96.0       1.14%     38496.0
# 50         ca5.relu1   24   1   1   24   1   1        0.0       0.00          24.0          24.0        96.0         96.0       0.26%       192.0
# 51           ca5.fc2   24   1   1  384   1   1     9216.0       0.00      18,048.0       9,216.0     36960.0       1536.0       1.00%     38496.0
# 52       ca5.sigmoid  384   1   1  384   1   1        0.0       0.00           0.0           0.0         0.0          0.0       0.25%         0.0
# 53     layer6.0.norm  384   2   2  384   2   2      768.0       0.01       6,144.0       3,072.0      9216.0       6144.0       0.58%     15360.0
# 54     layer6.0.drop  384   2   2  384   2   2        0.0       0.01           0.0           0.0         0.0          0.0       0.18%         0.0
# 55     layer6.0.conv  384   2   2  384   1   1   884736.0       0.00   1,769,088.0     884,736.0   3545088.0       1536.0      10.39%   3546624.0
# 56          layer6.1  384   1   1  384   1   1      768.0       0.00       1,536.0         768.0      4608.0       1536.0       0.58%      6144.0
# 57          layer6.2  384   1   1  384   1   1        0.0       0.00         384.0         384.0      1536.0       1536.0       0.26%      3072.0
# 58      ca6.avg_pool  384   1   1  384   1   1        0.0       0.00           0.0           0.0         0.0          0.0       0.70%         0.0
# 59      ca6.max_pool  384   1   1  384   1   1        0.0       0.00           0.0           0.0         0.0          0.0       0.64%         0.0
# 60           ca6.fc1  384   1   1   24   1   1     9216.0       0.00      18,408.0       9,216.0     38400.0         96.0       1.14%     38496.0
# 61         ca6.relu1   24   1   1   24   1   1        0.0       0.00          24.0          24.0        96.0         96.0       0.25%       192.0
# 62           ca6.fc2   24   1   1  384   1   1     9216.0       0.00      18,048.0       9,216.0     36960.0       1536.0       1.04%     38496.0
# 63       ca6.sigmoid  384   1   1  384   1   1        0.0       0.00           0.0           0.0         0.0          0.0       0.27%         0.0
# 64              fc.0          384           17     6545.0       0.00      13,039.0       6,528.0     27716.0         68.0       1.50%     27784.0
# total                                           2232849.0       1.07  48,709,103.0  24,424,544.0     27716.0         68.0     100.00%  10701212.0
# =================================================================================================================================================
# Total params: 2,232,849
# -------------------------------------------------------------------------------------------------------------------------------------------------
# Total memory: 1.07MB
# Total MAdd: 48.71MMAdd
# Total Flops: 24.42MFlops
# Total MemR+W: 10.21MB
#
#
# Process finished with exit code 0

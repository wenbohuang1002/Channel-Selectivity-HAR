import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import *
from selective_convolution import SelectiveConv2d
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from torchstat import stat
torch.cuda.set_device(0)
n_gpu = torch.cuda.device_count()
print(n_gpu)

train_x = np.load('experiments/UCI/np_train_x.npy')
train_y = np.load('experiments/UCI/np_train_y.npy')
test_x = np.load('experiments/UCI/np_test_x.npy')
test_y = np.load('experiments/UCI/np_test_y.npy')

print("\nShape of train_x:",train_x.shape,
      "\nShape of train_y:",train_y.shape,
      "\nShape of test_x:",test_x.shape,
      "\nShape of test_y:",test_y.shape,)

train_x = np.reshape(train_x, [-1, 1, 128, 9])
test_x = np.reshape(test_x, [-1, 1, 128, 9])
train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)
test_x = torch.from_numpy(test_x)
test_y = torch.from_numpy(test_y)
print("\nShape of train_x:",train_x.shape,
      "\nShape of train_y:",train_y.shape,
      "\nShape of test_x:",test_x.shape,
      "\nShape of test_y:",test_y.shape,)

batchSize = 64
epoch_list = []
accuracy_list = []
loss_list = []

torch_dataset = Data.TensorDataset(train_x,train_y)
train_loader = Data.DataLoader(dataset=torch_dataset,batch_size=batchSize,shuffle=True,num_workers=0)

class Net_SC(nn.Module):
    def __init__(self):
            super(Net_SC, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            )
            self.layer2 = nn.Sequential(
                SelectiveConv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0),
                                dropout_rate=0.01, gamma=0.0005, K=8, N_max=16),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            )
            self.layer3= nn.Sequential(
                SelectiveConv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1),
                                dropout_rate=0.01, gamma=0.0005, K=8, N_max=16),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            )
            self.layer4 = nn.Sequential(
                SelectiveConv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0),
                                dropout_rate=0.01, gamma=0.0005, K=8, N_max=16),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            )
            self.layer5 = nn.Sequential(
                SelectiveConv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1),
                                dropout_rate=0.1, gamma=0.0005, K=32, N_max=64),
                nn.BatchNorm2d(256),
                nn.ReLU(True)
            )
            self.layer6 = nn.Sequential(
                SelectiveConv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0),
                                dropout_rate=0.1, gamma=0.0005, K=32, N_max=64),
                nn.BatchNorm2d(256),
                nn.ReLU(True)
            )
            self.fc = nn.Sequential(
                nn.Linear(15360, 6)
            )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0), -1)
        # x = nn.LayerNorm(x.size())(x.cpu())
        # x = x.cuda()
        # x = F.normalize(x.cuda())
        x = self.fc(x)
        return x

lr_list = []
LR = 0.001
net = Net_SC().cuda()
net = nn.DataParallel(net,device_ids=[0,1])
opt = torch.optim.Adam(net.parameters(),lr=LR,weight_decay=1e-5)
loss_func = nn.CrossEntropyLoss().cuda()
params = list(net.parameters())
scheduler = lr_scheduler.ExponentialLR(opt, gamma=0.9)
k = 0
for i in params:
    l = 1
    print("该层的结构：" + str(list(i.size())))
    for j in i.size():
        l *= j
    print("该层参数和：" + str(l))
    k = k + l
print("总参数数量和：" + str(k))

def flat(data):
    data=np.argmax(data,axis=1)
    return  data
for epoch in range(200):
    net.train()
    for step,(x,y) in enumerate(train_loader):
        x = x.type(torch.FloatTensor)
        x,y=x.cuda(),y
        output = net(x)
        y = flat(y).cuda()
        loss = loss_func(output,y.long())
        net.zero_grad()
        opt.zero_grad()
        loss.backward()
        opt.step()
    if epoch%1==0:
            net.eval()
            test_x = test_x.type(torch.FloatTensor)
            test_out = net(test_x.cuda())
            pred_y = torch.max(test_out,1)[1].data.squeeze().cuda()
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
np.save('Store/UCI_R1/confusion_matrix_sc.npy',cm)
np.save('Store/UCI_R1/epoch_sc.npy',epoch_list)
np.save('Store/UCI_R1/accuracy_sc.npy',accuracy_list)
np.save('Store/UCI_R1/loss_sc.npy',loss_list)

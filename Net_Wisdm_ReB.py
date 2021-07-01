import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchstat import stat
import sklearn.metrics as sm
torch.cuda.set_device(1)
n_gpu = torch.cuda.device_count()
print(n_gpu)

train_x = np.load('experiments/wisdm/x_train.npy')
train_y = np.load('experiments/wisdm/y_train.npy')
test_x = np.load('experiments/wisdm/x_test.npy')
test_y = np.load('experiments/wisdm/y_test.npy')

print("\nShape of train_x:",train_x.shape,
      "\nShape of train_y:",train_y.shape,
      "\nShape of test_x:",test_x.shape,
      "\nShape of test_y:",test_y.shape,)

train_x = np.reshape(train_x, [-1, 1, 200, 3])
test_x = np.reshape(test_x, [-1, 1, 200, 3])
train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)
test_x = torch.from_numpy(test_x)
test_y = torch.from_numpy(test_y)
print("\nShape of train_x:",train_x.shape,
      "\nShape of train_y:",train_y.shape,
      "\nShape of test_x:",test_x.shape,
      "\nShape of test_y:",test_y.shape,)

batchSize = 64
torch_dataset = Data.TensorDataset(train_x,train_y)
train_loader = Data.DataLoader(dataset=torch_dataset,batch_size=batchSize,shuffle=True,num_workers=0)

class Net_SC(nn.Module):
    def __init__(self):
            super(Net_SC, self).__init__()
            self.Block1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(6, 1), stride=(2, 1), padding=(0, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            )
            self.shortcut1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(6, 1), stride=(2, 1), padding=(0, 0)),
                nn.BatchNorm2d(128),
            )
            self.Block2 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(2, 1), padding=(0, 0)),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(256),
                nn.ReLU(True)
            )
            self.shortcut2 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(2, 1), padding=(0, 0)),
                nn.BatchNorm2d(256),
            )
            self.Block3 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(6, 2), stride=(2, 1), padding=(0, 0)),
                nn.BatchNorm2d(384),
                nn.ReLU(True),
                nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(384),
                nn.ReLU(True)
            )
            self.shortcut3 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(6, 2), stride=(2, 1), padding=(0, 0)),
                nn.BatchNorm2d(384),
            )
            self.fc = nn.Sequential(
                nn.Linear(16128, 17)
            )

    def forward(self, x):
        h1 = self.Block1(x)
        r = self.shortcut1(x)
        h1 = h1 + r
        h2 = self.Block2(h1)
        r = self.shortcut2(h1)
        h2 = h2 + r
        h3 = self.Block3(h2)
        r = self.shortcut3(h2)
        h3 = h3 + r
        x = h3.view(h3.size(0), -1)
        x = self.fc(x)
        # x = nn.LayerNorm(x.size())(x.cpu())
        # x = x.cuda()
        x = F.normalize(x.cuda())
        return x

lr_list = []
LR = 0.001
net = Net_SC().cuda()
net = nn.DataParallel(net,device_ids=[0,1])
opt = torch.optim.Adam(net.parameters(),lr=LR,weight_decay=1e-6)
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
        y = flat(y).cuda()
        loss = loss_func(output,y.long())
        net.zero_grad()
        opt.zero_grad()
        loss.backward()
        opt.step()
    if epoch%1 ==0:
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
np.save('Store/Wisdm_R/confusion_matrix_reb.npy',cm)
np.save('Store/Wisdm_R/epoch_resc.npy',epoch_list)
np.save('Store/Wisdm_R/accuracy_resc.npy',accuracy_list)
np.save('Store/Wisdm_R/loss_resc.npy',loss_list)

model = Net_SC()
stat(model, (1, 200, 3))


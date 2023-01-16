import dataloader_v1
import torch
import model as model
import numpy as np
import torchvision
import tqdm
import matplotlib.pyplot as plt
import copy
import sys
from torch.utils.data import DataLoader
from torchvision import transforms
# -------------------------------------------
from load_save import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_img = transforms.Compose([transforms.Resize((512,512))
                                , transforms.ToTensor()
                                , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                
transform_seg = transforms.Compose([transforms.Resize((512,512))
                                , transforms.ToTensor()])

# check point location
ckpt_dir = './ckpt'

# param
st_epoch = 0
num_epoch = 20
lr = 0.0001
batch_size = 16

# data, dataloader
train_data = dataloader_v1.Dataset('/root/test/dataset/Train/',transform_img=transform_img, transform_seg=transform_seg)
train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True)

val_data = dataloader_v1.Dataset('/root/test/dataset/Val/',transform_img=transform_img, transform_seg=transform_seg)
val_dataloader = DataLoader(val_data, batch_size=2, shuffle=True)

# network
net = model.UNet()
net.to(device)
# optim
optim = torch.optim.Adam(net.parameters(), lr=0.0001)

# loss
fn_loss = torch.nn.BCEWithLogitsLoss().to(device)


for epoch in range(st_epoch + 1, num_epoch + 1):
    net.train()
    loss_arr = []    

    for batch, data in enumerate(tqdm.tqdm(train_dataloader)):
        label = data['label'].to(device)
        input = data['input'].to(device)
        
        # print(label.shape)
        # print(input.shape)


        output = net(input)
        # print(output.dtype)
        # print(label.dtype)


        # backward pass
        optim.zero_grad()
        #오차 역전파에 사용하는 계산량을 줄여서 처리 속도를 높임

        a = torch.squeeze(output)
        # print(a.size())
        tf = transforms.ToPILImage()
        img = tf(a)
        # img.show()
        
        loss = fn_loss(output, label)
        loss.backward()
        optim.step()

        loss_arr += [loss.item()]

        print(f"TRAIN : EPOCH {epoch :04d} // BATCH {batch : 04d} / {len(train_data) : 04d} / LOSS {np.mean(loss_arr) : .4f}")



    with torch.no_grad():           #gradient계산 context를 비활성화, 필요한 메모리 감소, 연산속도 증가
        net.eval()                  #batch normalization과 dropout등과 같은 학습에만 필요한 기능 비활성화 추론할때의 상태로 조정(메모리와는 연관 없음)
        
        loss_arr = []
        best_model_wts = copy.deepcopy(net.state_dict())
        for batch, data in enumerate(val_dataloader, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            loss_arr += [loss.item()]

            print(
                f"VALID : EPOCH {epoch :04d} // BATCH {batch : 04d} / {len(val_data) : 04d} // LOSS {np.mean(loss_arr) : .4f}")
        
    if epoch % 20 == 0:
        save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)
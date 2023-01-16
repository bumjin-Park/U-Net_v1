import os
import numpy as np
import matplotlib.pyplot as plt
import dataloader_v1
import torch
import model as model
import torchvision
import tqdm
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

dataset_test = dataloader_v1.Dataset(data_dir=('/root/test/dataset/Test/'),transform_img=transform_img, transform_seg=transform_seg)
loader_test = DataLoader(dataset_test, batch_size=2, shuffle=False)

# 그밖에 부수적인 variables 설정하기
num_data_test = len(dataset_test)
num_batch_test = np.ceil(num_data_test / 2) # 나눈 값은 batch size

ckpt_dir = './ckpt'

# 결과 디렉토리 생성하기
result_dir = os.path.join('./', 'result')
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

net = model.UNet()
net.to(device)

optim = torch.optim.Adam(net.parameters(), lr=0.0001)

fn_loss = torch.nn.BCEWithLogitsLoss().to(device)

net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

with torch.no_grad():
      net.eval()
      loss_arr = []

      for batch, data in enumerate(loader_test, 1):
          # forward pass
          label = data['label'].to(device)
          input = data['input'].to(device)

          output = net(input)

          # 손실함수 계산하기
          loss = fn_loss(output, label)

          loss_arr += [loss.item()]

          print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                (batch, num_batch_test, np.mean(loss_arr)))

          # 테스트 결과 저장하기

          for j in range(label.shape[0]):
              id = num_batch_test * (batch - 1) + j

              plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze().to('cpu'), cmap='gray')
            #   print(input[j].permute(1,2,0).shape)
            #   sys.exit()
              plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].permute(1,2,0).squeeze().to('cpu'), cmap='gray')
              plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze().to('cpu'), cmap='gray')

              np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze().to('cpu'))
              np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze().to('cpu'))
              np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze().to('cpu'))

print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
        (batch, num_batch_test, np.mean(loss_arr)))




lst_data = os.listdir(os.path.join(result_dir, 'numpy'))

lst_label = [f for f in lst_data if f.startswith('label')]
lst_input = [f for f in lst_data if f.startswith('input')]
lst_output = [f for f in lst_data if f.startswith('output')]

lst_label.sort()
lst_input.sort()
lst_output.sort()

##
id = 0

label = np.load(os.path.join(result_dir,"numpy", lst_label[id]))
input = np.load(os.path.join(result_dir,"numpy", lst_input[id]))
output = np.load(os.path.join(result_dir,"numpy", lst_output[id]))

## 플롯 그리기
plt.figure(figsize=(8,6))
plt.subplot(131)
plt.imshow(input, cmap='gray')
plt.title('input')

plt.subplot(132)
plt.imshow(label, cmap='gray')
plt.title('label')

plt.subplot(133)
plt.imshow(output, cmap='gray')
plt.title('output')

plt.show()
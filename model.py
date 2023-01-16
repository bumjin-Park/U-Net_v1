import torch.nn as nn
import torch 
import torchsummary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNet(nn.Module):  
    def __init__(self): 
        super(UNet, self).__init__()
        def ConvRelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)]
            layers += [nn.ReLU()]
            layers += [nn.BatchNorm2d(num_features=out_channels)] 
            
            CR = nn.Sequential(*layers)

            return CR

        # Contracting path
        self.enc1_1 = ConvRelu(3, 64) # enc -> encoder  부분
        self.enc1_2 = ConvRelu(64, 64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = ConvRelu(64, 128)
        self.enc2_2 = ConvRelu(128, 128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.enc3_1 = ConvRelu(128, 256)
        self.enc3_2 = ConvRelu(256, 256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = ConvRelu(256, 512)
        self.enc4_2 = ConvRelu(512, 512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = ConvRelu(512, 1024)

        # Expansive path
        self.dec5_1 = ConvRelu(1024, 512)   #dec -> decoder부분

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0)

        self.dec4_2 = ConvRelu(2 * 512, 512)
        self.dec4_1 = ConvRelu(512, 256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0)

        self.dec3_2 = ConvRelu(2 * 256, 256)
        self.dec3_1 = ConvRelu(256, 128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0)
        
        self.dec2_2 = ConvRelu(2 * 128, 128)
        self.dec2_1 = ConvRelu(128, 64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)

        self.dec1_2 = ConvRelu(2 * 64, 64)
        self.dec1_1 = ConvRelu(64, 64)
        
        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        
# __init__ 함수에서 선언한 layer들 연결해서 data propa flow 만들기
    def forward(self, x):
        # Contracting path
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)
        
        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)
        
        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)
        
        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)
        
        # Expansive path

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        return x
    
model = UNet()
print(torchsummary.summary(model,(3,512,512),device='cpu'))
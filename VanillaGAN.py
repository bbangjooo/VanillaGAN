import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch import nn,cuda,optim
from torchvision import transforms
from torch.autograd import Variable

# Settings
download_root='mnist'
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,),std=(0.5,))
])

device= 'cuda' if cuda.is_available() else 'cpu'

batch_size=60
leraing_rate=0.0002
# Dataset

train_set=MNIST(download_root,train=True,transform=transform,download=True)
test_set=MNIST(download_root,train=False,transform=transform)

# Dataloader

train_loader=DataLoader(train_set,batch_size,shuffle=True)
test_loader=DataLoader(test_set,batch_size,shuffle=True)

# Model

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator=nn.Sequential(
            nn.Linear(100,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,1024),
            nn.Tanh(),
            nn.Linear(1024,784),
        )
    def forward(self,x):
        x=self.generator(x)
        x=torch.flatten(x,1)``
        return x
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator=nn.Sequential(
            nn.Linear(784,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(inplace=True),
            nn.Linear(1024,512),
            nn.ReLU(inplace=True),
            nn.Dropout(inplace=True),
            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Dropout(inplace=True),
            nn.Linear(256,1),
            nn.Sigmoid()   
        )
    def forward(self,x):
        x=torch.flatten(x,1)
        x=self.discriminator(x)
        return x

Gen=Generator()
Gen.to(device)
Discrim=Discriminator()
Discrim.to(device)
# Loss & Optim
criterion=nn.BCELoss()

G_optimizer=optim.Adam(Gen.parameters(),lr=leraing_rate, betas=(0.5, 0.999))
D_optimizer=optim.Adam(Discrim.parameters(),lr=leraing_rate, betas=(0.5, 0.999))

# Train

def G_train(epoch):
    for batch_idx,(data,_) in enumerate(train_loader):
        real_correct=Variable(torch.ones(batch_size,1))
        z=Variable(torch.randn(batch_size,100))
        data=Variable(data).to(device)
        ouput=Gen(z)
        G_optimizer.zero_grad()
        G_loss=criterion(ouput,real_correct)
        G_loss.backward()
        G_optimizer.step()
        if batch_idx % 10 ==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), G_loss.item()))


def D_train(epoch):
    for batch_idx, (data,_) in enumerate(train_loader):
        fake_correct=Variable(torch.zeros(batch_size,1))
        real_correct=Variable(torch.ones(batch_size,1))
        data=Variable(data).to(device)
        # 진짜 이미지를 진짜로 판별할 수 있게 학습
        real_output=Discrim(data)
        D_real_loss=criterion(real_output,real_correct)

        # 가짜 이미지를 가짜로 판별할 수 있게 학습
        z=Variable(torch.randn(batch_size,100))
        fake_image=Gen(z)
        fake_output=Discrim(fake_image)
        D_optimizer.zero_grad()
        D_fake_loss=criterion(fake_output,fake_correct)
        D_loss=D_real_loss+D_fake_loss
        D_loss.backward()
        D_optimizer.step()
        if batch_idx % 10 ==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), D_loss.item()))

# Test


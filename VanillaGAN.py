import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch import nn,cuda,optim
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid
from os import path
from google.colab import drive

notebooks_dir_name = 'notebooks'
drive.mount('/content/gdrive')
notebooks_base_dir = path.join('./gdrive/My Drive/', notebooks_dir_name)
if not path.exists(notebooks_base_dir):
  print('Check your google drive directory. See you file explorer')
# Settings
download_root='mnist'
stored_path='images'
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,),std=(0.5,))
])
device= 'cuda' if cuda.is_available() else 'cpu'


leraing_rate=0.0002
# Dataset
train_set=MNIST(download_root,train=True,transform=transform,download=True)

# Dataloader
train_loader=DataLoader(train_set,batch_size=60,shuffle=True)

# Image_dir
import os
import imageio

if not os.path.isdir(stored_path):
    os.makedirs(stored_path,exist_ok=True)

# Model
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        def gen_block(in_features,out_features):
            layers=[nn.Linear(in_features,out_features)]
            layers.append(nn.ReLU())
            return layers
        self.generator=nn.Sequential(
            *gen_block(100,128),
            *gen_block(128,256),
            *gen_block(256,512),
            *gen_block(512,1024),
            nn.Linear(1024,784),
            nn.Tanh()
        )
    def forward(self,x):
        x=self.generator(x)
        x=x.view(x.size(0),-1)
        return x
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def disc_block(in_features,out_features):
            layers=[nn.Linear(in_features,out_features)]
            layers.append(nn.ReLU())
            layers.append(nn.Dropout())
            return layers
        self.discriminator=nn.Sequential(
            *disc_block(784,1024),
            *disc_block(1024,512),
            *disc_block(512,256),
            nn.Linear(256,1),
            nn.Sigmoid()   
        )
    def forward(self,x):
        x=x.view(x.size(0),-1)
        x=self.discriminator(x)
        return x

Gen=Generator().to(device)
Discrim=Discriminator().to(device)

# Loss & Optim
criterion=nn.BCELoss()

G_optimizer=optim.Adam(Gen.parameters(),lr=leraing_rate, betas=(0.5, 0.999))
D_optimizer=optim.Adam(Discrim.parameters(),lr=leraing_rate, betas=(0.5, 0.999))

# Train
def train(epoch):
    for batch_idx,(data,_) in enumerate(train_loader):
        batch_size=data.size(0)
        fake_correct=Variable(torch.zeros(batch_size,1)).to(device)
        real_correct=Variable(torch.ones(batch_size,1)).to(device)
        z=torch.randn(batch_size, 100,device=device)
        data=Variable(data).to(device)

        # Gen 학습
        gen_img=Gen(z)
        G_optimizer.zero_grad()
        G_loss=criterion(Discrim(gen_img),real_correct)
        G_loss.backward()
        G_optimizer.step()
        # Discrim 학습
        # 진짜 이미지를 진짜로 판별할 수 있게 학습
        real_output=Discrim(data)
        D_real_loss=criterion(real_output,real_correct)

        # 가짜 이미지를 가짜로 판별할 수 있게 학습
        gen_img.detach().to(device) # Gen은 이미 학습해서 다시 학습 안 시키게 detach()
        fake_output=Discrim(gen_img)
        D_optimizer.zero_grad()
        D_fake_loss=criterion(fake_output,fake_correct)
        D_loss=(D_real_loss+D_fake_loss)/2
        D_loss.backward()
        D_optimizer.step()

        batch_finish=epoch * len(train_loader) + batch_idx
        if (batch_finish) % 400 == 0:
            print("[Epoch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, 200, D_loss.item(), G_loss.item())
            )
    if (epoch+1) % 10 == 0:
        gen_img = gen_img.reshape([batch_size, 1, 28, 28])
        img_grid = make_grid(gen_img, nrow=10, normalize=True)
        save_image(img_grid, "images/result_%d.png"%(epoch+1)) 
if __name__ == "__main__":
    for epoch in range(200):
        train(epoch)
    images=[]
    for file_name in os.listdir(stored_path):
        images.append(imageio.imread(file_name))
        imageio.mimsave('result.gif',images)
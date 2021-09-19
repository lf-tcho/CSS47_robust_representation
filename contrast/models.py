
import torch
import torch.nn as nn
import torch.nn.functional as F

class basic_block(torch.nn.Module):
    expansion=1
    def __init__(self,in_planes,planes,stride=1):
        super(basic_block,self).__init__()
        self.conv1=torch.nn.Conv2d(in_channels=in_planes,out_channels=planes,stride=stride,kernel_size=3,padding=1,bias=False)
        self.bn1=torch.nn.BatchNorm2d(planes)
        self.conv2=torch.nn.Conv2d(in_channels=planes,out_channels=planes,stride=1,kernel_size=3,padding=1,bias=False)
        self.bn2=torch.nn.BatchNorm2d(planes)

        self.shortcut=torch.nn.Sequential()
        if stride!=1 or in_planes!=self.expansion*planes:
            self.shortcut=torch.nn.Sequential(torch.nn.Conv2d(in_planes,self.expansion*planes,kernel_size=1,stride=stride,bias=False),
            torch.nn.BatchNorm2d(self.expansion*planes))
    
    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out+=self.shortcut(x)
        out=F.relu(out)
        return (out)

class preactblock(torch.nn.Module):
    expansion=1
    def __init__(self,in_planes,planes,stride=1):
        super(preactblock,self).__init__()
        self.bn1=torch.nn.BatchNorm2d(in_planes)
        self.conv1=torch.nn.Conv2d(in_planes,planes,stride=stride,kernel_size=3,padding=1,bias=False)
        self.bn2=torch.nn.BatchNorm2d(planes)
        self.conv2=torch.nn.Conv2d(planes,planes,stride=1,kernel_size=3,padding=1,bias=False)

        self.shortcut=torch.nn.Sequential()
        if stride!=1 or in_planes!=self.expansion*planes:
            self.shortcut=torch.nn.Sequential(torch.nn.Conv2d(in_planes,self.expansion*planes,kernel_size=1,stride=stride,bias=False))
    
    def forward(self,x):
        out=F.relu(self.bn1(x))
        shortcut=self.shortcut(out)
        out=self.conv1(out)
        out=self.conv2(F.relu(self.bn2(out)))
        out+=shortcut
        return (out)

class bottleneck(torch.nn.Module):
    expansion=4
    def __init__(self,in_planes,planes,stride=1):
        super(bottleneck,self).__init__()
        self.conv1=torch.nn.Conv2d(in_planes,planes,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn1=torch.nn.BatchNorm2d(planes)
        self.conv2=torch.nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=torch.nn.BatchNorm2d(planes)
        self.conv3=torch.nn.Conv2d(planes,self.expansion*planes,kernel_size=1,bias=False)
        self.bn3=torch.nn.BatchNorm2d(self.expansion*planes)

        self.shortcut=torch.nn.Sequential()
        if stride!=1 or in_planes!=self.expansion*planes:
            self.shortcut=torch.nn.Sequential(torch.nn.Conv2d(in_planes,self.expansion*planes,kernel_size=1,stride=stride,bias=False),
            torch.nn.BatchNorm2d(self.expansion*planes))
    
    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=F.relu(self.bn2(self.conv2(out)))
        out=self.bn3(self.conv3(out))
        out+=self.shortcut(x)
        out=F.relu(out)
        return (out)

class preact_bottelneck(torch.nn.Module):
    expansion=4
    def __init__(self,in_planes,planes,stride=1):
        super(preact_bottleneck,self).__init__()
        self.bn1=torch.nn.BatchNorm2d(in_planes)
        self.conv1=torch.nn.Conv2d(in_planes,planes,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn2=torch.nn.BatchNorm2d(planes)
        self.conv2=torch.nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn3=torch.nn.BatchNorm2d(planes)
        self.conv3=torch.nn.Conv2d(planes,self.expansion*planes,kernel_size=1,stride=1,padding=0,bias=False)

        self.shortcut=torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut=torch.nn.Sequential(torch.nn.Conv2d(in_planes,self.expansion*planes,kernel_size=1,stride=stride,bias=False))
    
    def forward(self,x):
        out=F.relu(self.bn1(x))
        shortcut=self.shortcut(out)
        out=self.conv1(out)
        out=self.conv2(F.relu(self.bn2(out)))
        out=self.conv3(F.relu(self.bn3(out)))
        out+=shortcut
        return (out)

class resnet(torch.nn.Module):
    def __init__(self,block,n_blocks,n_classes=10,contrast=True):
        super(resnet,self).__init__()
        self.in_planes=64
        self.conv1=torch.nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1=torch.nn.BatchNorm2d(64)
        self.layer1=self._make_layer(block,64,n_blocks[0],stride=1)
        self.layer2=self._make_layer(block,128,n_blocks[1],stride=2)
        self.layer3=self._make_layer(block,256,n_blocks[2],stride=2)
        self.layer4=self._make_layer(block,512,n_blocks[3],stride=2)
        self.contrast=contrast
        if self.contrast!=True:
            self.linear=torch.nn.Linear(512*self.expansion,n_classes)
    
    def _make_layer(self,block,planes,n_blocks,stride):
        strides=[stride] + [1]*(n_blocks-1)
        layers=[]
        for stride in strides:
            layers.append(block(self.in_planes,planes,stride))
            self.in_planes=planes * block.expansion
        final_layer=torch.nn.Sequential(*layers)
        return (final_layer)
    
    def forward(self,x,internal=False):
        out=x
        out_list=[]

        out=self.conv1(out)
        out=self.bn1(out)
        out=F.relu(out)
        out_list.append(out)

        out=self.layer1(out)
        out_list.append(out)

        # print(out.shape)
        out=self.layer2(out)
        # print(out.shape)
        out_list.append(out)

        out=self.layer3(out)
        out_list.append(out)

        out=self.layer4(out)
        out_list.append(out)

        out=F.avg_pool2d(out,4)
        out=out.view(out.shape[0],-1)
        
        if self.contrast!=True:
            out=self.linear(out)
        
        if internal:
            return (out,out_list)
        else:
            return (out)

#same as SimCLR projector 2 layer MLP (x==>z)
class projection(torch.nn.Module):
    def __init__(self,expansion=0):
        super(projection,self).__init__()
        self.linear1=torch.nn.Linear(in_features=512*expansion,out_features=2048)
        self.linear2=torch.nn.Linear(in_features=2048,out_features=128)
    
    def forward(self,x):
        x=self.linear1(x)
        x=F.relu(x)
        x=self.linear2(x)

        return (x)

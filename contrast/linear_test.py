import os
from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn

import torch.optim as optim
import dataloader
from adversarial_attack import rep_adv,fsgm

from models import projection,resnet,basic_block

from utils import progress_bar, checkpoint, AverageMeter, accuracy

from loss import pairwise_similarity,contrastive_loss
from torchlars import LARS
from warmup_scheduler import GradualWarmupScheduler

torch.manual_seed(2342)


lr=0.2
torch.backends.cudnn.benchmark=True
torch.cuda.set_device(0)
device=torch.device("cuda")
model_load_path="/vinai/sskar/CSS47_robust_representation/contrast/checkpoint/ckpt.t7sample_42"
n_epochs=200

dataset_type="cifar-10"
if dataset_type == "cifar-10":
    n_classes=10
else:
    n_classes=100

train_type="linear_eval"
if train_type == "contrastive" or train_type=="linear_eval":
    contrastive_learning=True
else:
    contrastive_learning=False

dataset_type="cifar-10"
train_type="linear_eval"
bs=128
color_jitter_strength=0.5
trainloader,traindst,testloader,testdst=dataloader.get_dataset(dataset_type,train_type,bs,color_jitter_strength)

expansion=1
def load(epoch):
    model=resnet(basic_block,[2,2,2,2],n_classes=n_classes,contrast=contrastive_learning)
    
    projector=projection(expansion=expansion)
    checkpoint_=torch.load(model_load_path)
    new_state_dict=OrderedDict()
    for k,v in checkpoint_['model'].items():
        name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    Linear=torch.nn.Sequential(torch.nn.Linear(512*expansion,10))
    
    model_params=[]
    model_params+=Linear.parameters()
    loptim = torch.optim.SGD(model_params,lr=0.2,momentum=0.9,weight_decay=5e-4)
    model.to(device)
    Linear.to(device)
    return (model,Linear,'None',loptim,'None')#model, Linear, projector, loptim, attacker

criterion=torch.nn.CrossEntropyLoss()

def linear_train(epoch,model,linear,projector,loptim,attacker=None):
    linear.train()
    model.eval()
    total_loss=0
    correct=0
    total=0


    for batch_idx,(ori,t1_x,t2_x,target) in enumerate(trainloader):
        ori,t1_x,t2_x,target=ori.to(device),t1_x.to(device),t2_x.to(device),target.to(device)
        inputs=ori
        #clean
        total_inputs=inputs
        total_targets=target
        
        feat=model(total_inputs)
        output=linear(feat)

        _,predx=torch.max(output.data,1)
        loss=criterion(output,total_targets)

        correct+=predx.eq(total_targets.data).cpu().sum().item()
        total+=total_targets.size(0)
        acc=100.*correct/total

        total_loss+=loss.data
        loptim.zero_grad()
        loss.backward()
        loptim.step()

        progress_bar(batch_idx, len(trainloader),
                    'Loss: {:.4f} | Acc: {:.2f}'.format(total_loss/(batch_idx+1), acc))
    print ("Epoch: {}, train accuracy: {}".format(epoch, acc))

    return (acc,model,linear,projector,loptim)

def test(model,linear):
    global best_acc

    model.eval()
    linear.eval()

    test_loss=0
    correct=0
    total=0

    for idx,(image, label) in enumerate(testloader):
        img=image.to(device)
        y=label.to(device)
        out=linear(model(img))

        _, predx=torch.max(out.data, 1)
        loss=criterion(out, y)

        correct+=predx.eq(y.data).cpu().sum().item()
        total+=y.size(0)
        acc=100.*correct/total

        test_loss += loss.data
        progress_bar(idx, len(testloader),'Testing Loss {:.3f}, acc {:.3f}'.format(test_loss/(idx+1),acc))
        
    print ("Test accuracy: {0}".format(acc))
    return (acc,model,linear)

def adjust_lr(epoch, optim):
    lr = 0.2
    if dataset_type=='cifar-10' or dataset_type=='cifar-100':
        lr_list = [30,50,100]
    if epoch>=lr_list[0]:
        lr = lr/10
    if epoch>=lr_list[1]:
        lr = lr/10
    if epoch>=lr_list[2]:
        lr = lr/10
    
    for param_group in optim.param_groups:
        param_group['lr']=lr

model,linear,projector,loptim,attacker=load(0)

for epoch in range(n_epochs):
    print('Training linear evaluation epoch:',epoch)

    train_acc,model,linear,projector,loptim=linear_train(epoch,model=model,linear=linear,projector=projector,loptim=loptim,attacker=attacker)
    test_acc,model,linear=test(model,linear)
    adjust_lr(epoch, loptim)

    if epoch % 10 == 0:
        checkpoint(model,test_acc,epoch,loptim)
        checkpoint(linear,test_acc,epoch,loptim, save_name_add='_linear')


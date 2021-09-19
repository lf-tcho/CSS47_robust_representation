import os
import torch
import torch.backends.cudnn as cudnn

import torch.optim as optim
import dataloader
from adversarial_attack import rep_adv

from models import projection,resnet,basic_block
from utils import progress_bar, checkpoint, AverageMeter, accuracy

from loss import pairwise_similarity,contrastive_loss
from torchlars import LARS
from warmup_scheduler import GradualWarmupScheduler



torch.backends.cudnn.benchmark=True
torch.cuda.set_device(0)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epochs=200
dataset_type="cifar-10"
train_type="contrastive"
bs=256
color_jitter_strength=0.5
train_loader,train_dataset,test_loader,test_dataset,train_sampler=dataloader.get_dataset(dataset_type,train_type,bs,color_jitter_strength)
# for bi,(image,t1_x,t2_x,label) in train_loader:
#     print(image.shape)


if dataset_type == "cifar-10":
    n_classes=10
else:
    n_classes=100

if train_type == "contrastive":
    contrastive_learning=True
else:
    contrastive_learning=False


model=resnet(basic_block,[2,2,2,2],n_classes=n_classes,contrast=contrastive_learning)
expansion=1
projector=projection(expansion=expansion)

model=model.to(device)
projector=projector.to(device)

epsilon=0.0314
alpha=0.007
min_val=0.0
max_val=1.0
max_iters=7
type_="linf"
loss_type="sim"
regularize="other"
rep=rep_adv(model,projector,epsilon,alpha,min_val,max_val,max_iters,device,type_,loss_type,regularize)


model_params=[]
model_params+=model.parameters()
model_params+=projector.parameters()

base_optimizer=torch.optim.SGD(model_params,lr=0.1,momentum=0.9, weight_decay=1e-6)
optimizer=LARS(optimizer=base_optimizer,eps=1e-8, trust_coef=0.001)

scheduler_cosine=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,n_epochs)
scheduler_warmup=GradualWarmupScheduler(optimizer,multiplier=15.0,total_epoch=10,after_scheduler=scheduler_cosine)



def train(epoch):
    print("training for epoch : ", epoch)
    model.train()
    projector.train()

    train_sampler.set_epoch(epoch)
    scheduler_warmup.step()

    total_loss=0
    reg_simloss=0
    reg_loss=0

    for batch_idx,(image,t1_x,t2_x,label) in enumerate(train_loader):
        image,t1_x,t2_x,label=image.to(device),t1_x.to(device),t2_x.to(device),label.to(device)
        attack_target=t2_x

        advinputs,adv_loss=rep.perturb_and_loss(image=t1_x,target=attack_target,optimizer=optimizer,weight=256,random_start=True)
        reg_loss+=adv_loss.data

        inputs=torch.cat((t1_x,t2_x,advinputs))
        outputs=model(inputs)
        outputs=projector(outputs)

        similarity,gathered_ops=pairwise_similarity(outputs,tau=0.5)
        simloss=contrastive_loss(similarity,device)
        loss=simloss+adv_loss
        
        optimizer.zero_grad()
        loss.backward()
        total_loss+=loss.data
        reg_simloss+=simloss.data

        optimizer.step()
        progress_bar(batch_idx, len(train_loader),
        'Loss: %.3f | SimLoss: %.3f | Adv: %.2f'
        % (total_loss/(batch_idx+1), reg_simloss/(batch_idx+1), reg_loss/(batch_idx+1)))

    return (total_loss/batch_idx,reg_simloss/batch_idx)

def test(epoch,train_loss):
    model=model.eval()
    projector=projector.eval()

    # Save at the last epoch #       
    if epoch == args.epoch - 1 :
        checkpoint(model, train_loss, epoch, args, optimizer)
        checkpoint(projector, train_loss, epoch, args, optimizer, save_name_add='_projector')
       
    # Save at every 100 epoch #
    elif epoch % 100 == 0:
        checkpoint(model, train_loss, epoch, args, optimizer, save_name_add='_epoch_'+str(epoch))
        checkpoint(projector, train_loss, epoch, args, optimizer, save_name_add=('_projector_epoch_' + str(epoch)))


for epoch in range(0,n_epochs):
    train_loss,reg_loss=train(epoch)
test(epoch,train_loss)
        




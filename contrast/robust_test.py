import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import dataloader   

from utils import progress_bar
from collections import OrderedDict

from adversarial_attack import fsgm
from models import resnet,basic_block

best_acc=0  
start_epoch=0
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


model=resnet(basic_block,[2,2,2,2],n_classes=n_classes,contrast=contrastive_learning)
expansion=1
linear=torch.nn.Sequential(torch.nn.Linear(512*expansion,10))

checkpoint_=torch.load(model_load_path)
new_state_dict=OrderedDict()
for k,v in checkpoint_['model'].items():
    name = k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)

linearcheckpoint_ = torch.load(model_load_path+'_linear')
new_state_dict = OrderedDict()
for k, v in linearcheckpoint_['model'].items():
    name = k
    new_state_dict[name] = v
linear.load_state_dict(new_state_dict)

criterion=torch.nn.CrossEntropyLoss()

model.cuda()
linear.cuda()
torch.backends.cudnn.benchmark = True

attacker=fsgm(model,linear,epsilon=0.0314,alpha=0.007,min_val=0.0,max_val=1.0,max_iters=10,device=device,type_="linf")

def test(attacker):
    global best_acc

    model.eval()
    linear.eval()

    test_clean_loss=0
    test_adv_loss=0
    clean_correct=0
    adv_correct=0
    clean_acc=0
    total=0

    for idx,(image,label) in enumerate(testloader):
        img,y=image.to(device),label.to(device)
        total+=y.size(0)
        out=linear(model(img))
        _,predx=torch.max(out.data,1)
        clean_loss=criterion(out,y)
        clean_correct+=predx.eq(y.data).cpu().sum().item()
        clean_acc=100.*clean_correct/total
        test_clean_loss+=clean_loss.data
        adv_inputs=attacker.perturb(image=img,labels=y,loss="mean",random_start=True)

        out=linear(model(adv_inputs))
        _,predx=torch.max(out.data, 1)
        adv_loss=criterion(out, y)
        adv_correct+=predx.eq(y.data).cpu().sum().item()
        adv_acc=100.*adv_correct/total
        test_adv_loss+=adv_loss.data

        progress_bar(idx,len(testloader),'Testing Loss {:.3f}, acc {:.3f} , adv Loss {:.3f},adv acc {:.3f}'.format(test_clean_loss/(idx+1), clean_acc, test_adv_loss/(idx+1), adv_acc))
    
    print ("Test accuracy: {0}/{1}".format(clean_acc, adv_acc))
    return (clean_acc,adv_acc)

test_acc,adv_acc=test(attacker)
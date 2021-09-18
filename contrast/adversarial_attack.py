import torch
import torch.nn.functional as F
from loss import pairwise_similarity, contrastive_loss

def project(x,original,epsilon,type="linf"):
    max_x=original+epsilon
    min_x=original-epsilon
    x=torch.max(torch.min(x,max_x),min_x)
    return (x)

class rep_adv(object):
    def __init__(self,model,projector,epsilon,alpha,min_val,max_val,max_iters,device,type_="linf",loss_type="sim",regularize="original"):
        self.model=model
        self.projector=projector
        self.epsilon=epsilon
        self.alpha=alpha
        self.min_val=min_val
        self.max_val=max_val
        self.max_iters=max_iters
        self.device=device
        self.type=type_
        self.loss_type=loss_type
        self.regularize=regularize

    def perturb_and_loss(self,image,target,optimizer,weight,random_start=True):
        if random_start == True:
            delta=torch.FloatTensor(image.shape).uniform_(-self.epsilon,self.epsilon).float().to(self.device)
            x=image.float().clone()+delta
            x=torch.clamp(x,self.min_val,self.max_val)
        else:
            x=image.clone()
        x.requires_grad=True
        self.model.eval()
        self.projector.eval()
        bs=x.shape[0]

        with torch.enable_grad():
            for i in range(self.max_iters):
                self.model.zero_grad()
                self.projector.zero_grad()

                inputs=torch.cat((x,target))
                output=self.model(inputs)
                output=self.projector(output)
                similarity_mat,_=pairwise_similarity(output,tau=0.5)
                loss=contrastive_loss(similarity_mat,self.device)

                grads=torch.autograd.grad(loss,x,grad_outputs=None,only_inputs=True,retain_graph=False)[0]
                grad_signs=torch.sign(grads.data)

                x.data+=self.alpha*(grad_signs)# equation 5 from paper t(x)^{i+1}
                x=torch.clamp(x,self.max_val,self.min_val)
                x=project(x,image,self.epsilon,self.type)
        
        self.model.train()
        self.projector.train()
        optimizer.zero_grad()

        if self.regularize == "original":
            inputs=torch.cat((x,original_image))
        else:
            inputs=torch.cat((x,target))
        output=self.model(inputs)
        output=self.projector(output)
        similarity_mat,_=pairwise_similarity(output,tau=0.5)
        loss=contrastive_loss(similarity_mat,self.device)
        loss/=weight

        return (x.detach(),loss)                           
import torch
import torch.nn.functional as F
#2 parts to the relic loss function: (see equtaion 3 in paper)
#part 1 : standard contrastive loss but with summing over all possible augmentations(assume h,f are the same encoder network) 
#part 2 : kl divergence betweem 2 neural nets for proxy tasks (instance discrimination) i.e inavriance across style augmentations

##part1
def pairwise_similarity(output,tau):
    """the outputs in this case would be aggregated from  2 different augmented inputs"""
    bs=output.shape[0]
    outputs_norm=outputs/(outputs.norm(dim=1).view(bs,1)+1e-8)
    similarity_matrix=(1./tau)*torch.mm(output_norm,outputs_norm.transpose(0,1).detach())
    return (similarity_matrix,outputs)

def contrastive_loss(similarity_matrix,device,n_augs=2):
    N2=len(similarity_matrix)
    N=int(len(similarity_matrix)/n_augs)
    similarity_matrix_exp=torch.exp(similarity_matrix)
    mask=~torch.eye(N2,N2).bool().to(device)
    similarity_matrix_exp=similarity_matrix_exp*(1-torch.eye(N2,N2)).to(device)

    contrastive_loss=-torch.log(similarity_matrix_exp/(torch.sum(similarity_matrix_exp,dim=1).view(N2,1)+1e-8)+1e-8)

    loss_total=torch.sum(torch.diag(contrast_loss[0:N,N:2*N]) + torch.diag(contrast_loss[N:2*N,0:N]) 
    + torch.diag(contrast_loss[0:N,2*N:]) + torch.diag(contrast_loss[2*N:,0:N])
    + torch.diag(contrast_loss[N:2*N,2*N:]) + torch.diag(contrast_loss[2*N:,N:2*N]))

    loss_total*=(1./float(N2))

    return(loss_total)


##part2
def augmentation_invariance_loss(output,n_classes):
    bs=output.shape[0]
    N=int(bs/2)
    op1=output[0:N,:]
    op2=output[N:,:]

    sftx_dist1=F.softmax(op1)
    sftx_dist2=F.softmax(op2)

    #loss=(sftx_dist1.log(),(sftx_dist1/sftx_dist2).log()).sum()
    #first one is faster for small batch sizes 
    loss=F.kl_div(sftx_dist2.log(),sftx_dist1,None,None,'sum')
    return(loss)
    

def relic_lossfunc(output,tau,n_classes,device,alpha):
    sim_matrix,_=pairwise_similarity(output)
    contrastive_loss=contrastive_loss(sim_matrix,device,n_augs=2)

    aug_loss=augmentation_invariance_loss(output,n_classes)

    total_loss=contrastive_loss + alpha * (aug_loss)
    return (total_loss)
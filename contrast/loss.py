import diffdist.functional as distops 
import torch
import torch.distributed as dist

def pairwise_similarity(outputs,tau):
    #the pairwise similarity should actually be 
    bs=outputs.shape[0]
    outputs_norm=outputs/(outputs.norm(dim=1).view(bs,1)+1e-8)
    similarity_matrix=(1./tau)*torch.mm(outputs_norm,outputs_norm.transpose(0,1).detach())#(bs,bs)
    return (similarity_matrix,outputs)

def contrastive_loss(similarity_matrix,device):
    N2=len(similarity_matrix)#bs
    #subtract the diagonal from the similarity matrix for computational efficiency to remove self similarity
    N=int(len(similarity_matrix)/3)#t(x); t_prime(x); t_adv(x)
    #removing the diagonal by masking the diagonal
    similarity_matrix_exp=torch.exp(similarity_matrix)
    mask=~torch.eye(N2,N2).bool().to(device)
    similarity_matrix_exp=similarity_matrix_exp*(1-torch.eye(N2,N2)).to(device)

    contrast_loss=-torch.log(similarity_matrix_exp/(torch.sum(similarity_matrix_exp,dim=1).view(N2,1) + 1e-8) + 1e-8)

    # contrast_loss=(similarity_matrix_exp/(torch.sum(similarity_matrix_exp,dim=1).view(N2,1)+1e-5)+1e-5)
    # contrast_loss=-torch.log(contrast_loss)
    #for calculating the total loss adding the similarity prameters by indexing for the pairs eg:(x1x2+x2x1)+(x1x3+x3x1)+()
    # after removing the self similarity
    """[0,x1x2,x1x3
        x2x1,0,x2x3
        x3x1,x3x2,0]
    """ 
    loss_total=torch.sum(torch.diag(contrast_loss[0:N,N:2*N]) + torch.diag(contrast_loss[N:2*N,0:N]) 
    + torch.diag(contrast_loss[0:N,2*N:]) + torch.diag(contrast_loss[2*N:,0:N])
    + torch.diag(contrast_loss[N:2*N,2*N:]) + torch.diag(contrast_loss[2*N:,N:2*N]))

    loss_total*=(1./float(N2))

    return (loss_total)



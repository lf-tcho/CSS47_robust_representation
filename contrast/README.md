Pytorch Implementation of ROCL . Based on the paper:

  > [Adversarial Self-Supervised Contrastive Learning](https://arxiv.org/abs/2006.07589)\
  > Minseon Kim, Jihoon Tack, Sung Ju Hwang\
  > arXiv:2006.07589


### To-DO
- [x] Write the ROCL implementation
- [x] Write the RELIC loss implementation
- [ ] Train and check if  ROCL code eproduces the result: In progress 
- [ ] Check RELIC loss with different architectures 
- [x] Commit RELIC loss file

###  Help Needed
- [] Write a training script for training relic loss

### Classification and robustness on CIFAR 10

| Model name         |    Accuracy     |   robustness   |
| ------------------ |---------------- | -------------- |
| Official           |    83.71 %      |     40.27%     |
| Ours               |    71.91 %      |     20.21%     |

I only trained for 200 epochs instead of 1000 in the paper.
Please run python train.py and resume training from the 200th epoch. 

### Paper Notes
- [ROCL](https://dramatic-durian-120.notion.site/ICLR-CSS-Robust-Self-supervised-Learning-8e0853e04da74efdb3de27735184d932)
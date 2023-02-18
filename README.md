# CLOM

NeurIPS 2022 paper: Margin-Based Few-Shot Class-Incremental Learning with Class-Level Overfitting Mitigation

## Abstract
Few-shot class-incremental learning (FSCIL) is designed to incrementally recognize novel classes with only few training samples after the (pre-)training on base classes with sufficient samples, which focuses on both base-class performance and novel-class generalization. A well known modification to the base-class training is to apply a margin to the base-class classification. However, a dilemma exists that we can hardly achieve both good base-class performance and novel-class generalization simultaneously by applying the margin during the base-class training, which is still under explored. In this paper, we study the cause of such dilemma for FSCIL. We first interpret this dilemma as a class-level overfitting (CO) problem from the aspect of pattern learning, and then find its cause lies in the easily-satisfied constraint of learning margin-based patterns. Based on the analysis, we propose a novel margin-based FSCIL method to mitigate the CO problem by providing the pattern learning process with extra constraint from the margin-based patterns themselves. Extensive experiments on CIFAR100, Caltech-USCD Birds-200-2011 (CUB200), and miniImageNet demonstrate that the proposed method effectively mitigates the CO problem and achieves state-of-the-art performance.

## Requirements
- [PyTorch >= version 1.1](https://pytorch.org)
- tqdm

We follow [CEC](https://github.com/icoz69/CEC-CVPR2021) to use the same data index list for training. Please first follow CEC to prepare the data under the `data/` folder.

## Training scripts

CIFAR

    $ python train.py -project base -dataset cifar100  -base_mode ft_cos -new_mode avg_cos -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 100 -schedule Milestone -milestones 60 70 -gpu 0 -temperature 16 -num_workers 4 -in_domain_feat_cls_weight 1.0 -in_domain_feat_dim 256 -average_cosMargin -0.2 -cosMargin -0.5 -in_domain_average_cosMargin 0.1 -in_domain_feat_cosMargin 0.2 -class_relation wg -in_domain_class_relation wg -gpu 0

CUB200
	
    $ python train.py -project base -dataset cub200 -base_mode ft_cos -new_mode avg_cos -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 80 -schedule Milestone -milestones 40 50 60 -gpu 0 -temperature 16 -num_workers 4 -in_domain_feat_cls_weight 0.01 -in_domain_feat_dim 8192 -average_cosMargin -0.2 -cosMargin -0.25 -class_relation wg -in_domain_average_cosMargin 0.3 -in_domain_feat_cosMargin 0.6 -in_domain_class_relation wg -gpu 0

miniImageNet:

    $ python train.py -project base -dataset mini_imagenet -base_mode ft_cos -new_mode avg_cos -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 180 -schedule Milestone -milestones 130 140 -temperature 16 -num_workers 4 -in_domain_feat_cls_weight 1.0 -in_domain_feat_dim 4096 -average_cosMargin -0.2 -cosMargin -0.5 -class_relation wg -in_domain_average_cosMargin 0.1 -in_domain_feat_cosMargin 0.2 -in_domain_class_relation wg -gpu 0

## Acknowledgment

- [CEC](https://github.com/icoz69/CEC-CVPR2021)

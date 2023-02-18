from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy

from .helper import *
from utils import *
from dataloader.data_utils import *


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        out_log_path = self.set_save_path()
        self.args = set_up_datasets(self.args)
        self.out_log = open(out_log_path, 'a')

        self.model = MYNET(self.args, mode=self.args.base_mode)
        #self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def get_optimizer_base(self):
        if self.args.dataset == 'cub200':
            encoder_params = []; fc_params = []
            for name, param in self.model.named_parameters():
                if 'encoder' in name:
                    encoder_params.append(param)
                else:
                    fc_params.append(param)

            optimizer = torch.optim.SGD([{'params': encoder_params, 'lr': self.args.lr_base * 0.1},
                                         {'params': fc_params}], 
                                        self.args.lr_base, momentum=0.9, nesterov=True,
                                        weight_decay=self.args.decay)
            log(self.out_log, 'num encoder params = %d, num fc params = %d'%(len(encoder_params), len(fc_params)))
            log(self.out_log, 'scale lr of encoder to 0.1*lr_base')

        else:
            optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                        weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base, verbose=True)

        return optimizer, scheduler

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        log(self.out_log, str(args))

        base_train_set, base_trainloader, base_testloader = self.get_dataloader(0)

        self.model.load_state_dict(self.best_model_dict)

        optimizer, scheduler = self.get_optimizer_base()

        max_acc = 0.0

        for epoch in range(args.epochs_base):
            start_time = time.time()
            # train base sess
            self.model.mode = self.args.base_mode
            tl, ta = base_train(self.model, base_trainloader, optimizer, scheduler, epoch, args, is_base=True)
            
            # test base sess
            if args.in_domain_feat_cls_weight == 0.0:
                tsl, tsa = test(self.model, base_testloader, epoch, args, 0, is_base=True)
            else:
                tsl, tsa, ind_va, cmb_va = test(self.model, base_testloader, epoch, args, 0, is_base=True)
            
            ##### test model with all sessions #####
            self.model = replace_base_fc(base_train_set, base_testloader.dataset.transform, self.model, args)
                       
            self.model.mode = self.args.new_mode
            for k in range(1, args.sessions):
                session_train_set, session_trainloader, session_testloader = self.get_dataloader(k)
                self.model.update_fc(session_trainloader, np.unique(session_train_set.targets), k)
            
            # here session_testloader is the last session's testloader
            return_list = test_all_sessions(self.model, session_testloader, 0, args, args.sessions)

            if args.in_domain_feat_cls_weight == 0.0:
                vls, vas = return_list
                key_acc = vas[-2]
            else:
                vls, vas, ind_vas, cmb_vas = return_list
                key_acc = cmb_vas[-2]
            
            log_str = 'epoch: %d'%epoch
            log_str += ', base acc: {:.4f}'.format(tsa)
            if args.in_domain_feat_cls_weight != 0.0:
                log_str += '| ind: {:.4f}| cmb: {:.4f}'.format(ind_va, cmb_va)

            log_str += ', novel acc: {:.4f}'.format(vas[-1])
            if args.in_domain_feat_cls_weight != 0.0:
                log_str += '| ind: {:.4f}| cmb: {:.4f}'.format(ind_vas[-1], cmb_vas[-1])
            
            log_str += ', sess acc: %s'%str(np.around(vas[:-1], 4))
            if args.in_domain_feat_cls_weight != 0.0:
                log_str += '| ind: %s| cmb: %s'%(str(np.around(ind_vas[:-1], 4)), str(np.around(cmb_vas[:-1], 4)))

            if key_acc > max_acc:
                max_acc = key_acc
                self.best_model_dict = deepcopy(self.model.state_dict())
                save_model_dir = os.path.join(args.save_path, 'max_acc.pth')
                torch.save(dict(params=self.model.state_dict()), save_model_dir)

            log_str += ', max acc: {:.4f}'.format(max_acc)

            log(self.out_log, log_str)
            #log(self.out_log, str([epoch, np.around(tsa, 3), np.around(vas[-1], 3), np.around(vas[:-1], 3)]))
            
            scheduler.step()

            print('This epoch takes %d seconds' % (time.time() - start_time), 'still need around %.2f mins' % ((time.time() - start_time) * (args.epochs_base - epoch) / 60))
            

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Total time used %.2f mins' % total_time)

    def set_save_path(self):
        mode = self.args.base_mode + '-' + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Cosine':
            self.args.save_path = self.args.save_path + 'Cosine-Epo_%d-Lr_%.4f' % (
                self.args.epochs_base, self.args.lr_base)

        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)

        if self.args.tag != '':
            self.args.save_path = self.args.save_path + '_' + self.args.tag
        
        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        
        out_log_path = self.args.save_path + '/log.txt'

        ensure_path(self.args.save_path)
        
        return out_log_path

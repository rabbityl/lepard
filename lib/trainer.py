import gc
import os

import torch
import torch.nn as nn
import numpy as np

from tensorboardX import SummaryWriter
from tqdm import tqdm

from lib.timer import AverageMeter
from lib.utils import Logger, validate_gradient
from lib.tictok import Timers


class Trainer(object):
    def __init__(self, args):
        self.config = args
        # parameters
        self.start_epoch = 1
        self.max_epoch = args.max_epoch
        self.save_dir = args.save_dir
        self.device = args.device
        self.verbose = args.verbose

        self.model = args.model
        self.model = self.model.to(self.device)


        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_freq = args.scheduler_freq
        self.snapshot_dir = args.snapshot_dir

        self.iter_size = args.iter_size
        self.verbose_freq = args.verbose_freq // args.batch_size + 1
        if 'overfit' in self.config.exp_dir:
            self.verbose_freq = 1
        self.loss = args.desc_loss

        self.best_loss = 1e5
        self.best_recall = -1e5
        self.summary_writer = SummaryWriter(log_dir=args.tboard_dir)
        self.logger = Logger(args.snapshot_dir)
        self.logger.write(f'#parameters {sum([x.nelement() for x in self.model.parameters()]) / 1000000.} M\n')

        if (args.pretrain != ''):
            self._load_pretrain(args.pretrain)

        self.loader = dict()
        self.loader['train'] = args.train_loader
        self.loader['val'] = args.val_loader
        self.loader['test'] = args.test_loader

        self.timers = args.timers

        with open(f'{args.snapshot_dir}/model', 'w') as f:
            f.write(str(self.model))
        f.close()

    def _snapshot(self, epoch, name=None):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_recall': self.best_recall
        }
        if name is None:
            filename = os.path.join(self.save_dir, f'model_{epoch}.pth')
        else:
            filename = os.path.join(self.save_dir, f'model_{name}.pth')
        self.logger.write(f"Save model to {filename}\n")
        torch.save(state, filename, _use_new_zipfile_serialization=False)

    def _load_pretrain(self, resume):
        print ("loading pretrained", resume)
        if os.path.isfile(resume):
            state = torch.load(resume)
            self.model.load_state_dict(state['state_dict'])
            self.start_epoch = state['epoch']
            self.scheduler.load_state_dict(state['scheduler'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.best_loss = state['best_loss']
            self.best_recall = state['best_recall']

            self.logger.write(f'Successfully load pretrained model from {resume}!\n')
            self.logger.write(f'Current best loss {self.best_loss}\n')
            self.logger.write(f'Current best recall {self.best_recall}\n')
        else:
            raise ValueError(f"=> no checkpoint found at '{resume}'")

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']


    def inference_one_batch(self, inputs, phase):
        assert phase in ['train', 'val', 'test']
        inputs ['phase'] = phase


        if (phase == 'train'):
            self.model.train()
            if self.timers: self.timers.tic('forward pass')
            data = self.model(inputs, timers=self.timers)  # [N1, C1], [N2, C2]
            if self.timers: self.timers.toc('forward pass')


            if self.timers: self.timers.tic('compute loss')
            loss_info = self.loss( data)
            if self.timers: self.timers.toc('compute loss')


            if self.timers: self.timers.tic('backprop')
            loss_info['loss'].backward()
            if self.timers: self.timers.toc('backprop')


        else:
            self.model.eval()
            with torch.no_grad():
                data = self.model(inputs, timers=self.timers)  # [N1, C1], [N2, C2]
                loss_info = self.loss(data)


        return loss_info


    def inference_one_epoch(self, epoch, phase):
        gc.collect()
        assert phase in ['train', 'val', 'test']

        # init stats meter
        stats_meter = None #  self.stats_meter()

        num_iter = int(len(self.loader[phase].dataset) // self.loader[phase].batch_size) # drop last incomplete batch
        c_loader_iter = self.loader[phase].__iter__()

        self.optimizer.zero_grad()
        for c_iter in tqdm(range(num_iter)):  # loop through this epoch

            if self.timers: self.timers.tic('one_iteration')

            ##################################
            if self.timers: self.timers.tic('load batch')
            inputs = c_loader_iter.next()
            # for gpu_div_i, _ in enumerate(inputs):
            for k, v in inputs.items():
                if type(v) == list:
                    inputs [k] = [item.to(self.device) for item in v]
                elif type(v) in [ dict, float, type(None), np.ndarray]:
                    pass
                else:
                    inputs [k] = v.to(self.device)
            if self.timers: self.timers.toc('load batch')
            ##################################


            if self.timers: self.timers.tic('inference_one_batch')
            loss_info = self.inference_one_batch(inputs, phase)
            if self.timers: self.timers.toc('inference_one_batch')


            ###################################################
            # run optimisation
            # if self.timers: self.timers.tic('run optimisation')
            if ((c_iter + 1) % self.iter_size == 0 and phase == 'train'):
                gradient_valid = validate_gradient(self.model)
                if (gradient_valid):
                    self.optimizer.step()
                else:
                    self.logger.write('gradient not valid\n')
                self.optimizer.zero_grad()
            # if self.timers: self.timers.toc('run optimisation')
            ################################

            torch.cuda.empty_cache()

            if stats_meter is None:
                stats_meter = dict()
                for key, _ in loss_info.items():
                    stats_meter[key] = AverageMeter()
            for key, value in loss_info.items():
                stats_meter[key].update(value)

            if phase == 'train' :
                if (c_iter + 1) % self.verbose_freq == 0 and self.verbose  :
                    curr_iter = num_iter * (epoch - 1) + c_iter
                    for key, value in stats_meter.items():
                        self.summary_writer.add_scalar(f'{phase}/{key}', value.avg, curr_iter)

                    dump_mess=True
                    if dump_mess:
                        message = f'{phase} Epoch: {epoch} [{c_iter + 1:4d}/{num_iter}]'
                        for key, value in stats_meter.items():
                            message += f'{key}: {value.avg:.2f}\t'
                        self.logger.write(message + '\n')


            if self.timers: self.timers.toc('one_iteration')


        # report evaluation score at end of each epoch
        if phase in ['val', 'test']:
            for key, value in stats_meter.items():
                self.summary_writer.add_scalar(f'{phase}/{key}', value.avg, epoch)

        message = f'{phase} Epoch: {epoch}'
        for key, value in stats_meter.items():
            message += f'{key}: {value.avg:.2f}\t'
        self.logger.write(message + '\n')

        return stats_meter




    def train(self):
        print('start training...')
        for epoch in range(self.start_epoch, self.max_epoch):
            with torch.autograd.set_detect_anomaly(True):
                if self.timers: self.timers.tic('run one epoch')
                stats_meter = self.inference_one_epoch(epoch, 'train')
                if self.timers: self.timers.toc('run one epoch')

            self.scheduler.step()


            if  'overfit' in self.config.exp_dir :
                if stats_meter['loss'].avg < self.best_loss:
                    self.best_loss = stats_meter['loss'].avg
                    self._snapshot(epoch, 'best_loss')

                if self.timers: self.timers.print()

            else : # no validation step for overfitting

                if self.config.do_valid:
                    stats_meter = self.inference_one_epoch(epoch, 'val')
                    if stats_meter['loss'].avg < self.best_loss:
                        self.best_loss = stats_meter['loss'].avg
                        self._snapshot(epoch, 'best_loss')


                if self.timers: self.timers.print()

        # finish all epoch
        print("Training finish!")
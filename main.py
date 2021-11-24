import os, torch, json, argparse, shutil
from easydict import EasyDict as edict
import yaml
from datasets.dataloader import get_dataloader, get_datasets
from models.pipeline import Pipeline
from lib.utils import setup_seed
from lib.tester import get_trainer
from models.loss import MatchMotionLoss
from lib.tictok import Timers
from configs.models import architectures

from torch import optim



setup_seed(0)

def join(loader, node):
    seq = loader.construct_sequence(node)
    return '_'.join([str(i) for i in seq])

yaml.add_constructor('!join', join)


if __name__ == '__main__':
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help= 'Path to the config file.')
    args = parser.parse_args()
    with open(args.config,'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config['snapshot_dir'] = 'snapshot/%s/%s' % (config['dataset']+config['folder'], config['exp_dir'])
    config['tboard_dir'] = 'snapshot/%s/%s/tensorboard' % (config['dataset']+config['folder'], config['exp_dir'])
    config['save_dir'] = 'snapshot/%s/%s/checkpoints' % (config['dataset']+config['folder'], config['exp_dir'])
    config = edict(config)

    os.makedirs(config.snapshot_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.tboard_dir, exist_ok=True)

    if config.gpu_mode:
        config.device = torch.device("cuda:0")
    else:
        config.device = torch.device('cpu')
    
    # backup the
    if config.mode == 'train':
        os.system(f'cp -r models {config.snapshot_dir}')
        os.system(f'cp -r configs {config.snapshot_dir}')
        os.system(f'cp -r cpp_wrappers {config.snapshot_dir}')
        os.system(f'cp -r datasets {config.snapshot_dir}')
        os.system(f'cp -r kernels {config.snapshot_dir}')
        os.system(f'cp -r lib {config.snapshot_dir}')
        shutil.copy2('main.py',config.snapshot_dir)

    
    # model initialization
    config.kpfcn_config.architecture = architectures[config.dataset]
    config.model = Pipeline(config)
    # config.model = KPFCNN(config)

    # create optimizer 
    if config.optimizer == 'SGD':
        config.optimizer = optim.SGD(
            config.model.parameters(), 
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            )
    elif config.optimizer == 'ADAM':
        config.optimizer = optim.Adam(
            config.model.parameters(), 
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
    

    #create learning rate scheduler
    if  'overfit' in config.exp_dir :
        config.scheduler = optim.lr_scheduler.MultiStepLR(
            config.optimizer,
            milestones=[config.max_epoch-1], # fix lr during overfitting
            gamma=0.1,
            last_epoch=-1)

    else:
        config.scheduler = optim.lr_scheduler.ExponentialLR(
            config.optimizer,
            gamma=config.scheduler_gamma,
        )


    config.timers = Timers()

    # create dataset and dataloader
    train_set, val_set, test_set = get_datasets(config)
    config.train_loader, neighborhood_limits = get_dataloader(train_set,config,shuffle=True)
    config.val_loader, _ = get_dataloader(val_set, config, shuffle=False, neighborhood_limits=neighborhood_limits)
    config.test_loader, _ = get_dataloader(test_set, config, shuffle=False, neighborhood_limits=neighborhood_limits)
    
    # config.desc_loss = MetricLoss(config)
    config.desc_loss = MatchMotionLoss (config['train_loss'])

    trainer = get_trainer(config)
    if(config.mode=='train'):
        trainer.train()
    else:
        trainer.test()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch, sys, os, copy
sys.path.append('./')

from tqdm import tqdm as tqdm_base
def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)

from ncsnv2.models        import get_sigmas
from ncsnv2.models.ema    import EMAHelper
from ncsnv2.models.ncsnv2 import NCSNv2Deepest
from ncsnv2.losses        import get_optimizer

from parameters import pairwise_dist
from parameters import sigma_rate
from parameters import step_size
from parameters import anneal_dsm_score_estimation

from loaders          import Channels
from torch.utils.data import DataLoader

from dotmap import DotMap

# Always !!!
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False

# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";

# Model config
config          = DotMap()
config.device   = 'cuda:0'
# Inner model
config.model.ema           = True
config.model.ema_rate      = 0.999 # Exponential moving average, for stable FID scores (Song'20)
config.model.normalization = 'InstanceNorm++'
config.model.nonlinearity  = 'elu'
config.model.sigma_dist    = 'geometric'
config.model.num_classes   = 2311 # Number of train sigmas and 'N'
config.model.ngf           = 128
config.model.experiment    = 1

# Optimizer
config.optim.weight_decay  = 0.000 # No weight decay
config.optim.optimizer     = 'Adam'
config.optim.lr            = 0.0001
config.optim.beta1         = 0.9
config.optim.amsgrad       = False
config.optim.eps           = 0.001

# Training
config.training.batch_size     = 30
config.training.num_workers    = 4
config.training.n_epochs       = 200
config.training.anneal_power   = 2
config.training.log_all_sigmas = False
config.training.eval_freq      = 100 # In steps

# Data
config.data.channels       = 2 # {Re, Im}
config.data.num_pilots     = 64
config.data.noise_std      = 0.01 # 'Beta' in paper
config.data.image_size     = [240, 240] # Channel size = Nr x Nt

# Validation config
val_config = copy.deepcopy(config)
val_config.training.batch_size = 10

# Get datasets and loaders for channels
dataset     = Channels(config, data='train', experiment=config.model.experiment)
dataloader  = DataLoader(dataset, batch_size=config.training.batch_size, 
                         shuffle=True, num_workers=config.training.num_workers, 
                         drop_last=True)

# Create separate validation sets
val_dataset     = Channels(val_config, data='val', experiment=val_config.model.experiment)
val_dataloader  = DataLoader(val_dataset, batch_size=val_config.training.batch_size, 
                         shuffle=True, num_workers=val_config.training.num_workers, 
                         drop_last=True)

# pairwise_dist(dataset, tqdm)

config.model.sigma_begin = np.loadtxt(sys.path[0] + '/parameters/knee-mri_max_pairwise_dist.txt')
# config.model.sigma_rate = sigma_rate(dataset, tqdm)
config.model.sigma_rate = 0.995
config.model.sigma_end  = config.model.sigma_begin * config.model.sigma_rate ** (config.model.num_classes - 1)
config.model.step_size = step_size(config)

print(config.model.sigma_begin)
print(config.model.sigma_rate)
print(config.model.sigma_end)

# Get a model
diffuser = NCSNv2Deepest(config)
diffuser = diffuser.cuda()
# Get optimizer
optimizer = get_optimizer(config, diffuser.parameters())

# Counter
start_epoch = 0
step = 0
if config.model.ema:
    ema_helper = EMAHelper(mu=config.model.ema_rate)
    ema_helper.register(diffuser)

# Get a collection of sigma values
sigmas = get_sigmas(config)

# Always the same initial points and data for validation
val_X_list = []
val_sample = next(iter(val_dataloader))
val_X_list.append(val_sample['X'].cuda())

# More logging
config.log_path = 'models/\
sigma_begin%d_sigma_end%.1f_num_classes%.1f_sigma_rate%.1f_epochs%.1f' % (
    config.model.sigma_begin, config.model.sigma_end,
    config.model.num_classes, config.model.sigma_rate, 
    config.training.n_epochs)

if not os.path.exists(config.log_path):
    os.makedirs(config.log_path)

# No sigma logging
hook = test_hook = None

# Logged metrics
min_loss = 10000000
train_loss, val_loss, train_nrmse  = [], [], []
val_errors, val_epoch, val_nrmse = [], [], []

for epoch in tqdm(range(start_epoch, config.training.n_epochs)):
    for i, sample in tqdm(enumerate(dataloader)):
        # Safety check
        diffuser.train()
        step += 1
        
        # Move data to device
        for key in sample:
            sample[key] = sample[key].cuda()

        # Get loss on Hermitian channels
        loss, nrmse = anneal_dsm_score_estimation(
            diffuser, sample['X'], sigmas, None, 
            config.training.anneal_power, hook)
        
        # Keep a running loss
        if step == 1:
            running_loss = loss.item()
        else:
            running_loss = 0.99 * running_loss + 0.01 * loss.item()

        if loss < min_loss:
            min_loss = loss
            optim_state = diffuser.state_dict()
    
        # Log
        train_loss.append(loss.item())
        train_nrmse.append(nrmse.item())
        
        # Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # EMA update
        if config.model.ema:
            ema_helper.update(diffuser)
            
        # Verbose
        if step % config.training.eval_freq == 0:
            if config.model.ema:
                val_score = ema_helper.ema_copy(diffuser)
            else:
                val_score = diffuser
            
            # For each validation setup
            local_val_losses, local_val_nrmse = [], []
            for idx in range(1):
                with torch.no_grad():
                    val_dsm_loss, val_dsm_nrmse = \
                        anneal_dsm_score_estimation(
                            val_score, val_X_list[idx],
                            sigmas, None,
                            config.training.anneal_power,
                            hook=test_hook)
                # Store
                local_val_losses.append(val_dsm_loss.item())
                local_val_nrmse.append(val_dsm_nrmse.item())
                
            # Sanity delete
            del val_score
            # Log
            val_loss.append(local_val_losses)
            val_nrmse.append(local_val_nrmse)
                
            # Print
            if len(local_val_losses) == 1:
                print('Epoch %d, Step %d, Train Loss (EMA) %.3f, Train NRMSE %.3f, \
    Val. Loss %.3f, Min Loss %.3f' % (
                    epoch, step, running_loss, nrmse,
                    local_val_losses[0], min_loss))
            elif len(local_val_losses) == 2:
                print('Epoch %d, Step %d, Train Loss (EMA) %.3f,  Train NRMSE %.3f,\
    Val. Loss (Split) %.3f %.3f, Min Loss %.3f' % (
                    epoch, step, running_loss, nrmse, 
                    local_val_losses[0], local_val_losses[1], min_loss))
        
# Save snapshot
torch.save({'diffuser': diffuser,
            'model_state': diffuser.state_dict(),
            'optim_state': optim_state,
            'min_loss': min_loss,
            'config': config,
            'train_loss': train_loss,
            'train_nrmse': train_nrmse,
            'val_loss': val_loss,
            'val_nrmse': val_nrmse,
            'val_errors': val_errors,
            'val_epoch': val_epoch}, 
   os.path.join(config.log_path, 'final_model.pt'))
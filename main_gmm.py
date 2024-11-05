import torch
import numpy as np
import argparse

from xvfm.gmm.resampler import Resampler
from xvfm.gmm.objectives import apg_objective, rws_objective
from xvfm.gmm.apg_training import train, init_apg_models, init_rws_models

def main():
    parser = argparse.ArgumentParser('GMM Experiment')

    parser.add_argument('--data_dir', default='../../data/gmm/')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--budget', default=100, type=int)
    parser.add_argument('--num_sweeps', default=10, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--resample_strategy', default='systematic', choices=['systematic', 'multinomial'])
    parser.add_argument('--block_strategy', default='decomposed', choices=['decomposed', 'joint'])
    parser.add_argument('--num_clusters', default=3, type=int)
    parser.add_argument('--data_dim', default=2, type=int)
    parser.add_argument('--num_hidden', default=32, type=int)

    args = parser.parse_args()

    sample_size = int(args.budget / args.num_sweeps)
    CUDA = torch.cuda.is_available()
    device = torch.device('cuda:%d' % args.device)
    
    data = torch.from_numpy(np.load(args.data_dir + 'ob.npy')).float() 
    assignments = torch.from_numpy(np.load(args.data_dir + 'assignment.npy')).float()

    print('Start training for gmm clustering task..')

    if args.num_sweeps == 1: ## rws method
        model_version = 'rws-gmm-num_samples=%s' % (sample_size)
        print('version='+ model_version)
        models, optimizer = init_rws_models(
            args.num_clusters, 
            args.data_dim, 
            args.num_hidden, 
            CUDA, 
            device, 
            load_version=None, 
            lr=args.lr
            )
        train(
            rws_objective, 
            optimizer, 
            models, 
            data, 
            assignments, 
            args.num_epochs, 
            sample_size, 
            args.batch_size, 
            CUDA, 
            device
            )
        
    elif args.num_sweeps > 1: # apg sampler
        model_version = 'apg-gmm-block=%s-num_sweeps=%s-num_samples=%s' % (args.block_strategy, args.num_sweeps, sample_size)

        print('version=' + model_version)

        models, optimizer = init_apg_models(
            args.num_clusters, 
            args.data_dim, 
            args.num_hidden, 
            CUDA, 
            device, 
            load_version=None, 
            lr=args.lr
            )
        resampler = Resampler(
            args.resample_strategy, 
            sample_size, 
            CUDA, 
            device
            )
        train(
            apg_objective, 
            optimizer, 
            models, 
            data, 
            assignments, 
            args.num_epochs, 
            sample_size, 
            args.batch_size, 
            CUDA, 
            device, 
            num_sweeps=args.num_sweeps, 
            block=args.block_strategy, 
            resampler=resampler
            )
        
    else:
        raise ValueError
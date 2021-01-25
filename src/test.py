## Original code from Neuropoly, Lucas
## Code modified by Reza Azad
from __future__ import print_function, absolute_import
import _init_paths
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from Metrics import *
import torch
import numpy as np
from train_utils import *
# from Data2array import *
from pose_code.hourglass import hg
from pose_code.atthourglass import atthg
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import skimage
import pickle
from torch.utils.data import DataLoader 

# main script

def main(args):
    global cuda_available
    cuda_available = torch.cuda.is_available()
    print('load image')
    # put image into an array
    ds = load_Data_Bids2Array(path, mode=conf['mode'], split='test', aim=goal)
    print('extract mid slices')
    full = extract_groundtruth_heatmap(ds)
    full[0] = full[0][:, :, :, :, 0]
    print('retrieving ground truth coordinates')
    
    # coord_gt = retrieves_gt_coord(ds)
    # intialize metrics
    global distance_l2
    global zdis
    global faux_pos
    global faux_neg
    global tot
    distance_l2 = []
    zdis = []
    faux_pos = []
    faux_neg = []
    tot = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.att:
        model = atthg(num_stacks=args.stacks, num_blocks=args.blocks, num_classes=args.njoints)
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load('./final_code/weight_model_att', map_location='cpu')['model_weights'])
    else:
        model = hg(num_stacks=args.stacks, num_blocks=args.blocks, num_classes=args.njoints)
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load('./final_code/weight_model', map_location='cpu')['model_weights'])


    ## Get the visualization resutls of the test set
    print(full[0].shape, full[1].shape)
    full_dataset_test = image_Dataset(image_paths=full[0],target_paths=full[1], use_flip = False)
    MRI_test_loader   = DataLoader(full_dataset_test, batch_size= 4, shuffle=False, num_workers=0)
    model.eval()
    for i, (input, target, vis) in enumerate(MRI_test_loader):
        input, target = input.to(device), target.to(device, non_blocking=True)
        output = model(input) 
        output = output[-1]
        save_epoch_res_as_image2(input, output, target, epoch_num=i+1, target_th=0.5, pretext= True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verterbal disc labeling using pose estimation')
    
    ## Parameters
    parser.add_argument('--datapath', default='./final_code/prepared_ds_test', type=str,
                        help='Dataset address')                     
    parser.add_argument('--njoints', default=11, type=int,
                        help='Number of joints')
    parser.add_argument('--resume', default= False, type=bool,
                        help=' Resume the training from the last checkpoint') 
    parser.add_argument('--att', default= False, type=bool,
                        help=' Use attention mechanism') 
    parser.add_argument('-s', '--stacks', default=2, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')



    main(parser.parse_args())               
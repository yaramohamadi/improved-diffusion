''''
Code is taken from https://github.com/WisconsinAIVision/few-shot-gan-adaptation/blob/main/feat_cluster.py
'''

import random
import torch
import torch.nn as nn
from torchvision import utils
from tqdm import tqdm
import sys
import lpips
from torchvision import transforms, utils
from torch.utils import data
import os
from PIL import Image
import numpy as np


def intra_cluster_dist(baseline, dataset):

    device = 'cuda'

    with torch.no_grad():
        lpips_fn = lpips.LPIPS(net='vgg').to(device)
        preprocess = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        cluster_size = 50
        base_path = os.path.join("cluster_centers", "%s" %(dataset), "%s" % (baseline))
        avg_dist = torch.zeros([10, ])
        for k in range(10):
            curr_path = os.path.join(base_path, "c%d" % (k))
            files_list = os.listdir(curr_path)
            files_list.remove('center.png')

            random.shuffle(files_list)
            files_list = files_list[:cluster_size]
            dists = []
            for i in range(len(files_list)):
                for j in range(i+1, len(files_list)):
                    input1_path = os.path.join(curr_path, files_list[i])
                    input2_path = os.path.join(curr_path, files_list[j])

                    input_image1 = Image.open(input1_path)
                    input_image2 = Image.open(input2_path)

                    input_tensor1 = preprocess(input_image1)
                    input_tensor2 = preprocess(input_image2)

                    input_tensor1 = input_tensor1.to(device)
                    input_tensor2 = input_tensor2.to(device)

                    dist = lpips_fn(input_tensor1, input_tensor2)

                    dists.append(dist)
            dists = torch.tensor(dists)
            print ("Cluster %d:  Avg. pairwise LPIPS dist: %f/%f" %
                   (k, dists.mean(), dists.std()))
            avg_dist[k] = dists.mean()

        print ("Final avg. %f/%f" % (avg_dist[~torch.isnan(avg_dist)].mean(), avg_dist[~torch.isnan(avg_dist)].std()))


def get_close_far_members(baseline, dataset):

    device = 'cuda'

    with torch.no_grad():
        lpips_fn = lpips.LPIPS(net='vgg').to(device)
        preprocess = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        cluster_size = 50
        base_path = os.path.join("cluster_centers", "%s" %(dataset), "%s" % (baseline))
        avg_dist = torch.zeros([10, ])
        for k in range(10):
            curr_path = os.path.join(base_path, "c%d" % (k))
            files_list = os.listdir(curr_path)
            files_list.remove('center.png')

            random.shuffle(files_list)
            files_list = files_list[:cluster_size]
            dists = []
            min_dist, max_dist = 1000.0, 0.0
            min_ind, max_ind = 0, 0
            # center image
            input1_path = os.path.join(curr_path, 'center.png')
            input_image1 = Image.open(input1_path)
            input_tensor1 = preprocess(input_image1)
            input_tensor1 = input_tensor1.to(device)
            
            for i in range(len(files_list)):
                input2_path = os.path.join(curr_path, files_list[i])
                input_image2 = Image.open(input2_path)
                input_tensor2 = preprocess(input_image2)
                input_tensor2 = input_tensor2.to(device)
                dist = lpips_fn(input_tensor1, input_tensor2)
                if dist <= min_dist:
                    min_ind = i
                    min_dist = dist
                if dist >= max_dist:
                    max_ind = i
                    max_dist = dist
            
            print (min_ind, max_ind)
            if len(files_list) > 0:
                # saving the closest member
                path_closest = os.path.join(curr_path, files_list[min_ind])
                new_closest = os.path.join(curr_path, 'closest.png')
                os.system("cp %s %s" %(path_closest, new_closest))

                # saving the farthest member
                path_farthest = os.path.join(curr_path, files_list[max_ind])
                new_farthest = os.path.join(curr_path, 'farthest.png')
                os.system("cp %s %s" %(path_farthest, new_farthest))
            else:
                print("no members in cluster %d" %(k))


''''
To run, execute the following functions
'''

baseline = '/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_results/time_linear_guidance/0_9-1/evallog.csv'
dataset = 

intra_cluster_dist(baseline, dataset)
get_close_far_members(baseline, dataset)
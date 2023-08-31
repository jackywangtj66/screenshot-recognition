# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 14:25:59 2023

@author: YB9i
"""

#import torch
from PIL import Image
import numpy as np
import os
import shutil

# def gen_dataset(dir):
#     files = []
#     subdirs = [x[0] for x in os.walk(dir)][1:]
#     img_names = []
#     for subdir in subdirs:
#         names = 

# gen_dataset('.\\img')
np.random.seed(0)
file_list = []
for root,ds,fs in os.walk(".\\img"):
    for f in fs:
        file_list.append(os.path.join(root,f))

print(file_list)
labels = [file.split("\\")[-3] for file in file_list]

index = np.random.permutation(len(file_list))
file_list= np.array(file_list)[index]
labels = np.array(labels)[index]
train_split = int(len(index)*0.8)
train_files,val_files = file_list[:train_split],file_list[train_split:]
train_labels,val_labels = labels[:train_split],labels[train_split:]

#picpath = '.\img\train\{}'.format(train_labels[0])
count = {'gaming':0,'working':0,'creator':0,'entertain':0,'learning':0,'meeting':0}
for i in range(len(train_files)):
    picpath = os.path.join('.\img','train',train_labels[i])
    if not os.path.exists(picpath):
        os.makedirs(picpath)
    #print(train_labels[i])
    shutil.copy(train_files[i],os.path.join(picpath,str(count[train_labels[i]])+'.jpeg'))
    count[train_labels[i]] += 1
    
count = {'gaming':0,'working':0,'creator':0,'entertain':0,'learning':0,'meeting':0}
for i in range(len(val_files)):
    picpath = os.path.join('.\img','val',val_labels[i])
    if not os.path.exists(picpath):
        os.makedirs(picpath)
    shutil.copy(val_files[i],os.path.join(picpath,str(count[val_labels[i]])+'.jpeg'))
    count[val_labels[i]] += 1

# for i in range(len(train_files)):
#     shutil.copy(file_list[i],os.path.join(".\img\train",'{}.jpeg'.format(str(i))))
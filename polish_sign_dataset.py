from genericpath import samefile
import json
import math
import os
import random

import numpy as np
import torch.utils.data as data
import cv2
import torch
import torch.nn as nn

import utils

from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from tqdm import tqdm





class Polish_dataset(Dataset):
    def __init__(self, index_file_path, split, pose_root, sample_strategy='seq', num_samples=50, num_copies=4,
                 img_transforms=None, video_transforms=None, test_index_file=None , FACE=False):
        self.data = []
        self.index_file_path = index_file_path
        self.num_samples = num_samples
        self.video_transforms =video_transforms
        self.FACE = FACE
        self._make_dataset()
        
        
    def __len__(self):
        print("jvfjfnjf",len(self.data))
        return len(self.data)
    
    def __getitem__(self, index):
        video_id, gloss_cat = self.data[index]
        x = self._load_poses(video_id)
        # if x==None :
        #     return None
        
        # else :
        #     if self.video_transforms:
        #         x = self.video_transforms(x)
        #     y = gloss_cat
        #     return x, y, video_id
        
        
        if self.video_transforms:
                x = self.video_transforms(x)
        y = gloss_cat.lstrip("0")
        y= int(y)
        # y = torch.tensor(gloss_cat)
        
        
        # print(y)
        return x, y, video_id
        
        
    
    # def _make_dataset(self, index_file_path, split):
    def _make_dataset(self):
        
        # arr = os.listdir("video__landmarks")
        # arr = [i[0:5] for i in arr]
        
        
        arr = os.listdir("argentin_keypoints")
        arr = [(i[0:11] ,i[0:3]) for i in arr]
        self.data = arr 


        

    
    def _load_poses(self, video_id):
        
        """ Load frames of a video. Start and end indices are provided just to avoid listing and sorting the directory unnecessarily.
         """
        # print(frame_start , frame_end ,num_samples ,video_id,"dncskjnkoooo")
        pad = None
        
        poses = []
        poses = np.load("argentin_keypoints/{}.npy".format(video_id))
        face_slice = [range(132,1536)]
        poses = np.delete(poses, face_slice,1)
        # print(poses.shape[0])
        
        # print (poses.shape[0])
        frames_to_sample =seq_sampler2(0,poses.shape[0] , 51)
        # print(len(frames_to_sample))
        poses = torch.from_numpy(poses)
        poses = poses[frames_to_sample]
        
        # print(poses.shape)
        return poses 

        
        # # # # # #if FACE = True 
        # # # # # # face_slice = [range(100,1503)]
        # # # # # # poses = np.delete(poses, face_slice,1)

        # # # # # # print(poses.shape)

        
        # # # # # if len(poses.shape) == 1:
        # # # # #     poses_across_time = None
        # # # # #     # print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
        
        
                
        # # # # # return poses_across_time

def sequential_sampling(frame_start, frame_end, num_samples):
    """Keep sequentially ${num_samples} frames from the whole video sequence by uniformly skipping frames."""
    num_frames = frame_end - frame_start + 1


def sequential_sampling(frame_start, frame_end, num_samples):
    """Keep sequentially ${num_samples} frames from the whole video sequence by uniformly skipping frames."""
    num_frames = frame_end - frame_start + 1

    frames_to_sample = []
    if num_frames > num_samples:
        frames_skip = set()

        num_skips = num_frames - num_samples
        interval = num_frames // num_skips 
        # print(interval)

        for i in range(frame_start, frame_end + 1):
            if i % interval == 0 and len(frames_skip) <= num_skips:
            # if i % interval == 0 :
    
                frames_skip.add(i)

        for i in range(frame_start, frame_end + 1):
            if i not in frames_skip:
                frames_to_sample.append(i-1)
    else:
        frames_to_sample = list(range(frame_start, frame_end + 1-1))

    return frames_to_sample


def seq_sampler(start ,nb_frame , Nsample):
    frames_to_sample =[]
    if nb_frame -Nsample >0:
        intervall = nb_frame//Nsample
        missing  = nb_frame - Nsample *intervall
        toskip = random.sample( range(0,nb_frame) ,missing)
        c=0
        
        
        for i in range(0 , nb_frame -missing ,intervall ):
            if i in toskip:
                # c+=1
                c= c+1
            frames_to_sample.append(i + c)

        # print(intervall)
        # print(missing)
    print(frames_to_sample)
    print(len(frames_to_sample))



def seq_sampler2(start ,nb_frame , Nsample):
    frames_to_sample =[]
    if nb_frame -Nsample >0:
        intervall = nb_frame//Nsample
        missing  = nb_frame - Nsample *intervall
        toskip = random.sample( range(0,nb_frame-missing) ,missing )
        c=0
        
        
        for i in range(0 , nb_frame -missing ,intervall ):
            if i in toskip:
                # c+=1
                c= c+1
            frames_to_sample.append(i + c)

        # print(intervall)
        # print(missing)
    
    else :
        frames_to_sample = list(range(start, nb_frame))
        
    # print(frames_to_sample)
    # print(len(frames_to_sample))

    return frames_to_sample

def rand_start_sampling(frame_start, frame_end, num_samples):
    """Randomly select a starting point and return the continuous ${num_samples} frames."""
    num_frames = frame_end - frame_start + 1

    if num_frames > num_samples:
        select_from = range(frame_start, frame_end - num_samples + 1)
        sample_start = random.choice(select_from)
        frames_to_sample = list(range(sample_start, sample_start + num_samples))
    else:
        frames_to_sample = list(range(frame_start, frame_end + 1))

    return frames_to_sample


        
        
        
        
        
        
        
        
        
        
        
        
        
        







if __name__ == '__main__':

    
    train_dataset = Polish_dataset(index_file_path='/home/user/Documents/projet/floderwlasl/WLASL/data/splits/asl100.json', 
                           split=['train'], pose_root="video__landmarks",img_transforms=None, video_transforms=None, num_samples= 50)
    
    
    
    train_data_loader = data.DataLoader(dataset=train_dataset ,batch_size=64,
                                                    shuffle=True ,drop_last=True)
    

    for batch_idx ,sample in enumerate(train_data_loader): 
        print(batch_idx)
        x,y, video_id = sample
        print(x.shape)
        # dataa, target ,vid_id= sample
        





    
    
    





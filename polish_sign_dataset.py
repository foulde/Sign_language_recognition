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


        # exit(0)
        
        
        # # for gloss in 
        # with open(index_file_path, 'r') as f:
        #     content = json.load(f)

        # # create label encoder
        # glosses = sorted([gloss_entry['gloss'] for gloss_entry in content])

        # self.label_encoder.fit(glosses)
        # self.onehot_encoder.fit(self.label_encoder.transform(self.label_encoder.classes_).reshape(-1, 1))

        # if self.test_index_file is not None:
        #     print('Trained on {}, tested on {}'.format(index_file_path, self.test_index_file))
        #     with open(self.test_index_file, 'r') as f:
        #         content = json.load(f)

        # # make dataset
        # for gloss_entry in content:
        #     gloss, instances = gloss_entry['gloss'], gloss_entry['instances']
        #     gloss_cat = utils.labels2cat(self.label_encoder, [gloss])[0]

        #     for instance in instances:
        #         # if instance['split'] not in split:
        #         #     continue
        #         # if instance['video_id']  in arr:
        #         #     continue
        #         # print(instance['video_id'])
                
        #         if instance['split'] not in split or instance['video_id'] not in arr:
        #             continue
            

        #         frame_end = instance['frame_end']
        #         frame_start = instance['frame_start']
        #         video_id = instance['video_id']

        #         instance_entry = video_id, gloss_cat, frame_start, frame_end
        #         self.data.append(instance_entry)

    
    
    
    
    
    
    
    # def _load_poses(self, video_id, frame_start, frame_end, sample_strategy, num_samples):
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

        # # # # #         frame_end = poses.shape[0]            
            
        # # # # #     if sample_strategy == 'rnd_start':
        # # # # #         frames_to_sample = rand_start_sampling(frame_start, frame_end, num_samples)
        # # # # #     elif sample_strategy == 'seq':
        # # # # #         frames_to_sample = seq_sampler2(frame_start, frame_end, num_samples)
        # # # # #     # elif sample_strategy == 'k_copies':
        # # # # #     #     frames_to_sample = k_copies_fixed_length_sequential_sampling(frame_start, frame_end, num_samples,
        # # # # #     #                                                                 self.num_copies)
            
        # # # # #     else:
        # # # # #         raise NotImplementedError('Unimplemented sample strategy found: {}.'.format(sample_strategy))
        # # # # #     # print(len(frames_to_sample))
        # # # # #     # print(frames_to_sample)
        # # # # #     poses = poses[frames_to_sample]
        # # # # #     poses = torch.from_numpy(poses)
            
        # # # # #     # print(poses.shape)
        # # # # #     if len(poses) < num_samples:
        # # # # #         num_padding = num_samples - len(frames_to_sample)
        # # # # #         last_pose = poses[-1]
        # # # # #         # print(last_pose)
                
        # # # # #         # pad = last_pose.repeat(1, num_padding)
        # # # # #         pad = last_pose.repeat(num_padding ,1)
        # # # # #         # print(pad.shape)

        # # # # #     if pad is not None:             
        # # # # #         # print(pad.shape)
        # # # # #         poses_across_time = torch.cat([poses,pad])
        # # # # #         # print(poses_across_time.shape)
        # # # # #         # print(poses_across_time.shape)

        # # # # #     # for i in frames_to_sample:
        # # # # #     #     pose_path = os.path.join(self.pose_root, video_id, self.framename.format(str(i).zfill(5)))
        # # # # #     #     # pose = cv2.imread(frame_path, cv2.COLOR_BGR2RGB)
        # # # # #     #     pose = read_pose_file(pose_path)

        # # # # #     #     if pose is not None:
        # # # # #     #         if self.img_transforms:
        # # # # #     #             pose = self.img_transforms(pose)

        # # # # #     #         poses.append(pose)
        # # # # #     #     else:
        # # # # #     #         try:
        # # # # #     #             poses.append(poses[-1])
        # # # # #     #         except IndexError:
        # # # # #     #             print(pose_path)
            

        # # # # #     # pad = None
            

        # # # # #     # if len(frames_to_sample) < num_samples:
        # # # # #     # if len(poses) < num_samples:
        # # # # #     #     num_padding = num_samples - len(frames_to_sample)
        # # # # #     #     last_pose = poses[-1]
        # # # # #     #     pad = last_pose.repeat(1, num_padding)

        # # # # #     # poses_across_time = torch.cat(poses, dim=1)
        # # # # #     # if pad is not None:
        # # # # #     #     poses_across_time = torch.cat([poses_across_time, pad], dim=1)
        # # # # #     if  num_samples <=len(poses):
        # # # # #         poses_across_time = poses
                
        # # # # # return poses_across_time

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
        











    # arr = os.listdir("argentin_keypoints")
    # # # arr = [i[0:5] for i in arr]
    # arr = [(i[0:11] ,i[0:3]) for i in arr]
    # # print(arr[2][0]) 
    
    # # # # # # # a =np.load("argentin_keypoints/001_001_001.npy")
    # # # # # # # # print(a.shape)
    # l =[]
    # arr = os.listdir("argentin_keypoints")
    # for mot in tqdm(arr):
    #     eo = np.load("argentin_keypoints/{}".format(mot))
    #     # print("shape : {}".format(eo.shape))
    #     # print(eo.shape[0])
    #     # e =seq_sampler2(0,eo.shape[0] , 51)
    #     l.append(eo.shape[0])
    #     # print("length of e  : {}".format(len(e)))
    #     # print("e  : {}".format(e))
    #     # # print(e)
    # print(min(l) ,max(l))
        
        
    # # seq_sampler(0,120 ,50)
    # # seq_sampler2(0,120 ,50)
    # # seq_sampler2(0,120 ,50)
    # # seq_sampler2(0,50 ,50)
    
    
    # # print(random.randrange(0,120))
    # # print (random.sample(range(0, 120), 10)  )
    
        
        
    #     # sequential_sampling(0 ,)
        
    #     # if len(eo.shape) == 1:
    #     #     print(eo.shape ,  mot)
            
    # # arr = [i[0:5] for i in arr]
    # # arr = [(i[0:11] ,i[0:3]) for i in arr]
    # # print(arr[2][0]) 
    
    
    





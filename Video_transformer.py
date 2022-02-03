

from typing import Sequence
from cv2 import data
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
# import re
import math
import numpy as np 
import os
import random

from numpy.core.arrayprint import printoptions 
from tqdm import tqdm


 

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions





def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections





def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
                             



def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z ,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])











def make_table(path , id  ):
    acces ="all/"+id+".mp4"
    if os.path.exists(path):
        chemin = "argentin_keypoints/"+id
        
        # path = "videos/69451.mp4"
        # id = (os.path.basename(path)).split(".mp4")[0]
        # print(id)
        # id = re.split(".mp4" , path)
        # print(id)
        # collection_path= "video__landmarks/"
        cap = cv2.VideoCapture(path)
        # seq_length = 25        
        count=0
        sequence =[]
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
            # for frame_num in range(seq_length):
                
                # Read feed
                ret, frame = cap.read()
                if ret == False :break

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                # print(results.landmarks)
                
                # Draw landmarks
                draw_styled_landmarks(image, results)
                # extract_keypoints(results)
                # # sequence = np.append(sequence , extract_keypoints(results),axis=0)
                # extract_keypoints(results)
                sequence.append(extract_keypoints(results))
                
                # pose = []
                # for res in results.pose_landmarks.landmark:
                #     test = np.array([res.x, res.y, res.z])
                #     pose.append(test)
                # print (pose)
                # exit(0)
                count+=1
                # Show to screen
                cv2.imshow('OpenCV Feed', image)

                # # Break gracefully
                # if cv2.waitKey(10) & 0xFF == ord('q'):
                #     break
            # cap.release()
            # cv2.destroyAllWindows()
            
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
            cap.release()
                
            
        sequence =np.array(sequence)
        # return sequence
        # np.save(chemin,sequence)



if __name__ == '__main__':
    l=os.listdir("all")
    # print(len(l))
    id = l[0]
    # print(id[0:11])
    ids = [ i[0:11] for i in l]
    # print(ids)
    # print(l[3])
    # make_table("all/001_001_001.mp4" ,"001_001_001")
    chemin ='all/{}.mp4'
    c=0
    tab=[]
    for ident in tqdm(ids):
        c+=1
        # print(ident)
        # print(chemin.format(ident))
        make_table(chemin.format(ident) , ident)
        tab.append(ident)
        
    np.save("ids" , tab)
    print(c)
    # print(o.shape)
    
    # print(o)

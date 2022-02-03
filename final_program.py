import mediapipe as mp 
import numpy as np 
import cv2
# from torch._C import Stream, float32
from model_SLR import SLRGRU , test 
import torch
PATH1 = "model_e{EPOCHS}.pth"


input_dim = 258
hidden_dim= 64
output_dim =65
n_layers = 2



num_to_word = {}
num_to_word['1'] = "Opaque"
num_to_word['2'] = "Red"
num_to_word['3'] = "Green"
num_to_word['4'] = "Yellow"
num_to_word['5'] = "Bright"
num_to_word['6'] = "Light-blue"
num_to_word['7'] = "Colors"
num_to_word['8'] = "Red"
num_to_word['9'] = "Women"
num_to_word['10'] = "Enemy"
num_to_word['11'] = "Son"
num_to_word['12'] = "Man"
num_to_word['13'] = "Away"
num_to_word['14'] = "Drawer"
num_to_word['15'] = "Born"
num_to_word['16'] = "Learn"
num_to_word['17'] = "Call"
num_to_word['18'] = "Skimmer"
num_to_word['19'] = "Bitter"
num_to_word['20'] = "Sweet milk"
num_to_word['21'] = "Milk"
num_to_word['22'] = "Water"
num_to_word['23'] = "Food"
num_to_word['24'] = "Argentina"
num_to_word['25'] = "Uruguay"
num_to_word['26'] = "Country"
num_to_word['27'] = "Last name"
num_to_word['28'] = "Where"
num_to_word['29'] = "Mock"
num_to_word['30'] = "Birthday"
num_to_word['31'] = "Breakfast"
num_to_word['32'] = "Photo"
num_to_word['33'] = "Hungry"
num_to_word['34'] = "Map"
num_to_word['35'] = "Coin"
num_to_word['36'] = "Music"
num_to_word['37'] = "Ship"
num_to_word['38'] = "None"
num_to_word['39'] = "Name"
num_to_word['40'] = "Patience"
num_to_word['41'] = "Perfume"
num_to_word['42'] = "Deaf"
num_to_word['43'] = "Trap"
num_to_word['44'] = "Rice"
num_to_word['45'] = "Barbecue"
num_to_word['46'] = "Candy"
num_to_word['47'] = "Chewing-gum"
num_to_word['48'] = "Spaghetti"
num_to_word['49'] = "Yogurt"
num_to_word['50'] = "Accept"
num_to_word['51'] = "Thanks"
num_to_word['52'] = "Shut down"
num_to_word['53'] = "Appear"
num_to_word['54'] = "To land"
num_to_word['55'] = "Catch"
num_to_word['56'] = "Help"
num_to_word['57'] = "Dance"
num_to_word['58'] = "Bathe"
num_to_word['59'] = "Buy"
num_to_word['60'] = "Copy"
num_to_word['61'] = "Run"
num_to_word['62'] = "Realize"
num_to_word['63'] = "Give"
num_to_word['64'] = "Find"




mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions




def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


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
                             
def extract_keypoints_no_face(results):
    pose = np.array([[res.x, res.y, res.z ,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,lh, rh])

def testing(model): 
    
    stream = []
    # stream = torch.tensor(stream)
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                # print("SALUT !!!!")
            # for frame_num in range(seq_length):
                
                # Read feed
                ret, frame = cap.read()
                # if ret == False :break

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                image =  extract_keypoints_no_face(results)
                stream.append(image)
                
                if len(stream) == 51 : 
                    print(len(stream))
                    # h = model.init_hidden 
                    print(torch.tensor([stream]).shape)
                    out ,h = model(torch.tensor([stream], dtype=torch.float), model.init_hidden(1))
                    _, predicted = torch.max(out, 1)
                    a = predicted.item()
                    print(a)
                    print(num_to_word[str(a)])
                    stream = []
                
                
                
                
                
                # print(results.landmarks)
                
                # Draw landmarks
                # draw_styled_landmarks(image, results)
                
                # extract_keypoints(results)
                # sequence = np.append(sequence , extract_keypoints(results),axis=0)
                # extract_keypoints(results)
                # sequence.append(extract_keypoints(results))s
                
                # pose = []
                # for res in results.pose_landmarks.landmark:
                #     test = np.array([res.x, res.y, res.z])
                #     pose.append(test)
                # print (pose)
                # exit(0)
                # count+=1
                # Show to screen
                # cv2.imshow('OpenCV Feed', image)

                # # Break gracefully
                # if cv2.waitKey(10) & 0xFF == ord('q'):
                #     break
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
    cap.release()

        



if __name__ == "__main__": 
    model = SLRGRU(input_dim, hidden_dim, output_dim, n_layers)
    model.load_state_dict(torch.load(PATH1))

    testing(model)

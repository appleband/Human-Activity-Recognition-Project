import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

# Included new script to optimise video capture through treading - andy
from videocapturebufferless import VideoCaptureBufferless

# Included new script to call LSTM class
import tf_pose.estimator as estimator
from tf_pose.estimator import initialize_variables
import requests
import json
import config
from sklearn import metrics
import random
from random import randint
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0
TEST_FRAME_SIZE = 32
FRAME_BREAK_TEST = 320
# JetsonNode Sever Connection
url = config.serverPath
featureName = config.featureName
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
camid = 0
devicename = 0


# Script to determine the number of humans detected using the size of human object
'''
hnum: 0 based human index 
pos:  keypoint
'''
def get_keypoint(humans, hnum, pos):
    # check for invalid human index
    if len(humans)<= hnum:
        return None
    #check invalid keypoint, human parts may not contain certain keypoint. i.e. missing arm or leg
    if pos not in humans[hnum].body_parts.keys():
        return False
    
    part = humans[hnum].body_parts[pos]
    return part

'''
return the keypoint position in (x, y) coordinates in the image
'''
def get_point_from_part(image, part):
    image_h, image_w = image.shape[:2]
    return (int(part.x*image_w + 0.5), int(part.y * image_h + 0.5))

# count number of humans detected in an image
def human_cnt(humans):
    if humans is None:
        return 0
    return len(humans)

# Append the right data points to sequence

    # ensure that the sequence has 32 frames and contain valid data
def sequence_32(person, personData, sequence_arr,row):
    label = ''
    # Assign the appropriate Person Data to Inference Sequence Array 
    sequence_arr = personData[person]["sequence"]
    # Take the last X num of data from the sequence
    if len(sequence_arr)>TEST_FRAME_SIZE and len(row)> 0:
        sequence_arr = sequence_arr[-TEST_FRAME_SIZE:]
    
        # calling the LSTM model
        X=estimator.load_X(sequence_arr)
        # get the max array value on 2D axis        
        label=estimator.getPose(X)
        print("This is the Label from One Hot Prediction: ",label)
        printPersonAction(person,label) 

    return sequence_arr,label

def appendRow(humans, hnum,row):
    for i in range(18):
         # assumption that j= 0 is the first index
        part = get_keypoint(humans, hnum, i)
        if part is None:
            continue
        elif part is False:
            row.append(0)
            row.append(0)
            continue
        pos =  get_point_from_part(image, part)
        #print('\t No: %2d    Name [ %s ] \t X: %3d \t Y: %3d \t Score: %f'%( part.part_idx, part.get_part_name(),  pos[0] , pos[1] , part.score))
        cv2.putText(image,str(part.part_idx),  (pos[0] + 10, pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        if (pos[0] > 0 and  pos[1] > 0):
            row.append(pos[0]) 
            row.append(pos[1]) 
    
    return row

def assignDataToPerson(person, personData, row):
    personData[person]["sequence"].append(row)
    return personData

def printPersonData(personData):
    print(personData)

def printPersonAction(person, label):
    y = 50
    if person == 'B':
        y = 70 
    elif person == 'C':
        y = 90   
    elif person == 'D':
        y = 110     
    elif person == 'E':
        y = 130 
    cv2.putText(image,"Person %s %s" % (person,label),(10, y),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    # to stream through motion, replace default= 0 to "http://localhost:8081"
    parser.add_argument('--camera', type=int, default="0")
    # changd the default --resize resolution to 368x368 or 432x368                                                                                                                                                                                       
    parser.add_argument('--resize', type=str, default='368x368',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    
    #parser.add_argument('--tensorrt', type=str, default="False",
     #                   help='for tensorrt process.')
    args = parser.parse_args()

    # Set up the LSTM models initialiser
    (sess, accuracy, pred, optimizer) = initialize_variables()
    # w & h represents the resolution size of the intended inference model
    #logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    # amended from cv2.VideoCapture(args.camera) to threaded VideoCaptureBufferless(args.camera)
    cam = cv2.VideoCapture(args.camera)
    #cam = VideoCaptureBufferless(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))


    # track the number of passing frames
    #frameId = 0
    frame_number=0
    sequence_arr=[]
    person = ["A", "B", "C", "D", "E" ]
    personData = {
        "A": {
            "sequence" : [] 
            },
        "B": {
            "sequence" : [] 
            },
        "C": {
            "sequence" : [] 
            },
        "D": {
            "sequence" : [] 
            },
        "E": {
            "sequence" : [] 
            },
  
    }
    label = ''
 
    while True:
        ret_val, image = cam.read()
        if ret_val == True:
            #logger.debug('image process+')
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            
            human_num = human_cnt(humans)
            print('\nFrame: %5d \t Number of humans detected: %2d' % (frame_number+1, human_num))
            row = []
            # print keypoint data (max 18 points) for each human detected - andy
            for j in range(human_num):
                row = []  # clear row values to prevent concatination of data A & B
                print('\nPerson: ',person[j])
                ### Control of data sequencing to split person A & B must be done here
                #print and assign the keypoint to row_array
                row = appendRow(humans,j,row)
                # Store row data by person
                personData = assignDataToPerson(person[j],personData,row)

                # form the 32 frame sequence array, appending 0 to missing data
                sequence_arr,label = sequence_32(person[j],personData,sequence_arr,row)
                #print("\nPerson: %s \nSequence Array:\n %s \n" % (person[j],sequence_arr))

            #logger.debug('postprocess+')
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            try:
                if label == '':
                    label = "No Activity Detected"
                ## It seems that 'r' is a file path that will be loaded into the y_path
                #r = requests.post(url, data=json.dumps({
                 #   "featureName":featureName,
                  #  "label":label,
                  #  "camId": camid,
                  #  "deviceName": devicename}),
                  #  headers=headers)
            except:
                print("Check the connection with the node server")

            print("Human Activity Prediction: ",label) 
            cv2.putText(image,"No. of people: %d" % (human_num),(10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
            #cv2.putText(image,label,(50, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)

            #logger.debug('show+')
            cv2.putText(image,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.imshow('Human Activity Recognition Demo', image)
            fps_time = time.time()
            # Press 'esc'key 27 to end process
            if cv2.waitKey(1) == 27:
                break
            #logger.debug('finished+')
            frame_number = frame_number + 1

            # quick Sequencing testing
            if frame_number == FRAME_BREAK_TEST :
                break
        else:
            cam = cv2.VideoCapture(args.camera)

    cam.release()
    cv2.destroyAllWindows()
    print("\n------------ Demo has ended ------------")
    print("Total No. of frames captured:         ",frame_number)
    print("No. of group of %d frames sequenced:   %d " % (TEST_FRAME_SIZE, int(frame_number / TEST_FRAME_SIZE)))
    print("No. of ungrouped frames remaining:    ", frame_number % TEST_FRAME_SIZE)
    #printPersonData(personData)
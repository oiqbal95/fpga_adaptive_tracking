from ctypes import *
import numpy as np
import cv2
import vart
import os
import xir
import argparse
import torchvision
import numpy as np
from collections import OrderedDict
from src.datasets import get_dataset
import importlib
from src.kalman_filter import KFilter, Box
from src.dpu_utils import get_child_subgraph_dpu, runDPU
from src.pytracking_utils_w import Sequence, _read_image, preprocess, initial_preprocess, get_parameters
from src.yolo_utils import runYoloDPU
import time

divider='---------------------------'


def app(image_dir,threads,eco_model, seq: Sequence, key_frames = 5):
    #Timing variables
    time_init = 0
    time_eco_init = 0
    time_eco_track = 0

    time_init_start = time.perf_counter()
    
    print(divider)

    # List images in directory
    runTotal = 100
    #listimage=os.listdir(image_dir)
    #listimage.sort()

    # Create ECO DPU runner
    g = xir.Graph.deserialize(eco_model)
    subgraphs = get_child_subgraph_dpu(g)
    ecoRunner = vart.Runner.create_runner(subgraphs[0], "run")

    # Create YOLO DPU runner
    yolo_model = 'yolov3.xmodel'
    y = xir.Graph.deserialize(yolo_model)
    yoloSubgraphs = get_child_subgraph_dpu(y)
    yoloRunner = vart.Runner.create_runner(yoloSubgraphs[0], "run")



    # Initialize Camera
    ramp_frames = 10
    camera = cv2.VideoCapture(0)
    camera.open(0, cv2.CAP_V4L2)
    camera.set(3, 640)
    camera.set(4, 480)

    for i in range(ramp_frames):
        _, temp = camera.read()

    time_init = time.perf_counter() - time_init_start

    # Run loop
    print("[INFO][app.py] Running loop: ")
    #for i in range(runTotal):
    bboxes = []
    for frame_num in range(runTotal):
        
        # First frame: Initialize ECO, Create Kalman Filter
        if frame_num == 0: 
            time_eco_init_start = time.perf_counter()
            #Read image
            #image = _read_image(os.path.join(image_dir,listimage[frame_num]))
            _, image = camera.read()
            #cv2.imshow("Image", image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            
            #Get initial bounding box using yolo
            initial_bbox = runYoloDPU(yoloRunner, image)

            #Limit initial bounding box size
            x, y, w, h = initial_bbox

            mid_x = (x + w/2)
            mid_y = (y + h/2)
            g_w = 65
            g_h = 30
            g_x = mid_x - g_w / 2
            g_y = mid_y - g_h / 2
            initial_bbox = [g_x, g_y, g_w, g_h]

            #Get initial info
            init_info = seq.init_info()
            info = seq.init_info()
            info['init_bbox'] = initial_bbox
            init_info['init_bbox'] = initial_bbox

            #Get patches
            _, im_patches, _, sample_pos, _, img_sample_sz, _ = initial_preprocess(image, initial_bbox)        
        
            #Get previous output
            out = {'target_bbox': initial_bbox}
            info['previous_output'] = out
            
            #Run DPU
            features = runDPU(ecoRunner, im_patches)
            
            # Create parameters object with FPGA produced features
            params = get_parameters(features)

            # Create the ECO Tracker Class
            tracker_module = importlib.import_module('pytracking.tracker.{}'.format('eco'))
            tracker_class = tracker_module.get_tracker_class()
            tracker = tracker_class(params)
            tracker.visdom = None

            #Run ECO Initialize
            print("[INFO][app.py] Running initialize... ")
            out,img_sample_sz,sample_scales = tracker.initialize(image, init_info, features)
            x = int(out['target_bbox'][0])
            y = int(out['target_bbox'][1])
            w = int(out['target_bbox'][2])
            h = int(out['target_bbox'][3])
            prev_output = OrderedDict(out)

            det_box = Box()
            det_box.x = x
            det_box.y = y
            det_box.w = w
            det_box.h = h

            #Create KF
            dt = 1/10
            kf = KFilter(dt)
            
            #Update KF
            updated_bbox = kf.update(det_box)
            bbox = [x,y,w,h]
            bboxes.append(bbox)

            time_eco_init = time.perf_counter() - time_eco_init_start

            #Save frame with bbox
            #image = cv2.rectangle(image, (bbox[0],bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), (0,255,0), 1)
            #frame_name = 'results/'+str(frame_num+1).zfill(4)+'.png'
            #cv2.imwrite(frame_name, image)


        # Key frame: Update Kalman Filter
        elif frame_num>0 and frame_num%key_frames==0:
            time_eco_track_start = time.perf_counter()
            #Read image
            #image = _read_image(os.path.join(image_dir,listimage[frame_num]))
            _, image = camera.read()


            #Get info
            info = seq.frame_info(frame_num)
            
            #Get patches
            im_patches, _ = preprocess(image, info, sample_pos, sample_scales, img_sample_sz)

            #Get previous output
            info['previous_output'] = prev_output
            
            #Run DPU
            features = runDPU(ecoRunner, im_patches)
            
            #Run ECO Track
            out, sample_pos, sample_scales = tracker.track(features, image, info)

            #Pass bounding box to next frame
            prev_output = OrderedDict(out)
            x = int(out['target_bbox'][0])
            y = int(out['target_bbox'][1])
            w = int(out['target_bbox'][2])
            h = int(out['target_bbox'][3])
               
            #Update KF
            det_box = Box()
            det_box.x = x
            det_box.y = y
            det_box.w = w
            det_box.h = h
            updated_bbox = kf.update(det_box)

            #Append and print bouding box
            bbox = [x,y,w,h]
            bboxes.append(bbox)
            
            #Save frame with bbox
            #image = cv2.rectangle(image, (bbox[0],bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), (0,255,0), 1)
            #frame_name = 'results/'+str(frame_num+1).zfill(4)+'.png'
            #cv2.imwrite(frame_name, image)

            #print(bbox)

            time_eco_track += time.perf_counter() - time_eco_track_start

        
        elif frame_num%key_frames!=0:
            time_eco_track_start = time.perf_counter()
            #Predict Update KF
            bounding_box = kf.predict()
            kf_x = bounding_box.x
            kf_y = bounding_box.y
            kf_w = bounding_box.w
            kf_h = bounding_box.h

            #Pass bounding box to next frame
            out = {'target_bbox': [kf_x,kf_y,kf_w,kf_h]}
            prev_output = OrderedDict(out)

            #Read and write image
            _, image = camera.read()
            image = cv2.rectangle(image, (kf_x,kf_y), (kf_x+kf_w,kf_y+kf_h), (0,255,0), 1)
            frame_name = 'results/'+str(frame_num+1).zfill(4)+'.png'
            cv2.imwrite(frame_name, image)

            #Append and print bouding box
            bbox = [kf_x,kf_y,kf_w,kf_h]
            bboxes.append(bbox)
            #print(bbox)

            time_eco_track += time.perf_counter() - time_eco_track_start

    time_total = time.perf_counter() - time_init_start

    #Save numpy array with bounding boxes
    #bboxes = np.array(bboxes)
    #print("Bboxes shape: ", bboxes.shape)
    #np.save('bboxes', bboxes)

    #Print timing
    print(divider)
    print("---------- Timing analysis ----------")
    print("Total time: ", time_total)
    print("Initialize time: ", time_init)
    print("Eco initialize time: ", time_eco_init)
    print("Eco average track time: ", time_eco_track/(runTotal-1))
    return


# only used if script is run as 'main' from command line
def main():
    
  # construct the argument parser and parse the arguments
  #TODO: add key_frame argument
  ap = argparse.ArgumentParser()  
  ap.add_argument('-d', '--image_dir', type=str, default='dataset/Basketball/img', help='Path to folder of images. Default is images')
  ap.add_argument('-t', '--threads',   type=int, default=1,        help='Number of threads. Default is 1')
  ap.add_argument('-m', '--model',     type=str, default='eco.xmodel', help='Path of xmodel. Default is CNN_zcu102.xmodel')
  ap.add_argument('--sequence', type=str, default='Basketball', help='Sequence number or name.')
  args = ap.parse_args()  
  print(divider)
  print ('Command line options:')
  print (' --image_dir : ', args.image_dir)
  print (' --threads   : ', args.threads)
  print (' --model     : ', args.model)
  print(divider)
  dataset = get_dataset('otb')
  dataset = dataset[args.sequence]

  app(args.image_dir,args.threads,args.model,dataset)

if __name__ == '__main__':
  main()


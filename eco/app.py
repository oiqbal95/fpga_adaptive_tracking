from ctypes import *
import numpy as np
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
from src.pytracking_utils import Sequence, _read_image, preprocess, initial_preprocess, get_parameters


divider='---------------------------'


def app(image_dir,threads,model, seq: Sequence, key_frames = 5):
    
    print(divider)

    # List images in directory
    listimage=os.listdir(image_dir)
    listimage.sort()

    # Deserialize subgraph
    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)

    # Creating DPU runners
    dpuRunner = vart.Runner.create_runner(subgraphs[0], "run")

    # Run loop
    print("[INFO][app.py] Running loop: ")
    #for i in range(runTotal):
    bboxes = []
    for frame_num in range(len(listimage)):
        # First frame: Initialize ECO, Create Kalman Filter
        if frame_num == 0: 
            #Read image - 2nd argument can be one of the following - "noise", "quantization" or "normal"
            image = _read_image(os.path.join(image_dir,listimage[frame_num]),'normal')

            #Get initial info of sequence
            init_info = seq.init_info()
            info = seq.init_info()

            #Get patches
            _, im_patches, _, sample_pos, _, img_sample_sz, _ = initial_preprocess(image, info)           
        
            #Get previous output
            out = {'target_bbox': info.get('init_bbox')}
            info['previous_output'] = out
            
            #Run DPU
            features = runDPU(dpuRunner, im_patches)
            
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

        # Key frame: Update Kalman Filter
        elif frame_num>0 and frame_num%key_frames==0:
            #Read image
            image = _read_image(os.path.join(image_dir,listimage[frame_num]))

            #Get info
            info = seq.frame_info(frame_num)
            
            #Get patches
            im_patches, _ = preprocess(image, info, sample_pos, sample_scales, img_sample_sz)

            #Get previous output
            info['previous_output'] = prev_output
            
            #Run DPU
            features = runDPU(dpuRunner, im_patches)
            
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
            print(bbox)

        
        elif frame_num%key_frames!=0:
            #Predict Update KF
            bounding_box = kf.predict()
            kf_x = bounding_box.x
            kf_y = bounding_box.y
            kf_w = bounding_box.w
            kf_h = bounding_box.h

            #Pass bounding box to next frame
            out = {'target_bbox': [kf_x,kf_y,kf_w,kf_h]}
            prev_output = OrderedDict(out)

            #Append and print bouding box
            bbox = [kf_x,kf_y,kf_w,kf_h]
            bboxes.append(bbox)
            print(bbox)

    #Save numpy array with bounding boxes
    bboxes = np.array(bboxes)
    print("Bboxes shape: ", bboxes.shape)
    np.save('bboxes', bboxes)
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


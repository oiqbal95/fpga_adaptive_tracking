#Performs YOLO object detection from video captured on webcam and shows detections and bounding box

from numpy.lib.financial import nper
import cv2
import numpy as np
import vart
import xir
import argparse
import time


import src.kalman_filter as kf
from src.utils import *


divider = "---------------------------------------------------"

if __name__ == '__main__':

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()  
    ap.add_argument('-t', '--threads',      type=int, default=1,                        help='Number of threads. Default is 1')
    ap.add_argument('-m', '--model',        type=str, default='yolov3_voc_tf.xmodel',   help='Path of xmodel. Default is yolov3_voc_tf.xmodel')
    ap.add_argument('-f', '--total_frames', type=int, default=100,                      help='Total frames to capture' )
    ap.add_argument('-k', '--key_frame',    type=int, default=10,                       help='Keyframes where detection is performed' )

    args = ap.parse_args()
    model = args.model 
    threads = args.threads 
    total_frames = args.total_frames
    key_frame = args.key_frame

    print(divider)
    print('YOLO + KF Adaptive subsampling implementation')
    print ('Command line options:')
    print (' --threads      : ', threads)
    print (' --model        : ', model)
    print (' --total_frames : ', total_frames)
    print (' --key_frames   : ', key_frame)
    print(divider)

    #Arguments
    classes_path = "./image/voc_classes.txt"
    anchors_path = "./model_data/yolo_anchors.txt"


    #Timer: Initialization start
    tic = time.perf_counter()
    
    #Get subgraphs from model
    print("[INFO] Deserializing model subgraphs...")
    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    #print("Subgraphs: ", subgraphs)

    #Create DPU Runner
    print("[INFO] Creating DPU Runner...")
    dpu = vart.Runner.create_runner(subgraphs[0], "run")

    #Get DPU info
    #print("[INFO] DPU properties: ")
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim1 = tuple(outputTensors[0].dims)
    output_ndim2 = tuple(outputTensors[1].dims)
    output_ndim3 = tuple(outputTensors[2].dims)
    #print("Input tensors: ", inputTensors)
    #print("Output tensors: ", outputTensors)

    #Initialize camera
    print("[INFO] Camera init...")
    camera = cv2.VideoCapture(0)
    camera.open(0, cv2.CAP_V4L2)
    camera.set(3, 640)
    camera.set(4, 480)

    #Timer: Initialization end
    toc = time.perf_counter()
    print("Initialization delay: ", toc-tic)


    #Timing: Variable definition
    imageCapture = 0
    imagePreprocessing = 0
    DPUprocessing = 0
    postProcess = 0 
    roiDelay = 0
    drawDelay = 0
    kfUpdate = 0
    kfPredict = 0

    #Kalman filter variables
    bounding_box = kf.bbox(0, 0, 0, 0)
    key_frame_count = 0
    delta_t = .1

    #Create KF
    k1 = kf.KFilter(delta_t)


    #Timer: Total frame
    total_tic = time.perf_counter()

    print("[INFO] Running algorithm...")
    #Look through list of images and perform detection
    for frame_num in range(total_frames):
        
        #Timing: image capture start
        tic = time.perf_counter()

        #Capture image
        _, image = camera.read()
        image_size = image.shape[:2]

        #Timing: image capture end
        toc = time.perf_counter()
        imageCapture += toc - tic




        #if key_frame, perform detection and KF update, else KF predict
        if frame_num % key_frame == 0:
            key_frame_count += 1

            #Timing: image reprocessing start
            tic = time.perf_counter()

            #Preprocess image
            #img = [np.array(pre_process(image, (416, 416)), dtype=np.float32)]
            #img = cv2.resize(image, (416,416), interpolation=cv2.INTER_LINEAR)    
            img = [np.array(preprocessing(image,(416,416)), dtype=np.float32)]

            #Timing: image preprocessing end
            toc = time.perf_counter()
            imagePreprocessing += toc - tic



            #Timing: runDPU start
            tic = time.perf_counter()

            #Prepare input and output data
            inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]
            outputData = [np.empty(output_ndim1, dtype=np.float32, order="C"),np.empty(output_ndim2, dtype=np.float32, order="C"),np.empty(output_ndim3, dtype=np.float32, order="C")]
            
            #Load image into inputData
            imageRun = inputData[0]
            imageRun[0, ...] = img[0].reshape(input_ndim[1:])

            #Run DPU
            job_id = dpu.execute_async(inputData,outputData)
            dpu.wait(job_id)
            
            #Timing: runDPU end
            toc = time.perf_counter()
            DPUprocessing += toc - tic



            #Timing: post processing start
            tic = time.perf_counter()

            #Extract yolo outputs
            yolo_outputs = [outputData[0], outputData[1], outputData[2]]
            
            #Post-process outputs
            out_boxes, out_scores, out_classes = eval(yolo_outputs, image_size, classes_path, anchors_path)

            #Timing: post processing end
            toc = time.perf_counter()
            postProcess += toc - tic



            #Timing: KF Update start
            tic = time.perf_counter()

            if out_boxes.size !=0:
                x1, y1, x2, y2 = out_boxes[0]
                bounding_box = kf.bbox(x1, y1, x2, y2)
            else:
                bounding_box = bounding_box

            #Update KF
            bounding_box = k1.update(bounding_box)

            #Timing: KF Update end
            toc = time.perf_counter()
            kfUpdate += toc - tic



            #Timing: draw and show start
            tic = time.perf_counter()
            

            #Computer ROI
            result_image = roi(image, bounding_box)
            cv2.imshow("Result", result_image)
            cv2.waitKey(1)


            #Show image
            #Draw rectangle 
            #color = (0,0,255)
            #thickness = 1
            #start_point = (bounding_box.y, bounding_box.x)
            #end_point = (bounding_box.y+bounding_box.h,bounding_box.x+bounding_box.w)
            #result_image = cv2.rectangle(image, start_point, end_point, color, thickness)
            #cv2.imshow("Result", result_image)
            #cv2.waitKey(1)

            #Timing: draw and show end
            toc = time.perf_counter()
            drawDelay += toc - tic

        else:

            #Timing: KF Update start
            tic = time.perf_counter()
            
            #Predict Update KF
            bounding_box = k1.predict()

            #Timing: KF Update end
            toc = time.perf_counter()
            kfPredict += toc - tic



            #Timing: roi and show start
            tic = time.perf_counter()

            #Computer ROI
            result_image = roi(image, bounding_box)
            cv2.imshow("Result", result_image)
            cv2.waitKey(1)

            #Timing: roi and show end
            toc = time.perf_counter()
            roiDelay += toc - tic
        

    #Destroy cv2 windows
    cv2.destroyAllWindows()

    #Timing 
    print(divider)
    total_toc = time.perf_counter()
    print("---------------- Timing statistics ----------------")
    print("Total time elapsed: ", total_toc-total_tic)
    print("Total frames: ", total_frames)
    print("Time per frame: ", (total_toc - total_tic) / total_frames)
    print("FPS: ", total_frames / (total_toc - total_tic))

    print(divider)
    
    print("Average capture time:        %.6f" % (imageCapture / total_frames) ) 
    print("Average preprocessing time:  %.6f" % (imagePreprocessing / key_frame_count ) )
    print("Average run DPU time:        %.6f" % (DPUprocessing / key_frame_count ) )
    print("Average Postprocessing time: %.6f" % (postProcess / key_frame_count ))
    print("Average KF Update time:      %.6f" % (kfUpdate / key_frame_count ) )
    print("Average KF Update time:      %.6f" % (kfPredict / ( total_frames - key_frame_count) )) 
    print("Average draw and show time:  %.6f" % (drawDelay / key_frame_count) )
    print("Average roi and show time:   %.6f" % (roiDelay / ( total_frames - key_frame_count) )) 
    print(divider)



    



    

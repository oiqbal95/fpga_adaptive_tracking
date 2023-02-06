# adaptive_tracking
- tested on Xilinx ZCU102 
- xmodels generated with Vitis AI

## eco
- run app.py to start tracker
- pre-processing block can be tuned for quantization and noise experimentation -- insert argument "noise", "quantization", "roi" or "normal" while reading in images in the app.py script. To modify the tuning parameters go to the _read_image function in eco/src/pytracking_utils
- energy model provided in eco/energy_model -- save output ROIs per video to area_per_video.npy and run mobisys_energy_model.py 
- current sequence is in eco/dataset/Basketball/img



## yoloV3 + kf
- unzip yolov3 .xmodel
- run app_roi.py to capture frames for webcam, run algorithm and display results.
- run app_roi_save.py to capture frames from webcam, run algorithm and save frames for later postprocessing.



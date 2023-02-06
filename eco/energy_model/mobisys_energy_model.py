#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 20:40:10 2022

@author: ame
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 17:21:14 2021

@author: ame
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 16:22:42 2021

@author: ame
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 15:27:50 2021

@author: ame
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:48:56 2020

@author: iqbal
"""

import numpy as np
import os
import cv2
# Assuming image sensor B1, Res (3264*2448); B2, Res (2592*1944); B3, Res (752*480)
def energy_seq_outdoors(alpha_1, c_2, R, T_exp, N):
    f = np.sqrt((c_2 * N)/(alpha_1*T_exp))
    E = alpha_1*R*T_exp*f + ((R*c_2*N)/f)
    return E

def rescale_area(filename, res_list, B):
    area_per_video = np.load(filename, allow_pickle=True)   
    rescaled_area_per_video = []
    for i, area_per_frame in enumerate(area_per_video):
        ratio = B/res_list[i]
        rescaled_area_per_frame = []
        for j, area in enumerate(area_per_frame):
            rescaled_area_per_frame.append(area*ratio) 
        rescaled_area_per_video.append(rescaled_area_per_frame)
    return rescaled_area_per_video

def pixel_2_power(rescaled_area_per_video, alpha_1, c_2, R, T_exp):
    energy_test_data = 0
    for i, rescaled_area_per_frame in enumerate(rescaled_area_per_video):
        energy_per_video = 0
        for j, area in enumerate(rescaled_area_per_frame):

            energy_per_video += area*(595+2800+677+4.6)
        energy_test_data += (energy_per_video/(j+1))
    return energy_test_data/len(rescaled_area_per_video)



data_dir = '/home/ame/Adaptive_Subsampling/TB_100/test'
listing = os.listdir(data_dir)
listing.sort()
res_list = []
for vid in range(len(listing)):
    current_main_dir = os.path.join(data_dir,listing[vid],listing[vid],'img')
    list_imgs = os.listdir(current_main_dir)
    list_imgs.sort()
    img = cv2.imread(os.path.join(current_main_dir,list_imgs[0]))
    vid_res = img.shape[0]*img.shape[1]
    res_list.append(vid_res)
        


    
filename = 'area_per_video.npy'      
b1_rescaled_video = rescale_area(filename, res_list, (3264*2448)) #B1
b2_rescaled_video = rescale_area(filename, res_list, (2592*1944)) #B2
b3_rescaled_video = rescale_area(filename, res_list, (752*480)) #B2
P_b1 = pixel_2_power(b1_rescaled_video, 0.000004, 159, 30, 0.00005)
P_b2 = pixel_2_power(b2_rescaled_video, 0.00000082, 93, 30, 0.00005)
P_b3 = pixel_2_power(b3_rescaled_video, 0.00000335, 13.1, 30, 0.00005)
print('P_B1 energy consumption in mJ', P_b1*10**-9)
print('P_B2 energy consumption  in mJ', P_b2*10**-9)
print('P_B3 energy consumption  in mJ', P_b3*10**-9)
print('Full resolution B1 sensor energy consumption in mJ:',(595+2800+677+4.6)*(3264*2448)*10**-9)
print('Full resolution B2 sensor energy consumption  in mJ:',(595+2800+677+4.6)*(2592*1944)*10**-9)
print('Full resolution B3 sensor energy consumption  in mJ:',(595+2800+677+4.6)*(752*480)*10**-9)

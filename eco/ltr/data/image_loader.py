import jpeg4py
import cv2 as cv
from PIL import Image
import numpy as np
import os 
import pandas

davis_palette = np.repeat(np.expand_dims(np.arange(0,256), 1), 3, 1).astype(np.uint8)
davis_palette[:22, :] = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                         [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                         [64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],
                         [64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],
                         [0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],
                         [0, 64, 128], [128, 64, 128]]


def default_image_loader(path):
    """The default image loader, reads the image from the given path. It first tries to use the jpeg4py_loader,
    but reverts to the opencv_loader if the former is not available."""
    if default_image_loader.use_jpeg4py is None:
        # Try using jpeg4py
        im = jpeg4py_loader(path)
        if im is None:
            default_image_loader.use_jpeg4py = False
            print('Using opencv_loader instead.')
        else:
            default_image_loader.use_jpeg4py = True
            return im
    if default_image_loader.use_jpeg4py:
        return jpeg4py_loader(path)
    return opencv_loader(path)

default_image_loader.use_jpeg4py = None


def jpeg4py_loader(path,seq_path,frame_id):
    """ Image reading using jpeg4py https://github.com/ajkxyz/jpeg4py"""
    try:
        return jpeg4py.JPEG(path).decode()
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None


def opencv_loader(path,seq_path,frame_id):
    """ Read image using opencv's imread function and returns it in rgb format"""
    try:
        im = cv.imread(path, cv.IMREAD_COLOR)
        gnd_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(gnd_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        currentline=gt[frame_id]
        x_min = int(currentline[0])
        y_min = int(currentline[1])
        x_max = int(currentline[0]+currentline[2])
        y_max = int(currentline[1]+currentline[3])
        mask = np.zeros([im.shape[0],im.shape[1],im.shape[2]],dtype=np.uint8)
        mask[y_min:y_max,x_min:x_max,:] = im[y_min:y_max,x_min:x_max,:]
        #print("HEEEEEEEEEEEEEEEEEEE")
        #print(x)
        #print(path)
        # convert to rgb and return
        return cv.cvtColor(mask, cv.COLOR_BGR2RGB)
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None


def jpeg4py_loader_w_failsafe(path):
    """ Image reading using jpeg4py https://github.com/ajkxyz/jpeg4py"""
    try:
        return jpeg4py.JPEG(path).decode()
    except:
        try:
            im = cv.imread(path, cv.IMREAD_COLOR)

            # convert to rgb and return
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        except Exception as e:
            print('ERROR: Could not read image "{}"'.format(path))
            print(e)
            return None


def opencv_seg_loader(path):
    """ Read segmentation annotation using opencv's imread function"""
    try:
        return cv.imread(path)
        #print(path)
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None


def imread_indexed(filename):
    """ Load indexed image with given filename. Used to read segmentation annotations."""

    im = Image.open(filename)
    print("ADIB")
    annotation = np.atleast_3d(im)[...,0]
    return annotation


def imwrite_indexed(filename, array, color_palette=None):
    """ Save indexed image as png. Used to save segmentation annotation."""

    if color_palette is None:
        color_palette = davis_palette

    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')

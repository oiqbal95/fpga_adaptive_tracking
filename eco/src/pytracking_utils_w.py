import cv2
from collections import OrderedDict
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from pytracking.features import augmentation
import importlib
import math


def _read_image(image_file: str):
        im = cv2.imread(image_file)
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def numpy_to_torch(a: np.ndarray):
    return torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0)

def imread_indexed(filename):
    """ Load indexed image with given filename. Used to read segmentation annotations."""

    im = Image.open(filename)
    
    annotation = np.atleast_3d(im)[...,0]
    return annotation

class Sequence:
    """Class for the sequence in an evaluation."""
    def __init__(self, name, frames, dataset, ground_truth_rect, ground_truth_seg=None, init_data=None,
                 object_class=None, target_visible=None, object_ids=None, multiobj_mode=False):
        self.name = name
        self.frames = frames
        self.dataset = dataset
        self.ground_truth_rect = ground_truth_rect
        self.ground_truth_seg = ground_truth_seg
        self.object_class = object_class
        self.target_visible = target_visible
        self.object_ids = object_ids
        self.multiobj_mode = multiobj_mode
        self.init_data = self._construct_init_data(init_data)
        self._ensure_start_frame()

    def _ensure_start_frame(self):
        # Ensure start frame is 0
        start_frame = min(list(self.init_data.keys()))
        if start_frame > 0:
            self.frames = self.frames[start_frame:]
            if self.ground_truth_rect is not None:
                if isinstance(self.ground_truth_rect, (dict, OrderedDict)):
                    for obj_id, gt in self.ground_truth_rect.items():
                        self.ground_truth_rect[obj_id] = gt[start_frame:,:]
                else:
                    self.ground_truth_rect = self.ground_truth_rect[start_frame:,:]
            if self.ground_truth_seg is not None:
                self.ground_truth_seg = self.ground_truth_seg[start_frame:]
                assert len(self.frames) == len(self.ground_truth_seg)

            if self.target_visible is not None:
                self.target_visible = self.target_visible[start_frame:]
            self.init_data = {frame-start_frame: val for frame, val in self.init_data.items()}

    def _construct_init_data(self, init_data):
        if init_data is not None:
            if not self.multiobj_mode:
                assert self.object_ids is None or len(self.object_ids) == 1
                for frame, init_val in init_data.items():
                    if 'bbox' in init_val and isinstance(init_val['bbox'], (dict, OrderedDict)):
                        init_val['bbox'] = init_val['bbox'][self.object_ids[0]]
            # convert to list
            for frame, init_val in init_data.items():
                if 'bbox' in init_val:
                    if isinstance(init_val['bbox'], (dict, OrderedDict)):
                        init_val['bbox'] = OrderedDict({obj_id: list(init) for obj_id, init in init_val['bbox'].items()})
                    else:
                        init_val['bbox'] = list(init_val['bbox'])
        else:
            init_data = {0: dict()}     # Assume start from frame 0

            if self.object_ids is not None:
                init_data[0]['object_ids'] = self.object_ids

            if self.ground_truth_rect is not None:
                if self.multiobj_mode:
                    assert isinstance(self.ground_truth_rect, (dict, OrderedDict))
                    init_data[0]['bbox'] = OrderedDict({obj_id: list(gt[0,:]) for obj_id, gt in self.ground_truth_rect.items()})
                else:
                    assert self.object_ids is None or len(self.object_ids) == 1
                    if isinstance(self.ground_truth_rect, (dict, OrderedDict)):
                        init_data[0]['bbox'] = list(self.ground_truth_rect[self.object_ids[0]][0, :])
                    else:
                        init_data[0]['bbox'] = list(self.ground_truth_rect[0,:])

            if self.ground_truth_seg is not None:
                init_data[0]['mask'] = self.ground_truth_seg[0]

        return init_data

    def init_info(self):
        info = self.frame_info(frame_num=0)
        return info

    def frame_info(self, frame_num):
        info = self.object_init_data(frame_num=frame_num)
        
        return info

    def init_bbox(self, frame_num=0):
        return self.object_init_data(frame_num=frame_num).get('init_bbox')

    def init_mask(self, frame_num=0):
        return self.object_init_data(frame_num=frame_num).get('init_mask')

    def get_info(self, keys, frame_num=None):
        info = dict()
        for k in keys:
            val = self.get(k, frame_num=frame_num)
            if val is not None:
                info[k] = val
        return info

    def object_init_data(self, frame_num=None) -> dict:
        if frame_num is None:
            frame_num = 0
        if frame_num not in self.init_data:
            return dict()

        init_data = dict()
        for key, val in self.init_data[frame_num].items():
            if val is None:
                continue
            init_data['init_'+key] = val

        if 'init_mask' in init_data and init_data['init_mask'] is not None:
            anno = imread_indexed(init_data['init_mask'])
           
            if not self.multiobj_mode and self.object_ids is not None:
                assert len(self.object_ids) == 1
                anno = (anno == int(self.object_ids[0])).astype(np.uint8)
            init_data['init_mask'] = anno

        if self.object_ids is not None:
            init_data['object_ids'] = self.object_ids
            init_data['sequence_object_ids'] = self.object_ids

        return init_data

    def target_class(self, frame_num=None):
        return self.object_class

    def get(self, name, frame_num=None):
        return getattr(self, name)(frame_num)

    def __repr__(self):
        return "{self.__class__.__name__} {self.name}, length={len} frames".format(self=self, len=len(self.frames))

def sample_patch(im: torch.Tensor, pos: torch.Tensor, sample_sz: torch.Tensor, output_sz: torch.Tensor = None,
                 mode: str = 'replicate', max_scale_change=None, is_mask=False):
    """Sample an image patch.

    args:
        im: Image
        pos: center position of crop
        sample_sz: size to crop
        output_sz: size to resize to
        mode: how to treat image borders: 'replicate' (default), 'inside' or 'inside_major'
        max_scale_change: maximum allowed scale change when using 'inside' and 'inside_major' mode
    """

    # if mode not in ['replicate', 'inside']:
    #     raise ValueError('Unknown border mode \'{}\'.'.format(mode))

    # copy and convert
    posl = pos.long().clone()

    pad_mode = mode

    # Get new sample size if forced inside the image
    if mode == 'inside' or mode == 'inside_major':
        pad_mode = 'replicate'
        im_sz = torch.Tensor([im.shape[2], im.shape[3]])
        shrink_factor = (sample_sz.float() / im_sz)
        if mode == 'inside':
            shrink_factor = shrink_factor.max()
        elif mode == 'inside_major':
            shrink_factor = shrink_factor.min()
        shrink_factor.clamp_(min=1, max=max_scale_change)
        sample_sz = (sample_sz.float() / shrink_factor).long()

    # Compute pre-downsampling factor
    if output_sz is not None:
        resize_factor = torch.min(sample_sz.float() / output_sz.float()).item()
        df = int(max(int(resize_factor - 0.1), 1))
    else:
        df = int(1)

    sz = sample_sz.float() / df     # new size

    # Do downsampling
    if df > 1:
        os = posl % df              # offset
        posl = (posl - os) // df     # new position
        im2 = im[..., os[0].item()::df, os[1].item()::df]   # downsample
    else:
        im2 = im

    # compute size to crop
    szl = torch.max(sz.round(), torch.Tensor([2])).long()

    # Extract top and bottom coordinates
    tl = posl - (szl - 1) // 2
    br = posl + szl//2 + 1

    # Shift the crop to inside
    if mode == 'inside' or mode == 'inside_major':
        im2_sz = torch.LongTensor([im2.shape[2], im2.shape[3]])
        shift = (-tl).clamp(0) - (br - im2_sz).clamp(0)
        tl += shift
        br += shift

        outside = ((-tl).clamp(0) + (br - im2_sz).clamp(0)) // 2
        shift = (-tl - outside) * (outside > 0).long()
        tl += shift
        br += shift

        # Get image patch
        # im_patch = im2[...,tl[0].item():br[0].item(),tl[1].item():br[1].item()]

    # Get image patch
    if not is_mask:
        im_patch = F.pad(im2, (-tl[1].item(), br[1].item() - im2.shape[3], -tl[0].item(), br[0].item() - im2.shape[2]), pad_mode)
    else:
        im_patch = F.pad(im2, (-tl[1].item(), br[1].item() - im2.shape[3], -tl[0].item(), br[0].item() - im2.shape[2]))

    # Get image coordinates
    patch_coord = df * torch.cat((tl, br)).view(1,4)

    if output_sz is None or (im_patch.shape[-2] == output_sz[0] and im_patch.shape[-1] == output_sz[1]):
        return im_patch.clone(), patch_coord

    # Resample
    if not is_mask:
        im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='bilinear')
    else:
        im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='nearest')
    return im_patch, patch_coord

def extract_patches(im: torch.Tensor, pos: torch.Tensor, scales, image_sz: torch.Tensor):
        """Extract features.
        args:
            im: Image.
            pos: Center position for extraction.
            scales: Image scales to extract features from.
            image_sz: Size to resize the image samples to before extraction.
        """
        if isinstance(scales, (int, float)):
            scales = [scales]

        # Get image patches
        patch_iter, coord_iter = zip(*(sample_patch(im, pos, s*image_sz, image_sz, mode='replicate',
                                                    max_scale_change=None) for s in scales))
        im_patches = torch.cat(list(patch_iter))

        patch_coords = torch.cat(list(coord_iter))
        return torch.Tensor(im_patches),patch_coords

def extract_patches_transformed(im: torch.Tensor, pos: torch.Tensor, scales, image_sz: torch.Tensor, transforms):
        """Extract features.
        args:
            im: Image.
            pos: Center position for extraction.
            scales: Image scales to extract features from.
            image_sz: Size to resize the image samples to before extraction.
        """
        # Get image patches
        im_patch,patch_coords = sample_patch(im, pos, scales*image_sz, image_sz)

        # Apply transforms
        im_patches = torch.cat([T(im_patch) for T in transforms])
        return torch.Tensor(im_patches),patch_coords

def initialize_get_parameters():
        """Get parameters."""
        param_module = importlib.import_module('pytracking.parameter.{}.{}'.format('eco', 'initialize_default'))
        params = param_module.parameters()
        return params
 
def preprocess(image, info: dict, sample_pos, sample_scales, img_sample_sz) -> dict:
    im = numpy_to_torch(image)
    im_patches, patch_coords = extract_patches(im, sample_pos, sample_scales, img_sample_sz)

    #Normalize patches
    im_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    im_std = torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    im_patches = im_patches / 255
    im_patches -= im_mean
    im_patches /= im_std

    return im_patches, patch_coords

def initial_preprocess(image, bbox) -> dict:
    params = initialize_get_parameters()

    transforms = [augmentation.Identity()]
    if 'shift' in params.augmentation:
        transforms.extend([augmentation.Translation(shift) for shift in params.augmentation['shift']])
    if 'fliplr' in params.augmentation and params.augmentation['fliplr']:
        transforms.append(augmentation.FlipHorizontal())
    if 'rotate' in params.augmentation:
        transforms.extend([augmentation.Rotate(angle) for angle in params.augmentation['rotate']])
    if 'blur' in params.augmentation:
        transforms.extend([augmentation.Blur(sigma) for sigma in params.augmentation['blur']])

    im = numpy_to_torch(image)
    state = bbox
    pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
    sample_pos = pos#.round()
    target_sz = torch.Tensor([state[3], state[2]])


    # Set search area
    target_scale = 1.0
    search_area = torch.prod(target_sz * 4.5).item()
    if search_area > 250**2:
        target_scale =  math.sqrt(search_area / 250**2)
    elif search_area < 200**2:
        target_scale =  math.sqrt(search_area / 200**2)

    scale_factors = 1.02**torch.arange(-2, 3).float() 
    sample_scales = target_scale * scale_factors

    base_target_sz = target_sz / target_scale
    img_sample_sz = torch.round(torch.sqrt(torch.prod(base_target_sz * 4.5))) * torch.ones(2)
    #print("[INFO][app_kf.py][initial_preprocess] img_sample_sz: ", img_sample_sz)

    feature_stride = [8, 8]
    feat_max_stride = max(feature_stride)
    #print("[INFO][app_kf.py][initial_preprocess] feat_max_stide: ", feat_max_stride)

    img_sample_sz += feat_max_stride - img_sample_sz % (2 * feat_max_stride)


    im_patches, patch_coords = extract_patches_transformed(im, pos, target_scale, img_sample_sz, transforms)

    #Normalize patches
    im_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    im_std = torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    im_patches = im_patches / 255
    im_patches -= im_mean
    im_patches /= im_std

    return state, im_patches, patch_coords, sample_pos, target_scale, img_sample_sz, target_sz 

def get_parameters(features):
        """Get parameters."""
        param_module = importlib.import_module('pytracking.parameter.{}.{}'.format('eco', 'default'))
        params = param_module.parameters(features)
        return params
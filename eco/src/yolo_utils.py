import numpy as np
import cv2


# Preprocessing
def preprocessing(image, dims):
    """ Turns image from original dimension to desired dimension using padding """
    #Get original dimensions
    oh, ow, _ = image.shape
    h, w = dims

    #Compute new dimensions
    scale = min(w/ow, h/oh)
    nh = int(oh*scale)
    nw = int(ow*scale)

    #Compute padding to achieve dimensions
    h_padding = int((h - nh) / 2)
    w_padding = int((w - nw)  / 2)

    #Rescale 
    image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_LINEAR)

    #Pad
    image = cv2.copyMakeBorder(image, h_padding, h_padding, w_padding, w_padding, cv2.BORDER_CONSTANT, value=(128,128,128))

    #Preprocess
    image_data = np.array(image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)

    return image_data

# Yolo postprocessing utils
def get_class(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)
    
def _get_feats(feats, anchors, num_classes, input_shape):
    num_anchors = len(anchors)
    anchors_tensor = np.reshape(np.array(anchors, dtype=np.float32), [1, 1, 1, num_anchors, 2])
    grid_size = np.shape(feats)[1:3]
    nu = num_classes + 5
    predictions = np.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, nu])
    grid_y = np.tile(np.reshape(np.arange(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
    grid_x = np.tile(np.reshape(np.arange(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
    grid = np.concatenate([grid_x, grid_y], axis = -1)
    grid = np.array(grid, dtype=np.float32)

    box_xy = (1/(1+np.exp(-predictions[..., :2])) + grid) / np.array(grid_size[::-1], dtype=np.float32)
    box_wh = np.exp(predictions[..., 2:4]) * anchors_tensor / np.array(input_shape[::-1], dtype=np.float32)
    box_confidence = 1/(1+np.exp(-predictions[..., 4:5]))
    box_class_probs = 1/(1+np.exp(-predictions[..., 5:]))
    return box_xy, box_wh, box_confidence, box_class_probs
	
def correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape, dtype = np.float32)
    image_shape = np.array(image_shape, dtype = np.float32)
    new_shape = np.around(image_shape * np.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([
        box_mins[..., 0:1],
        box_mins[..., 1:2],
        box_maxes[..., 0:1],
        box_maxes[..., 1:2]
    ], axis = -1)
    boxes *= np.concatenate([image_shape, image_shape], axis = -1)
    return boxes
	
def boxes_and_scores(feats, anchors, classes_num, input_shape, image_shape):
    box_xy, box_wh, box_confidence, box_class_probs = _get_feats(feats, anchors, classes_num, input_shape)
    boxes = correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = np.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = np.reshape(box_scores, [-1, classes_num])
    return boxes, box_scores	

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2-x1+1)*(y2-y1+1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 1)
        h1 = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= 0.55)[0]  # threshold
        order = order[inds + 1]

    return keep

'''Model post-processing'''
def yolo_eval(yolo_outputs, image_shape,classes_path, anchors_path, max_boxes = 20):
    score_thresh = 0.30
    nms_thresh = 0.30
    class_names = get_class(classes_path)
    anchors     = get_anchors(anchors_path)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    boxes = []
    box_scores = []
    
    input_shape = np.shape(yolo_outputs[0])[1 : 3]
    #print(input_shape)
    input_shape = np.array(input_shape)*32
    #print(input_shape)
    
    for i in range(len(yolo_outputs)):
        _boxes, _box_scores = boxes_and_scores(yolo_outputs[i], anchors[anchor_mask[i]], len(class_names), input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = np.concatenate(boxes, axis = 0)
    box_scores = np.concatenate(box_scores, axis = 0)
    
    mask = box_scores >= score_thresh
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(len(class_names)):
        class_boxes_np = boxes[mask[:, c]]
        class_box_scores_np = box_scores[:, c]
        class_box_scores_np = class_box_scores_np[mask[:, c]]
        nms_index_np = nms_boxes(class_boxes_np, class_box_scores_np) 
        class_boxes_np = class_boxes_np[nms_index_np]
        class_box_scores_np = class_box_scores_np[nms_index_np]
        classes_np = np.ones_like(class_box_scores_np, dtype = np.int32) * c
        boxes_.append(class_boxes_np)
        scores_.append(class_box_scores_np)
        classes_.append(classes_np)
    boxes_ = np.concatenate(boxes_, axis = 0)
    scores_ = np.concatenate(scores_, axis = 0)
    classes_ = np.concatenate(classes_, axis = 0)
    
    return boxes_, scores_, classes_


def runYoloDPU(dpuRunner, image):

    #Yolo tracker info
    classes_path = "./yolo_model_data/voc_classes.txt"
    anchors_path = "./yolo_model_data/yolo_anchors.txt"
    
    # Reshape image to 416x416 for yolo
    image_size = image.shape[:2]
    img = [np.array(preprocessing(image,(416,416)), dtype=np.float32)]

    #Get DPU info
    inputTensors = dpuRunner.get_input_tensors()
    outputTensors = dpuRunner.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim1 = tuple(outputTensors[0].dims)
    output_ndim2 = tuple(outputTensors[1].dims)
    output_ndim3 = tuple(outputTensors[2].dims)

    #print("\tInput tensors: ", inputTensors)
    #print("\tInput tensor dimensions: ", input_ndim)
    #print("\tOutput tensors: ", outputTensors)
    #print("\tOutput tensor 1 dimensions: ", output_ndim1)
    #print("\tOutput tensor 2 dimensions: ", output_ndim2)
    #print("\tOutput tensor 3 dimensions: ", output_ndim3)

    #Prepare input and output data
    inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]
    outputData= [np.empty(output_ndim1, dtype=np.float32, order="C"),
                 np.empty(output_ndim2, dtype=np.float32, order="C"),
                 np.empty(output_ndim3, dtype=np.float32, order="C")]
    

    #Load image into inputData
    imageRun = inputData[0]
    imageRun[0, ...] = img[0].reshape(input_ndim[1:])

    # Run DPU
    job_id = dpuRunner.execute_async(inputData, outputData)
    dpuRunner.wait(job_id)

    # Extract yolo outputs
    yolo_outputs = [outputData[0], outputData[1], outputData[2]]

    #Post-process outputs
    out_boxes, _, _ = yolo_eval(yolo_outputs, image_size, classes_path, anchors_path)
    x, y, x2, y2 = out_boxes[0]
    w = x2 - x
    h = y2 - y
    bbox = [x,y,w,h]

    return bbox
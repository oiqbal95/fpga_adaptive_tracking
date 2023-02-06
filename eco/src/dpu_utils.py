from typing import List
import numpy as np
import xir
import vart
import torch
from src.tensorlist import TensorList

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def runDPU(dpuRunner, im_patches):#id,start,dpu,img):

    #Get DPU info
    inputTensors = dpuRunner.get_input_tensors()
    outputTensors = dpuRunner.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim1 = tuple(outputTensors[0].dims)
    output_ndim2 = tuple(outputTensors[1].dims)
    output_ndim3 = tuple(outputTensors[2].dims)
    output_ndim4 = tuple(outputTensors[3].dims)

    #print("\tInput tensors: ", inputTensors)
    #print("\tInput tensor dimensions: ", input_ndim)
    #print("\tOutput tensors: ", outputTensors)
    #print("\tOutput tensor 1 dimensions: ", output_ndim1)
    #print("\tOutput tensor 2 dimensions: ", output_ndim2)
    #print("\tOutput tensor 3 dimensions: ", output_ndim3)
    #print("\tOutput tensor 4 dimensions: ", output_ndim4

    #Prepare input and output data
    inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]
    outputData= [np.empty(output_ndim1, dtype=np.float32, order="C"),
                 np.empty(output_ndim2, dtype=np.float32, order="C"),
                 np.empty(output_ndim3, dtype=np.float32, order="C"),
                 np.empty(output_ndim4, dtype=np.float32, order="C")]
    
     #Initialize empty containers
    features0 = np.zeros((len(im_patches),output_ndim1[3],output_ndim1[1],output_ndim1[2]))
    features1 = np.zeros((len(im_patches),output_ndim2[3],output_ndim2[1],output_ndim2[2]))
    features2 = np.zeros((len(im_patches),output_ndim3[3],output_ndim3[1],output_ndim3[2]))
    features3 = np.zeros((len(im_patches),output_ndim4[3],output_ndim4[1],output_ndim4[2]))

    #Run DPU once for every patch
    for pi in range(len(im_patches)):
        imageRun = inputData[0]
        imageRun[0, ...] = np.transpose(im_patches[pi], (1,2,0))

        job_id = dpuRunner.execute_async(inputData, outputData)
        dpuRunner.wait(job_id)

        features0[pi] = np.transpose(outputData[0], (0, 3, 1, 2))
        features1[pi] = np.transpose(outputData[1], (0, 3, 1, 2))
        features2[pi] = np.transpose(outputData[2], (0, 3, 1, 2))
        features3[pi] = np.transpose(outputData[3], (0, 3, 1, 2))

    #Convert to numpy array
    features0 = np.array(features0)
    features1 = np.array(features1)
    features2 = np.array(features2)
    features3 = np.array(features3)

    #Conver to tensor
    features0 = torch.from_numpy(features0)
    features1 = torch.from_numpy(features1)
    features2 = torch.from_numpy(features2)
    features3 = torch.from_numpy(features3)

    #Create tensorlist
    features_tensor = [features0,features1,features2,features3]
    features = TensorList(features_tensor)

    return features

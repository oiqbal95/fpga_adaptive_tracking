import numpy as np
import matplotlib.pyplot as plt
#from utils.utils import Box
class Box:
    def __init__(self):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.prob = float()
        self.label = ''



class KFilter():
    def __init__(self, delta_t, bbox=None, q=.1, r=.2, p=1000):
        """Initializes the filter with the:
            initial state estimate (state): initial measurement bounding box
                converted to state vector [x, y, vx, vy, w, h, vw, vh]
            initial covarience estimate (P): can be a rough estimate.
            state update matrix (A): contains the equations to go from current
                estimate to prediction.
            observation matrix (H): transformation matrix from the measurement
                space to the state space.
            process covarience (Q): uncertainty in the model (should be small value e.g. .0001)
            measurement covarience (R): uncertainty in tinit_m_boxhe measurement
            delta_t: time between measurements in seconds (used to detemine state
                update matrix)

            Vectors are lower case (like x) and matrices are upper case (like A)
        """
        if bbox is not None:
            self.state = np.array([bbox.x,bbox.y,0,0,bbox.w,bbox.h,0,0])
        else:
            self.state = np.zeros(8)

        self.P = np.eye(8) * p
        dt = delta_t
        self.A = np.array([[1,0,dt,0,0,0,0,0],
                           [0,1,0,dt,0,0,0,0],
                           [0,0,1,0,0,0,0,0],
                           [0,0,0,1,0,0,0,0],
                           [0,0,0,0,1,0,dt,0],
                           [0,0,0,0,0,1,0,dt],
                           [0,0,0,0,0,0,1,0],
                           [0,0,0,0,0,0,0,1]])

        self.H = np.array([[1,0,0,0,0,0,0,0],
                           [0,1,0,0,0,0,0,0],
                           [0,0,0,0,1,0,0,0],
                           [0,0,0,0,0,1,0,0]])
        self.Q = np.eye(8) * q
        self.R = np.eye(4) * r


    def update(self, bbox):
        """full kalman filter cycle:
                predict
                measure
                    innovation
                    covarience innovation
                update
                    kalman gain
                    state update
                    covarience update
            returns new current estimate bbox
        """
        self.predict()
        # measure
        z = np.array([bbox.x,bbox.y,bbox.w,bbox.h])
        # innovation (y = z - Hx)
        y = z - np.dot(self.H, self.state)
        # covarience innovation (S = HPH^t + R)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        # update
        # kalman gain (K = PHS^-1)
        inv_S = np.linalg.inv(S)
        K = np.dot(self.P, np.dot(self.H.T, inv_S))
        # state update (state = state + Ky)
        self.state += np.dot(K, y)
        # covarience update (P = (I - KH)P)
        I = np.eye(self.P[0].size)
        comp_K = I - np.dot(K, self.H)
        self.P = np.dot(comp_K, self.P)

        est_box = _state_to_bbox(self.state)
        return est_box

    def predict(self):
        """prediction only:
                state extrapolation
                covarience extrapolation
        """
        # state update (state = A*state)
        self.state = np.dot(self.A, self.state)
        # covarience update (P = APA^t + Q)
        self.P = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q
        bbox = _state_to_bbox(self.state)
        return bbox


def _state_to_bbox(state):
    box = Box()
    box.x = int(state[0])
    box.y = int(state[1])
    box.w = int(state[4])
    box.h = int(state[5])
    return box

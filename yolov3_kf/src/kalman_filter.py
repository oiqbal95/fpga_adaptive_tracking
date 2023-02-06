import numpy as np

#Kalman Filter class definition
class KFilter():
    def __init__(self, delta_t, bbox=None, q=.01, r=.01, p=.01):
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
        
#Bounding box class definition
class bbox:
    def __init__(self, x1=0, y1=0 , x2=0, y2=0):
        self.x = x1
        self.y = y1
        self.w = x2 - x1
        self.h = y2 - y1

#State to bounding box function
def _state_to_bbox(state):
    box = bbox()
    box.x = int(state[0])
    box.y = int(state[1])
    box.w = int(state[4])
    box.h = int(state[5])
    return box

#Unpack bounding box function
def unpack_bbox(coords, scores):
    max_score = np.max(scores)
    max_index = np.where(scores == max_score)
    x1, y1, x2, y2 = coords[max_index]

    return x1, y1, x2, y2


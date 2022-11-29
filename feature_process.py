import numpy as np
import cv2
# import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from scipy.fft import fft, fftfreq
from WT import transform,wavelets

### feature from raw landmarks ##########################################################################
def count_dist(raw, sel=[[0,1],[0,2],[1,3],[2,3],[3,4],[3,5],[4,6],[5,6]]):
    '''
    Distances for landmarks in each frame
    '''
    distances = []
    for [i,j] in sel:
        p1 = raw[:,2*i:2*i+2]
        p2 = raw[:,2*j:2*j+2]
        distances.append(np.linalg.norm(p2-p1,axis=1))
    return np.array(distances).T

def count_angle(raw, sel=[[0,3,6]]):
    '''
    count angles for 5 points dlc (raw) in each frame
    sel: angle of selected points (example:[[0,1,2],[1,2,3]] => angle of points)
    '''
    angle = []
    for p1,p2,p3 in sel:
        v1 = raw[:,2*p1:2*p1+2]-raw[:,2*p2:2*p2+2]
        v2 = raw[:,2*p3:2*p3+2]-raw[:,2*p2:2*p2+2]
        angle.append(abs(np.arctan2(v1[:,0],v1[:,1])-np.arctan2(v2[:,0],v2[:,1])))
    return np.array(angle).T

def count_disp(raw, step=1, threshold=None):
    '''
    count distances and vectors(directions) between frames of deeplabcut data
    threshold: distance set 0 for value under threshold
    dlc_raw shape: N*(2*landmarks) 
    distances shape: (N-1)*landmarks
    vectors shape: (N-1)*landmarks*2
    directions shape: (N-1)*landmarks
    '''
    distances = []
    # for each two frames
    for i in range(0,len(raw)-step,step):
        distance = []
        #vector = []
        #direction = []
        # for each landmark
        for j in range(int(len(raw[0])/2)):
            p1=raw[i,2*j:2*j+2]
            p2=raw[i+step,2*j:2*j+2]
            vec = p2-p1
            #direction.append(np.arctan2(vec[1],vec[0]))
            #vector.append(vec)
            dis = np.linalg.norm(vec)
            if threshold and dis<threshold:
                distance.append(0)
            else:
                distance.append(dis)
        #vectors.append(vector)
        distances.append(distance)
        #directions.append(direction)
    return np.array(distances)

def feature_normalize(feat, normalize_range=(0,1), sc='minmax'):
    if sc == 'minmax':
        scaler = MinMaxScaler(feature_range=normalize_range)
        scaler.fit(feat)
        feat = scaler.transform(feat)
    if sc == 'std':
        scaler = StandardScaler()
        scaler.fit(feat)
        feat = scaler.transform(feat)
    return feat
###############################################################################################

### feature from video #############################################################################
def count_optflow_feat(vid_path, mask=True, stop=None, white_back=False):
    '''
    count optical flow Fx,Fy for all points in video
    mask: remove noise flow by mice roi mask (frame==0)
    '''
    cap = cv2.VideoCapture(vid_path)
    flows = []
    ret, frame = cap.read()
    prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prvs = cv2.equalizeHist(prvs)
    if white_back:
        prvs[np.where(prvs==0)]=255
    i=0
    while(1):
        ret, frame = cap.read()
        if not ret:
            break
        if stop and i>=stop:
            break
        next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # next = cv2.equalizeHist(next)
        if white_back:
            next[np.where(next==0)]=255
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 5, 3, 5, 5, 0)
        if mask:
            flow[np.where(next==0)]=0
        flows.append(flow)
        i+=1
    return np.array(flows)

def count_mice_area(vid_path, stop=None):
    '''
    detect mice area in each frame to observe stretch and clinge
    '''
    cap = cv2.VideoCapture(vid_path)
    areas = []
    i=0
    while(cap.isOpened()):
        ret,frame = cap.read()
        if not ret:
            break
        if stop and i>=stop:
            break
        areas.append(len(np.where(frame==0)[0]))
    return np.array(areas)
#################################################################################################

### segment feature #############################################################################
def seg_statistic(feat, count_types=['avg'], window=10, step=10):
    '''
    feature statistic feature from each window of segment
    '''
    msk_feat = []
    for i in range(int(len(feat)/step)-1):
        msk = feat[i*step:i*step+window]
        newfeat = []
        if 'max' in count_types:
            newfeat.append(np.max(msk, axis=0))
        if 'min' in count_types:
            newfeat.append(np.min(msk, axis=0))
        if 'avg' in count_types:
            newfeat.append(np.mean(msk, axis=0))
        if 'std' in count_types:
            newfeat.append(np.std(msk, axis=0))
        if 'sum' in count_types:
            newfeat.append(np.sum(msk, axis=0))

        newfeat = np.concatenate(newfeat)
        msk_feat.append(newfeat)
    return np.array(msk_feat)

def generate_tmpfeat(feat, window=10, step=10):
    '''
    segment of feature
    '''
    tmp_feat = []
    for i in range(int(len(feat)/step)-1):
        tmp_feat.append(feat[i*step:i*step+window])
    return np.array(tmp_feat)


### wave feature #############################################################################
def cwt_signal(feat, window=10, step=10, flat=True):
    '''
    count the continious wavelet transform of all signal
    min_len : shortest signal (longest wavelet)
    sample_len : taking samples through frames to keep same length for all result
    '''
    powers = []
    for i in range(int(len(feat)/step)-1):
        x = feat[i*step:i*step+window]
        t = np.arange(len(x))
        dt = 1              # sampling frequency
        dj = 0.2             # scale distribution parameter
        wavelet = wavelets.Morlet()
        wa = transform.WaveletTransformTorch(dt, dj, wavelet, cuda=False)

        #cwt = wa.cwt(x) # Eular format
        if flat:
            powers.append(wa.power(x).flatten())
        else:
            shp = np.prod(wa.power(x).shape[1:])
            powers.append(wa.power(x).reshape(window,shp))

    return np.array(powers)

def fft_signal(feat, window=10,step=1, flat=True):
    msk_feat = []
    for i in range(int(len(feat)/window)):
        msk = feat[i*window:i*window+window]
        freq = fft(msk.T)
        if flat:
            freq=freq.flatten()
        msk_feat.append(freq)
    return np.array(msk_feat)
##################################################################################################

# def combine_feat(dlc_raw, sel_dist=[[0,1],[0,2],[1,3],[2,3],[3,4],[3,5],[4,6],[5,6]], sel_ang=[[0,3,6]], 
#                 sel_coord=[0,1,2,3,4,5,8,9,10,11,12,13], normalize=(1,5), index=True):
#     '''
#     return concatenation of distance and angle
#     '''
#     raw = dlc_raw.raw
#     frame_index = dlc_raw.frame_index

#     feat = np.zeros([len(raw), 0])
#     if sel_dist:
#         a = count_dist(raw, sel_dist)
#         feat = np.hstack([feat,a])
#     if sel_ang:
#         a = count_angle(raw, sel_ang)
#         feat = np.hstack([feat,a])
#     if sel_coord:
#         a = raw[:,sel_coord]
#         feat = np.hstack([feat,a])
    
#     if normalize:
#         scaler = MinMaxScaler(feature_range=normalize)
#         scaler.fit(feat)
#         feat = scaler.transform(feat)
#     # if dim_red:
#     #     pca = PCA(n_components=dim_red)
#     #     feat = pca.fit_transform(feat)
#     if index:
#         feat = np.hstack([frame_index, feat])
#     return feat


# def align_image(img, fixpt, rotatept, midx=200, midy=200, rotateTo=0):
#     '''
#     move mice to center and align the direction
#     convert image
#     '''
#     dx = midx-fixpt[0]
#     dy = midy-fixpt[1]
#     H = np.float32([[1,0,dx],[0,1,dy]])
#     move = cv2.warpAffine(img,H, (img.shape[1],img.shape[0]))
#     d_angle = (rotateTo-np.arctan2(rotatept[0],rotatept[1]))*180/np.pi
#     H = cv2.getRotationMatrix2D((midx,midy),d_angle,1)
#     move = cv2.warpAffine(move,H, (frame.shape[1],frame.shape[0]))
#     return move

# def align_point(coord, fixpt, rotatept, midx=200, midy=200, rotateTo=0):
#     '''
#     move mice to center and align the direction
#     convert landmarks coordinates
#     '''
#     newcoord = coord.copy()
#     dx = midx-fixpt[0]
#     dy = midy-fixpt[1]
#     newcoord[:,0] = newcoord[:,0]+dx
#     newcoord[:,1] = newcoord[:,1]+dy
#     d_angle = (rotateTo-np.arctan2(rotatept[0],rotatept[1]))*180/np.pi
#     H = cv2.getRotationMatrix2D((midx,midy),d_angle,1)
#     A = H[:,0:2]
#     B = H[:,2]
#     newcoord = np.matmul(newcoord,A.T)
#     newcoord[:,0] = newcoord[:,0]+B[0]
#     newcoord[:,1] = newcoord[:,1]+B[1]
#     newcoord = np.int32(newcoord)
#     return newcoord

# def align_all(raw_wrap, wrap=False, midx=200, midy=200, rotateTo=0, fixpt_index=3, rotatept_index=6):
#     '''
#     run align point for all instance in dlc raw_wrap
#     wrap : True-return wrap, False return raw
#     '''
#     newraw = raw_wrap.copy()
#     for i in range(len(newraw)):
#         newraw[i] = align_point(newraw[i], newraw[i,fixpt_index], newraw[i,rotatept_index], midx=200, midy=200, rotateTo=0)
#     if not wrap:
#         newraw = np.resize(newraw,(len(newraw),newraw.shape[1]*newraw.shape[2]))
#     return newraw

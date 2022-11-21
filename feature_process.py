import numpy as np
import cv2
# import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

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

### segment feature #############################################################################
def seg_statistic(feat, count_types=['avg'], window=10, step=1):
    '''
    feature statistic feature from each window of segment
    '''
    msk_feat = []
    for i in range(int(len(feat)/window)):
        msk = feat[i*window:i*window+window]
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
        # if 'fft' in count_types:
        #     freq = fft(feat.T)
        #     freq_feat = []
        #     for feat_freq in freq:
        #         newfeat.extend(feat_freq)

        newfeat = np.concatenate(newfeat)
        msk_feat.append(newfeat)
    return np.array(msk_feat)

def generate_tmpfeat(feat):
    '''
    segment of feature
    '''
    tmp_feat = []
    for i in range(int(len(feat)/10)):
        tmp_feat.append(feat[i*10:i*10+10])
    return np.array(tmp_feat)
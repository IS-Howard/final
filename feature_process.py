import numpy as np
import cv2
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import os
from scipy.fft import fft, fftfreq

class miceData:
    def __init__(self, dlc=None, vidc=None, vids=None, dep=None):
        if(dlc):
            self.dlcfile = dlc
            self.read_dlc()
        if(vidc):
            self.vidcfile = vidc
        if(vids):
            self.vidsfile = vids
        if(dep):
            self.depfile = dep
    
    ### DLC functions #############################################################################
    def read_dlc(self):
        if not os.path.isfile(self.dlcfile):
            print("no file")
            return
        raw = np.genfromtxt(self.dlcfile, delimiter=",",dtype=int)[3:]
        getcol = tuple(np.arange(len(raw[0]))[np.arange(len(raw[0]))%3!=0])
        self.dlc_index = np.expand_dims(raw[:,0], axis=1)
        self.dlc_raw = raw[:,getcol]
        #remove nan
        notnan = ~np.isnan(self.dlc_raw).any(axis=1)
        self.dlc_raw = self.dlc_raw[notnan]
        self.dlc_index = self.dlc_index[notnan]
    def dlc_wrap(self):
        return np.resize(self.dlc_raw,(len(self.dlc_raw),int(self.dlc_raw.shape[1]/2),2))
    ###############################################################################################
    
    ### config feature ##########################################################################
    def feature_config(self, sel_dist=[[0,1]], sel_ang=[[0,1,2]], 
                sel_coord=[], landmark_normalize=(0,1), include_index=False):
        self.sel_dist=sel_dist
        self.sel_ang=sel_ang
        self.sel_coord=sel_coord
        self.landmark_normalize=landmark_normalize
        self.include_index = include_index

        self.seg_window = 10

    ### landmark feature ##########################################################################
    def count_dist(self, sel=None):
        '''
        Distances for landmarks in each frame
        '''
        if not sel:
            sel = self.sel_dist
        distances = []
        for [i,j] in sel:
            p1 = self.dlc_raw[:,2*i:2*i+2]
            p2 = self.dlc_raw[:,2*j:2*j+2]
            distances.append(np.linalg.norm(p2-p1,axis=1))
        return np.array(distances).T

    def count_angle(self, sel=None):
        '''
        count angles for 5 points dlc (raw) in each frame
        sel: angle of selected points (example:[[0,1,2],[1,2,3]] => angle of points)
        '''
        if not sel:
            sel = self.sel_ang
        angle = []
        for p1,p2,p3 in sel:
            v1 = raw[:,2*p1:2*p1+2]-raw[:,2*p2:2*p2+2]
            v2 = raw[:,2*p3:2*p3+2]-raw[:,2*p2:2*p2+2]
            angle.append(abs(np.arctan2(v1[:,0],v1[:,1])-np.arctan2(v2[:,0],v2[:,1])))
        return np.array(angle).T
    
    def count_disp(self, step=1, threshold=None):
        '''
        count distances and vectors(directions) between frames of deeplabcut data
        threshold: distance set 0 for value under threshold
        dlc_raw shape: N*(2*landmarks) 
        distances shape: (N-1)*landmarks
        vectors shape: (N-1)*landmarks*2
        directions shape: (N-1)*landmarks
        '''
        data = self.dlc_raw

        distances = []
        # for each two frames
        for i in range(0,len(data)-step,step):
            distance = []
            #vector = []
            #direction = []
            # for each landmark
            for j in range(int(len(data[0])/2)):
                p1=data[i,2*j:2*j+2]
                p2=data[i+step,2*j:2*j+2]
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

    def count_landmark_feat(self):
        try:
            self.sel_dist
        except NameError:
            print('config undone')
            return
        raw = self.dlc_raw

        feat = np.zeros([len(raw), 0])
        if self.sel_dist:
            a = count_dist(raw, self.sel_dist)
            feat = np.hstack([feat,a])
        if self.sel_ang:
            a = count_angle(raw, self.sel_ang)
            feat = np.hstack([feat,a])
        if self.sel_coord:
            a = raw[:,self.sel_coord]
            feat = np.hstack([feat,a])
        
        if self.landmark_normalize:
            scaler = MinMaxScaler(feature_range=self.landmark_normalize)
            scaler.fit(feat)
            feat = scaler.transform(feat)
        if self.include_index:
            feat = np.hstack([self.frame_index, feat])

        self.landmark_feat = feat
    ###############################################################################################

    ### video feature #############################################################################
    def count_optflow_feat(self, mask=True, stop=None, white_back=False):
        '''
        count optical flow Fx,Fy for all points in video
        mask: remove noise flow by mice roi mask (frame==0)
        '''
        vid_path = self.vidsfile
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

    def mice_area(self):
        '''
        detect mice area in each frame to observe stretch and clinge
        '''
        vid_path = self.vidsfile
        cap = cv2.VideoCapture(vid_path)
        areas = []
        while(cap.isOpened()):
            ret,frame = cap.read()
            if not ret:
                break
            areas.append(len(np.where(frame==0)[0]))
        return np.array(areas)
    #################################################################################################
    
    ### segment feature #############################################################################
    def seg_statistic(self, feat, count_types=['avg'], window=None, step=None):
        if not window:
            window = self.seg_window
        msk_feat = []
        for i in range(int(len(feat)/window)):
            msk = feat[i*window:i*window+window]
            newfeat = []
            if 'max' in count_types:
                msk_feat.append(np.max(msk, axis=0))
            if 'min' in count_types:
                msk_feat.append(np.min(msk, axis=0))
            if 'avg' in count_types:
                msk_feat.append(np.mean(msk, axis=0))
            if 'std' in count_types:
                msk_feat.append(np.std(msk, axis=0))
            if 'fft' in count_types:
                freq = fft(feat.T)
                freq_feat = []
                for feat_freq in freq:
                    msk_feat.extend(feat_freq)

            newfeat = np.concatenate(newfeat)
            msk_feat.append(newfeat)
        return np.array(msk_feat)


class dlc:
    def __init__(self, dlc_path=None, raw=True, landmarknum=7):
        #dlc data
        self.raw = None
        self.raw_wrap = None
        self.frame_index = None

        if dlc_path:
            if raw: # dlc to coord
                self.read_dlc(dlc_path, landmarknum)
            else: # load coord directly
                self.read_dlc2(dlc_path)

    def read_dlc(self,dlc_path, landmarknum):
        if landmarknum == 7:
            getcol = (1,2,4,5,7,8,10,11,13,14,16,17,19,20)
        else:
            getcol = (1,2,4,5,7,8,10,11,13,14)
        raw = np.genfromtxt(dlc_path, delimiter=",")[3:,getcol]
        frame_index = np.genfromtxt(dlc_path, delimiter=",")[3:,0]
        #remove nan
        notnan = ~np.isnan(raw).any(axis=1)
        raw = raw[notnan]
        frame_index = frame_index[notnan]
        self.frame_index = np.expand_dims(frame_index, axis=1)
        #
        self.raw = raw.astype(int)
        # wrap 5 different landmark N*10 => N*5*2
        self.raw_wrap = np.resize(self.raw,(len(self.raw),int(self.raw.shape[1]/2),2))
        # way to unwarp c : np.resize(c,(len(c),c.shape[1]*c.shape[2]))

    def read_dlc2(self,dlc_path):
        raw = np.genfromtxt(dlc_path, delimiter=",")
        self.raw = raw.astype(int)
        self.raw_wrap = np.resize(self.raw,(len(self.raw),int(self.raw.shape[1]/2),2))
        frame_index = np.arange(len(self.raw))
        self.frame_index = np.expand_dims(frame_index, axis=1)
        

def mice_area(vid_path):
    '''
    detect mice area in each frame
    '''
    cap = cv2.VideoCapture(vid_path)
    areas = []
    while(cap.isOpened()):
        ret,frame = cap.read()
        if not ret:
            break
        areas.append(len(np.where(frame==0)[0]))
    return np.array(areas)


def count_dist(raw, sel=[[0,1],[0,2],[1,3],[2,3],[3,4],[3,5],[4,6],[5,6]]):
    '''
    count distances for points dlc (raw) in each frame
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

def combine_feat(dlc_raw, sel_dist=[[0,1],[0,2],[1,3],[2,3],[3,4],[3,5],[4,6],[5,6]], sel_ang=[[0,3,6]], 
                sel_coord=[0,1,2,3,4,5,8,9,10,11,12,13], normalize=(1,5), index=True):
    '''
    return concatenation of distance and angle
    '''
    raw = dlc_raw.raw
    frame_index = dlc_raw.frame_index

    feat = np.zeros([len(raw), 0])
    if sel_dist:
        a = count_dist(raw, sel_dist)
        feat = np.hstack([feat,a])
    if sel_ang:
        a = count_angle(raw, sel_ang)
        feat = np.hstack([feat,a])
    if sel_coord:
        a = raw[:,sel_coord]
        feat = np.hstack([feat,a])
    
    if normalize:
        scaler = MinMaxScaler(feature_range=normalize)
        scaler.fit(feat)
        feat = scaler.transform(feat)
    # if dim_red:
    #     pca = PCA(n_components=dim_red)
    #     feat = pca.fit_transform(feat)
    if index:
        feat = np.hstack([frame_index, feat])
    return feat

def generate_tmpfeat(feat):
    tmp_feat = []
    for i in range(int(len(feat)/10)):
        tmp_feat.append(feat[i*10:i*10+10])
    return np.array(tmp_feat)

def generate_mskfeat(feat):
    msk_feat = []
    for i in range(int(len(feat)/10)):
        msk = feat[i*10:i*10+10]
        mx = np.max(msk, axis=0)
        # mn = np.min(msk, axis=0)
        std = np.std(msk, axis=0)
        avg = np.mean(msk, axis=0)
        newfeat = np.concatenate([mx,std,avg])
        msk_feat.append(newfeat)
    return np.array(msk_feat)

def default_out(path):
    path = path.replace('\\','/')
    sp = path.rsplit('/',1)
    if sp[-1].find('basal')!=-1:
        bt = 'basal'
    else:
        bt = 'treat'
    return sp[0]+'/'+bt+'_feat.csv'

def align_image(img, fixpt, rotatept, midx=200, midy=200, rotateTo=0):
    '''
    move mice to center and align the direction
    convert image
    '''
    dx = midx-fixpt[0]
    dy = midy-fixpt[1]
    H = np.float32([[1,0,dx],[0,1,dy]])
    move = cv2.warpAffine(img,H, (img.shape[1],img.shape[0]))
    d_angle = (rotateTo-np.arctan2(rotatept[0],rotatept[1]))*180/np.pi
    H = cv2.getRotationMatrix2D((midx,midy),d_angle,1)
    move = cv2.warpAffine(move,H, (frame.shape[1],frame.shape[0]))
    return move

def align_point(coord, fixpt, rotatept, midx=200, midy=200, rotateTo=0):
    '''
    move mice to center and align the direction
    convert landmarks coordinates
    '''
    newcoord = coord.copy()
    dx = midx-fixpt[0]
    dy = midy-fixpt[1]
    newcoord[:,0] = newcoord[:,0]+dx
    newcoord[:,1] = newcoord[:,1]+dy
    d_angle = (rotateTo-np.arctan2(rotatept[0],rotatept[1]))*180/np.pi
    H = cv2.getRotationMatrix2D((midx,midy),d_angle,1)
    A = H[:,0:2]
    B = H[:,2]
    newcoord = np.matmul(newcoord,A.T)
    newcoord[:,0] = newcoord[:,0]+B[0]
    newcoord[:,1] = newcoord[:,1]+B[1]
    newcoord = np.int32(newcoord)
    return newcoord

def align_all(raw_wrap, wrap=False, midx=200, midy=200, rotateTo=0, fixpt_index=3, rotatept_index=6):
    '''
    run align point for all instance in dlc raw_wrap
    wrap : True-return wrap, False return raw
    '''
    newraw = raw_wrap.copy()
    for i in range(len(newraw)):
        newraw[i] = align_point(newraw[i], newraw[i,fixpt_index], newraw[i,rotatept_index], midx=200, midy=200, rotateTo=0)
    if not wrap:
        newraw = np.resize(newraw,(len(newraw),newraw.shape[1]*newraw.shape[2]))
    return newraw

if __name__=='__main__':
    '''
    input raw (deeplabcut) csv file and generate feature csv to same folder
    output default name : {input exp name}.csv
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='')
    parser.add_argument('--outpath', type=str, default='')
    opt = parser.parse_args()

    if not opt.inpath:
        print('input path error')
    else:
        path = opt.inpath
        dlc1 = dlc(path)
        feat = combine_feat(dlc1.raw)

        if not opt.outpath:
            outpath = default_out(path)
            np.savetxt(outpath,feat,delimiter=",")
        else:
            np.savetxt(opt.outpath,feat,delimiter=",")

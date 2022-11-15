from feature_process import *

class DataSet:
    '''
    Storing dataset to train/to test, root of related files, info of each single mice
    '''
    def __init__(self, dlc, vidc=None, vids=None, dep=None, specific=[]):
        self.specific = specific
        self.all_treatment = ['Capbasal','Cap','pH5.2basal','pH5.2','pH7.4basal','pH7.4']
        self.files = {}
        self.files['dlc'] = self.load_paths(dlc, True)
        self.files['vids'] = self.load_paths(vids)
        self.files['vidc'] = self.load_paths(vidc)
        self.files['dep'] = self.load_paths(dep)
        self.data_config()

    def load_paths(self, root, sav_treat=False):
        if not root:
            return []
        files = os.listdir(root)
        sav_files = []
        treatments = []
        names = []
        for file in files:
            sav = True
            for sp in self.specific:
                if file.find(sp)==-1:
                    sav = False
                    break
            if not sav:
                continue
            if sav_treat:
                treatment = file.split('-')[0]
                name = file.split('-')[1]
                if file.find('basal')!=-1:
                    treatments.append(treatment+'basal')
                else:
                    treatments.append(treatment)
                names.append(name)
            sav_files.append(root+'/'+file)
        if sav_treat:
            self.names = names
            self.treatments = treatments
        return sav_files

    def data_config(self):
        self.ind={}
        self.ind['basal'] = np.array([i for i, j in enumerate(self.treatments) if j.find('basal')!=-1])
        for t in self.all_treatment:
            self.ind[t] = np.array([i for i, j in enumerate(self.treatments) if j == t])
        # self.ind['Capbasal'] = np.array([i for i, j in enumerate(self.treatments) if j == 'Capbasal'])
        # self.ind['pH5.2basal'] = np.array([i for i, j in enumerate(self.treatments) if j == 'pH5.2basal'])
        # self.ind['pH7.4basal'] = np.array([i for i, j in enumerate(self.treatments) if j == 'pH7.4basal'])
        # self.ind['Cap'] = np.array([i for i, j in enumerate(self.treatments) if j == 'Cap'])
        # self.ind['pH5.2'] = np.array([i for i, j in enumerate(self.treatments) if j == 'pH5.2'])
        # self.ind['pH7.4'] = np.array([i for i, j in enumerate(self.treatments) if j == 'pH7.4'])
        print('basal:',len(self.ind['basal']),' ,pain:',len(self.ind['Cap']),' sng:',len(self.ind['pH5.2']),' pH7.4:',len(self.ind['pH7.4']))

    def sel_file(self, filetype='dlc', treatment='Cap'):
        if treatment == 'basal':
            return [self.files[filetype][i] for i, j in enumerate(self.treatments) if j.find('basal')!=-1]
        return [self.files[filetype][i] for i, j in enumerate(self.treatments) if j==treatment]

    def generate_feature(self):
        self.mice_feat = []
        for i in range(len(self.files['dlc'])):
            tmp = miceFeature(self.treatments[i], self.files['dlc'][i])#,self.files['vidc'][i],self.files['vids'][i],self.files['dep'][i])
            self.mice_feat.append(tmp)

    def generate_train_test(self):
        x_train = []
        y_train = []
        x_test = [] 
        y_test = []
        x_val = []
        y_val = []
        for t in self.all_treatment:
            inds = self.ind[t]
            for i in range(len(inds)-1): #last one for validate
                ind = inds[i]
                x_train.append(self.mice_feat[ind].x_train)
                y_train.append(self.mice_feat[ind].y_train)
                x_test.append(self.mice_feat[ind].x_test)
                y_test.append(self.mice_feat[ind].y_test)
            ind = inds[len(inds)-1]
            x_val.append(self.mice_feat[ind].feature)
            y_val.append(self.mice_feat[ind].label)
        self.x_train = np.concatenate(x_train)
        self.y_train = np.concatenate(y_train)
        self.x_test = np.concatenate(x_test)
        self.y_test = np.concatenate(y_test)
        self.x_val = np.concatenate(x_val)
        self.y_val = np.concatenate(y_val)




class miceFeature:
    '''
    Storing All data(file paths, landmarks, features ...) of single mice(file)
    '''
    def __init__(self, treatment, dlc=None, vidc=None, vids=None, dep=None):
        self.treatment = treatment
        if(dlc):
            self.dlcfile = dlc
            self.read_dlc()
        if(vidc):
            self.vidcfile = vidc
        if(vids):
            self.vidsfile = vids
        if(dep):
            self.depfile = dep

        self.count_feature()
        self.labeling()
        self.train_config(split=0.5, shuffle=True)
    
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
    
    ### generate feature ##########################################################################
    def count_feature(self):
        # config
        sel_dist=[[0,3],[3,6]]
        sel_ang=[[1,3,2]]
        sel_coord=[]
        normalize_range=(0,1)
        include_index = False
        seg_window = 10

        # frame feature pre
        dist = count_dist(self.dlc_raw, sel_dist)[1:]
        ang = count_angle(self.dlc_raw, sel_ang)[1:]
        disp = count_disp(self.dlc_raw, step=1, threshold=None)
        # frame feature
        feat = dist
        feat = np.hstack([feat, ang])
        feat = np.hstack([feat, disp[:,0:1]])
        # segment feature
        seg = abs(fft_signal(feat, window=seg_window, flat=True))
        tmp = np.hstack([disp, ang])
        seg = np.hstack([seg, seg_statistic(tmp, count_types=['avg'], window=10, step=1)])
        seg = np.hstack([seg, seg_statistic(dist, count_types=['sum'], window=10, step=1)])
        # normalize
        seg = feature_normalize(seg, normalize_range=normalize_range)

        self.feature = seg

    ### train test config ##########################################################################
    def labeling(self):
        # pain:1 sng:2 health:0
        labels = np.zeros_like(self.feature[:,0], dtype=int)
        if self.treatment == 'pH5.2':
            labels[:] = 2
        elif self.treatment == 'pH7.4' or self.treatment.find('basal')!=-1:
            labels[:] = 0
        elif self.treatment == 'Cap':
            labels[:] = 1
        self.label=labels
         
    def train_config(self, split=0.5, shuffle=True):
        # shuffle
        ind = np.arange(len(self.feature))
        np.random.shuffle(ind)
        self.shuffle = ind
        # split
        feat = self.feature
        label = self.label
        if shuffle:
            feat = self.feature[ind]
            label = self.label[ind]
        if split==0:
            self.x_train = feat
            self.y_train = label
            self.x_test = []
            self.y_test = []
        else:
            # split : training portion
            sp = int(len(label)*split)
            self.x_train = feat[:sp,:]
            self.y_train = label[:sp]
            self.x_test = feat[sp:,:]
            self.y_test = label[sp:]
                

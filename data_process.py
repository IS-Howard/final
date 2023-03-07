from feature_process import *
from pose_cluster import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_curve, RocCurveDisplay
from sklearn.inspection import permutation_importance
import tensorflow as tf
import csv

def train_balance(x,y):
    health = np.where(y==0)[0]
    pain = np.where(y==1)[0]
    sng = np.where(y==2)[0]
    mins = min([len(health),len(pain),len(sng)])
    health = np.random.choice(health, mins, replace=False)
    pain = np.random.choice(pain, mins, replace=False)
    sng = np.random.choice(sng, mins, replace=False)
    newidx = np.concatenate([health,pain,sng])
    return x[newidx],y[newidx]

class DataSet:
    '''
    Storing dataset to train/to test, root of related files, info of each single mice
    '''
    def __init__(self, dlc, bsoid=None, vidc=None, vids=None, dep=None, specific=[]):
        self.specific = specific
        self.all_treatment = ['Capbasal','Cap','pH5.2basal','pH5.2','pH7.4basal','pH7.4',
                                'pH5.2ASIC3KObasal','pH5.2ASIC3KO','CapTV1KObasal','CapTV1KO']
        self.files = {}
        self.files['dlc'] = self.load_paths(dlc, True)
        self.files['bsoid'] = self.load_paths(bsoid)
        self.files['vids'] = self.load_paths(vids)
        self.files['vidc'] = self.load_paths(vidc)
        self.files['dep'] = self.load_paths(dep)
        self.data_config()
        self.mclf=None

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
        print('basal:',len(self.ind['basal']),' ,pain:',len(self.ind['Cap']),
                ' sng:',len(self.ind['pH5.2']),' pH7.4:',len(self.ind['pH7.4']),
                ' sngKO:',len(self.ind['pH5.2ASIC3KO']),' CapKO:',len(self.ind['CapTV1KO']))

    def sel_file(self, filetype='dlc', treatment='Cap'):
        if treatment == 'basal':
            return [self.files[filetype][i] for i, j in enumerate(self.treatments) if j.find('basal')!=-1]
        return [self.files[filetype][i] for i, j in enumerate(self.treatments) if j==treatment]

    def sel_feat(self, treatment='all'):
        if treatment == 'basal':
            return [self.mice_feat[i] for i, j in enumerate(self.treatments) if j.find('basal')!=-1]
        return [self.mice_feat[i] for i, j in enumerate(self.treatments) if j==treatment]
    
    def sel_data(self, treatment='all', sel_type='x_train'):
        if treatment == 'basal':
            return [self.data[sel_type][i] for i, j in enumerate(self.treatments) if j.find('basal')!=-1]
        return [self.data[sel_type][i] for i, j in enumerate(self.treatments) if j==treatment]
    
    def generate_feature(self, feat_type='frame'):
        self.mice_feat = []
        for i in range(len(self.files['dlc'])):
            if feat_type[:-1] == 'bs':
                tmp = miceFeature(self.treatments[i], bsoid=self.files['bsoid'][i], feat_type=feat_type)
            else:
                tmp = miceFeature(self.treatments[i], self.files['dlc'][i], feat_type=feat_type)#,self.files['vidc'][i],self.files['vids'][i],self.files['dep'][i])
            self.mice_feat.append(tmp)

    def generate_train_test(self, split=0.5, motion_del=False, k=1):
        '''
        validation setting as last k-th of each three treatment mice
        '''
        # config for mice_feat
        for miceF in self.mice_feat:
            if self.mclf:
                miceF.labeling(self.mclf,self.motion_score)
            else:
                miceF.labeling()
            miceF.train_config(split=split, motion_del=motion_del)

        # start
        all_sets = ['x_train','y_train','x_test','y_test','x_val','y_val']
        self.data = {}
        for s in all_sets:
            self.data[s] = []

        for t in self.all_treatment:
            inds = self.ind[t]
            k = k%len(inds)
            for i in range(len(inds)):
                if i==len(inds)-k-1:
                    continue
                ind = inds[i]
                self.data['x_train'].append(self.mice_feat[ind].x_train)
                self.data['y_train'].append(self.mice_feat[ind].y_train)
                self.data['x_test'].append(self.mice_feat[ind].x_test)
                self.data['y_test'].append(self.mice_feat[ind].y_test)
            ind = inds[len(inds)-k-1]
            self.data['x_val'].append(self.mice_feat[ind].feature)
            self.data['y_val'].append(self.mice_feat[ind].label)

    def pose_cls(self, sel=['random'], sel_num=20, embed=False, k=10, cls_type='km', clf_type='svm'):
        # miceF : miceFeature class object
        # get feature
        feat = []
        if sel[0]=='random':
            miceFs = self.mice_feat
            ind = np.random.choice(np.arange(len(miceFs)), sel_num, replace=False)
            for i in ind:
                feat.append(miceFs[i].feature)
        else:
            miceFs = []
            for s in sel:
                miceFs.extend(self.sel_feat(s))
            for miceF in miceFs:
                feat.append(miceF.feature)
        feat = np.concatenate(feat)
        # if is lstm => flatten to 2d feature
        if len(feat.shape)>2:
            feat = feat.reshape(len(feat), feat.shape[1]*feat.shape[2])
        # cluster
        if embed:
            embeder, embeddings = embedfeat(feat)
            motions, mclf = motion_cluster(embeddings, k, cls_type)
            self.embeder = embeder
        else:
            motions, mclf = motion_cluster(feat, k, cls_type)
        motion_num = len(np.unique(motions))
        if not mclf:
            mclf = motion_clf(feat, motions, clf_type=clf_type)
        # cluster predict and save result
        motionsB = [0]*motion_num
        motionsT = [0]*motion_num
        miceFsB, miceFsT = self.sel_feat('Capbasal'), self.sel_feat('Cap')
        for i in range(len(miceFsB)):
            miceFB = miceFsB[i]
            miceFT = miceFsT[i]
            if embed:
                motionB = motion_predict(miceFB.feature, mclf, embeder)
                motionT = motion_predict(miceFT.feature, mclf, embeder)
            else:    
                motionB = motion_predict(miceFB.feature, mclf)
                motionT = motion_predict(miceFT.feature, mclf)
            for i in np.unique(motions):
                motionsB[i]+= len(np.where(motionB==i)[0])
                motionsT[i]+= len(np.where(motionT==i)[0])
        # motion score
        motion_num = len(motionsB)
        ratio = np.zeros((motion_num), dtype=float)
        for i in range(motion_num):
            if (motionsB[i]+motionsT[i])>0:
                ratio[i] = motionsT[i]/(motionsB[i]+motionsT[i])
        motion_score = np.zeros((motion_num), dtype=float)
        th = 0.4
        motion_score[(ratio<=th) | (ratio>=1-th)] = 1
        motion_score[(ratio>th) & (ratio<1-th)] = -1
        print("bad motions:", len(np.where(motion_score==-1)[0]))
        # plot 
        x = np.arange(motion_num)
        width = 0.3
        plt.bar(x, motionsB, width, color='green', label='basal')
        plt.bar(x + width, motionsT, width, color='red', label='treat')
        plt.xticks(x + width / 2, x)
        plt.legend(bbox_to_anchor=(1,1), loc='upper left')
        #plt.show()
        #plt.savefig()
        self.mclf = mclf
        self.motionsB = motionsB
        self.motionsT = motionsT
        self.motion_score = motion_score



class miceFeature:
    '''
    Storing All data(file paths, landmarks, features ...) of single mice(file)
    '''
    def __init__(self, treatment, dlc=None, bsoid=None, vidc=None, vids=None, dep=None, feat_type='frame'):
        self.treatment = treatment
        if(dlc):
            self.dlcfile = dlc
            self.read_dlc()
        if(bsoid):
            self.bsoidfile = bsoid
        if(vidc):
            self.vidcfile = vidc
        if(vids):
            self.vidsfile = vids
        if(dep):
            self.depfile = dep

        self.count_feature(feat_type=feat_type)
    
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
    def count_feature(self, feat_type='frame'):
        # config
        sel_dist=[[0,1],[0,2],[1,3],[2,3],[3,4],[3,5],[4,6],[5,6]]
        sel_ang=[[1,3,2],[0,3,6],[4,3,5]]
        sel_coord=[]
        normalize_range=(0,1)
        include_index = False
        window = 10
        if feat_type[-1]=='F':
            step = 10
        else:
            step = 5

########### frame ################################################
        if feat_type == 'frame':
            # config frame
            sel_dist=[[0,3],[3,4],[1,2]]
            sel_ang=[[0,1,3],[1,3,4]]
            # frame feature pre
            dist = count_dist(self.dlc_raw, sel_dist)
            ang = count_angle(self.dlc_raw, sel_ang)
            # disp = count_disp(self.dlc_raw, step=1, threshold=None)
            # frame feature
            feat = dist
            feat = np.hstack([feat, ang])
            # feat = np.hstack([feat, disp[:,0:1]])
            # normalize
            feat = feature_normalize(feat, normalize_range=normalize_range)
#########bsoid + cwt (full/half)##################################
        if feat_type[:-1] == 'bscwt':
            # frame feature pre
            dist = count_dist(self.dlc_raw, sel_dist)[1:]
            ang = count_angle(self.dlc_raw, sel_ang)[1:]
            disp = count_disp(self.dlc_raw, step=1, threshold=None)
            # frame feature
            feat = dist[:,5:6]
            feat = np.hstack([feat, dist[:,7:8]])
            feat = np.hstack([feat, ang[:,2:3]])
            feat = np.hstack([feat, disp[:,2:3]])
            # segment feature
            # seg = abs(fft_signal(feat, window=seg_window, flat=True))
            seg = cwt_signal(feat, window=window, step=step)
            # combine
            tmp = np.hstack([disp, ang])
            feat = np.hstack([seg, seg_statistic(tmp, count_types=['avg'], window=window, step=step)])
            feat = np.hstack([feat, seg_statistic(dist, count_types=['sum'], window=window, step=step)])
            # normalize
            feat = feature_normalize(feat, normalize_range=normalize_range)
########## bsoid ##########################################################
        if feat_type[:-1] == 'bs':
            savfile = joblib.load(self.bsoidfile)
            if len(savfile) > 10:
                feat = savfile
            else:
                feat = savfile[0]
########### bsoid LSTM ############################################
        if feat_type[:-1] == 'bsLSTM':
            # frame feature pre
            dist = count_dist(self.dlc_raw, sel_dist)[1:]
            ang = count_angle(self.dlc_raw, sel_ang)[1:]
            disp = count_disp(self.dlc_raw, step=1, threshold=None)
            # segment feature combine
            tmp = np.hstack([disp, ang, disp])
            tmp = feature_normalize(tmp, normalize_range=normalize_range)
            feat = generate_tmpfeat(tmp, window=window, step=step)
########### bsoid + cwt LSTM ############################################
        if feat_type[:-1] == 'bscwtLSTM':
            # frame feature pre
            dist = count_dist(self.dlc_raw, sel_dist)[1:]
            ang = count_angle(self.dlc_raw, sel_ang)[1:]
            disp = count_disp(self.dlc_raw, step=1, threshold=None)
            # frame feature
            feat = dist[:,5:6]
            feat = np.hstack([feat, dist[:,7:8]])
            feat = np.hstack([feat, ang[:,2:3]])
            feat = np.hstack([feat, disp[:,2:3]])
            # segment feature combine
            seg = cwt_signal(feat, window=window, step=step, flat=False)
            tmp = np.hstack([dist, ang, disp])
            tmp = feature_normalize(tmp, normalize_range=normalize_range)
            feat = generate_tmpfeat(tmp, window=window, step=step)
            feat = np.concatenate([feat, seg], axis=2)
#########################################################################
        self.feature = feat

    ### train test config ##########################################################################
    def labeling(self, mclf=None, motion_score=None):
        # pain:1 sng:2 health:0
        labels = np.zeros((self.feature.shape[0]), dtype=int)
        if self.treatment == 'pH5.2':
            labels[:] = 2
        elif self.treatment == 'pH7.4' or self.treatment.find('basal')!=-1 or \
                self.treatment.find('pH5.2ASIC3KO')!=-1 or self.treatment.find('CapTV1KO')!=-1:
            labels[:] = 0
        elif self.treatment == 'Cap':
            labels[:] = 1
        if mclf:
            motions = motion_predict(self.feature, mclf)
            for i in range(len(motion_score)):
                if motion_score[i] == -1:
                    labels[np.where(motions==i)] = 0 ## bad motion label
        self.label=labels
         
    def train_config(self, split=0.5, shuffle=True, motion_del=False):
        # select sample
        if motion_del:
            feat = self.feature[np.where(self.label!=-1)]
            label = self.label[np.where(self.label!=-1)]
        else:
            feat = self.feature
            label = self.label
        # shuffle
        ind = np.arange(len(feat))
        np.random.shuffle(ind)
        self.shuffle = ind
        # split
        if shuffle:
            feat = feat[ind]
            label = label[ind]
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
                
class Analysis:
    def __init__(self, model_type='svm', classes=3):
        if model_type == 'svm':
            self.model = SVC(kernel='rbf', C=1000)
        elif model_type == 'rf':
            self.model = RandomForestClassifier(random_state=42)
        elif model_type == 'dnn':
            self.model = DNN_model(classes)
        elif model_type == 'lstm':
            self.model = LSTM_model(classes)

    def train(self, x, y):
        self.model = self.model.fit(x,y)

    def test(self, x, y, show=False):
        sc = self.model.score(x, y)
        if show:
            print('accuracy = ',sc)
        return sc

    def analysis(self, x, y, seperate=False):
        pred = self.model.predict(x)

        # non-seperate
        tp = np.count_nonzero(((y==1) & (pred==1)) | ((y==2) & (pred==2)))
        tn = np.count_nonzero(((y==0) & (pred==0)) | ((y==-1) & (pred==-1)))
        fp = np.count_nonzero(((y==0) & ((pred==1)|(pred==2))) | ((y==-1) & ((pred==1)|(pred==2)))  |((y==1) & (pred==2)) | ((y==2) & (pred==1)) )  # mis postive 
        fn = np.count_nonzero(((y==1) & ((pred==0)|(pred==-1)|(pred==2))) | ((y==2) & ((pred==0)|(pred==-1)|(pred==1))))  #|((y==0) & (pred==-1)) | ((y==-1) & (pred==0)) ) # mis negative
        toler = np.count_nonzero(((y==0) & (pred==-1)) | ((y==-1) & (pred==0))) # miss negative is useless
        if (fp+tn)==0:
            fa = 0
        else:
            fa = fp/(fp+tn)
        if (tp+fn)==0:
            dr = 0
        else:
            dr = tp/(tp+fn)
        acc = (tp+tn)/(tp+tn+fp+fn)
        print('accuracy = ', acc)
        print("false alarm: ", fa)
        print("detection rate: ", dr)
        return [acc,fa,dr]

    def analysis2(self, x, y):
        '''
        seperate detection rate of pain/sng
        '''
        pred = self.model.predict(x)
        tn = np.count_nonzero(((y==0) & (pred==0)) | ((y==-1) & (pred==-1)))
        fn_p = np.count_nonzero((y==1) & ((pred==0)|(pred==-1)|(pred==2)))
        tp_p = np.count_nonzero((y==1) & (pred==1))
        # fp_p = np.count_nonzero(((y==0) & (pred==1)) | ((y==-1) & (pred==1)) | ((y==2) & (pred==1)) )
        fn_s = np.count_nonzero((y==2) & ((pred==0)|(pred==-1)|(pred==1)))
        tp_s = np.count_nonzero((y==2) & (pred==2))
        # fp_s = np.count_nonzero(((y==0) & (pred==2)) | ((y==-1) & (pred==2)) | ((y==1) & (pred==2)) )
        if (tp_p+fn_p)==0:
            dr_p = 0
        else:
            dr_p = tp_p/(tp_p+fn_p)
        if (tp_s+fn_s)==0:
            dr_s = 0
        else:
            dr_s = tp_s/(tp_s+fn_s)
        return [dr_p, dr_s]

    def feat_importance(self, x, y, save_path=None):
        r = permutation_importance(self.model, x, y, n_repeats=10, random_state=0)
        feature_names = np.arange(len(x[0]))
        features = np.array(feature_names)
        # sorted_idx = r.importances_mean.argsort()
        plt.barh(features, r.importances_mean)
        plt.xlabel("Permutation Importance")
        if save_path:
            plt.savefig(save_path)
        return r.importances_mean


class LSTM_model:
    def __init__(self, classes):
        self.classes = classes
        self.build_model()

    def build_model(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.LSTM(32, return_sequences=False))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(self.classes, activation='softmax'))
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    def fit(self, x, y):
        self.classes_ = np.arange(self.classes)
        if self.classes == 4:
            self.classes_ = self.classes_-1
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=20, mode='max')]
        self.model.fit(x, tf.one_hot(y,self.classes), epochs=200, batch_size=16) #,callbacks=callbacks)
        return self

    def predict(self, x):
        if self.classes == 4:
            return np.argmax(self.model.predict(x), axis=1)-1
        return np.argmax(self.model.predict(x), axis=1)

class DNN_model:
    def __init__(self, classes):
        self.classes = classes
        self.build_model()
    def build_model(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(32),activation='relu')
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(32),activation='relu')
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(self.classes, activation='softmax'))
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    def fit(self, x, y):
        self.classes_ = np.arange(self.classes)
        if self.classes == 4:
            self.classes_ = self.classes_-1
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=20, mode='max')]
        self.model.fit(x, tf.one_hot(y,self.classes), epochs=200, batch_size=16) #,callbacks=callbacks)
        return self

    def predict(self, x):
        if self.classes == 4:
            return np.argmax(self.model.predict(x), axis=1)-1
        return np.argmax(self.model.predict(x), axis=1)
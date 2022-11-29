from feature_process import *
from pose_cluster import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import tensorflow as tf
import csv

class DataSet:
    '''
    Storing dataset to train/to test, root of related files, info of each single mice
    '''
    def __init__(self, dlc, bsoid=None, vidc=None, vids=None, dep=None, specific=[]):
        self.specific = specific
        self.all_treatment = ['Capbasal','Cap','pH5.2basal','pH5.2','pH7.4basal','pH7.4']
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
        print('basal:',len(self.ind['basal']),' ,pain:',len(self.ind['Cap']),' sng:',len(self.ind['pH5.2']),' pH7.4:',len(self.ind['pH7.4']))

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
            for i in range(len(inds)): #last one for validate
                if i==len(inds)-k:
                    continue
                ind = inds[i]
                self.data['x_train'].append(self.mice_feat[ind].x_train)
                self.data['y_train'].append(self.mice_feat[ind].y_train)
                self.data['x_test'].append(self.mice_feat[ind].x_test)
                self.data['y_test'].append(self.mice_feat[ind].y_test)
            ind = inds[len(inds)-k]
            self.data['x_val'].append(self.mice_feat[ind].feature)
            self.data['y_val'].append(self.mice_feat[ind].label)

    def generate_train_test2(self, split=0.5, motion_del=False):
        '''
        validation setting as pH7.4
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
        for t in ['Capbasal','Cap','pH5.2basal','pH5.2']:
            inds = self.ind[t]
            for i in range(len(inds)):
                ind = inds[i]
                self.data['x_train'].append(self.mice_feat[ind].x_train)
                self.data['y_train'].append(self.mice_feat[ind].y_train)
                self.data['x_test'].append(self.mice_feat[ind].x_test)
                self.data['y_test'].append(self.mice_feat[ind].y_test)
        for t in ['pH7.4basal','pH7.4']:
            inds = self.ind[t]
            for i in range(len(inds)):
                ind = inds[i]
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
        plt.show()
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
        sel_dist=[[0,3],[3,6],[0,1],[0,2],[3,4],[3,5]]
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
            feat = dist[:,0:2]
            feat = np.hstack([feat, ang[:,0:1]])
            feat = np.hstack([feat, disp[:,0:1]])
            # segment feature
            # seg = abs(fft_signal(feat, window=seg_window, flat=True))
            seg = cwt_signal(feat, window=window, step=step)
            # combine
            tmp = np.hstack([dist, ang])
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
            tmp = np.hstack([dist, ang, disp])
            tmp = feature_normalize(tmp, normalize_range=normalize_range)
            feat = generate_tmpfeat(tmp, window=window, step=step)
########### bsoid + cwt LSTM ############################################
        if feat_type[:-1] == 'bscwtLSTM':
            # frame feature pre
            dist = count_dist(self.dlc_raw, sel_dist)[1:]
            ang = count_angle(self.dlc_raw, sel_ang)[1:]
            disp = count_disp(self.dlc_raw, step=1, threshold=None)
            # frame feature
            feat = dist[:,0:2]
            feat = np.hstack([feat, ang[:,0:1]])
            feat = np.hstack([feat, disp[:,0:1]])
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
        labels = np.zeros_like(self.feature[:,0], dtype=int)
        if self.treatment == 'pH5.2':
            labels[:] = 2
        elif self.treatment == 'pH7.4' or self.treatment.find('basal')!=-1:
            labels[:] = 0
        elif self.treatment == 'Cap':
            labels[:] = 1
        if mclf:
            motions = motion_predict(self.feature, mclf)
            for i in range(len(motion_score)):
                if motion_score[i] == -1:
                    labels[np.where(motions==i)] = -1
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
    def __init__(self, model_type='svm', classes=3, save_path=''):
        self.save_path = ''
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

    def plot_cm(self, x, y, score=True):
        pred = self.model.predict(x)
        if score:
            # print('accuracy = ', accuracy_score(y, pred))
            self.analysis(y, pred)
        cm = confusion_matrix(y, pred, labels=self.model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_)
        disp.plot()
        if(len(self.save_path)>0):
            disp.savefig(save_path+'/cm.png')

    def analysis(self, y, pred):
        tp = np.count_nonzero(((y==1) & (pred==1)) | ((y==2) & (pred==2)))
        tn = np.count_nonzero(((y==0) & (pred==0)) | ((y==-1) & (pred==-1)))
        fp = np.count_nonzero(((y==0) & ((pred==1)|(pred==2))) | ((y==-1) & ((pred==1)|(pred==2))) )
        fn = np.count_nonzero(((y==1) & ((pred==0)|(pred==-1)|(pred==2))) | ((y==2) & ((pred==0)|(pred==-1)|(pred==1))) )
        tolor = np.count_nonzero(((y==0) & (pred==-1)) | ((y==-1) & (pred==0)))
        if (fp+tn)==0:
            fa = 0
        else:
            fa = fp/(fp+tn)
        if (tp+fn)==0:
            dr = 0
        else:
            dr = tp/(tp+fn)
        acc = (tp+tn)/(tp+tn+fp+fn+tolor)
        print('accuracy = ', acc)
        print("false alarm: ", fa)
        print("detection rate: ", dr)
        if(len(self.save_path)>0):
            file = open(r'C:\Users\x\Desktop\final_data/analysis.csv',mode='a', newline='')
            writer = csv.writer(file)
            save_path = self.save_path
            if(save_path[-1]=='/'or save_path[-1]=='\\'):
                save_path = save_path[:-1]
            save_path = save_path.split('/')[-1]
            save_path = save_path.split('\\')[-1]
            writer.writerow([save_path,acc,fa,dr])
            file.close()


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
        self.model.fit(x, tf.one_hot(y,self.classes), epochs=500, batch_size=16,callbacks=callbacks)

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
        self.model.add(tf.keras.layers.Dense(32))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(32))
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
        self.model.fit(x, tf.one_hot(y,self.classes), epochs=500, batch_size=16,callbacks=callbacks)
        return self

    def predict(self, x):
        if self.classes == 4:
            return np.argmax(self.model.predict(x), axis=1)-1
        return np.argmax(self.model.predict(x), axis=1)
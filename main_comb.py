from feature_process import *
from data_process import *
from pose_cluster import *
import csv

add_clusters=['N','Y'] #'N','Y'
feat_types=['bsF','bsH','bscwtF','bscwtH','bsLSTMF','bsLSTMH','bscwtLSTMF','bscwtLSTMH','frame']#'frame','bsF','bsH','bscwtF','bscwtH','bsLSTMF','bsLSTMH','bscwtLSTMF','bscwtLSTMH'
model_types=['svm', 'rf', 'dnn', 'lstm'] #'svm', 'rf', 'dnn', 'lstm'
save_root = r'C:\Users\x\Desktop\final_data/analysis2/'

#skip finish
cur=0
fin = 50

for feat_type in feat_types:
    # feat type
    if feat_type == 'frame':
        dlc_root = r'C:\Users\x\Desktop\final_data\mix_landmark5'
        dlc = DataSet(dlc_root)
    else:
        dlc_root = r'C:\Users\x\Desktop\final_data\mix_landmark7'
        if feat_type[-1]=='H':
            bs_root = r'C:\Users\x\Desktop\final_data\mix_bsoidfeat'
        else:
            bs_root = r'C:\Users\x\Desktop\final_data\mix_bsoidfeat2'
        dlc = DataSet(dlc_root, bsoid=bs_root)
    print('generate feature...')
    dlc.generate_feature(feat_type=feat_type)
    for add_cluster in add_clusters:
        # add cluster
        if add_cluster == 'Y':
            print('pose clustering...')
            dlc.pose_cls(sel=['Cap','Capbasal'], sel_num=20, embed=False, k=50, cls_type='km', clf_type='svm')
            classes=3
        else:
            classes=3

        for model_type in model_types:
            if(model_type=='lstm' and feat_type.find('LSTM')==-1):
                continue
            if(feat_type.find('LSTM')!=-1 and model_type!='lstm'):
                continue

            # 10 different true test
            exp = add_cluster+feat_type+model_type
            print(exp)
            for i in range(10):

                ## skip finish
                if cur < fin:
                    cur+=1
                    continue
                cur+=1
                #####

                res = []

                dlc.generate_train_test(split=0.1, motion_del=False, k=i+1)

                # model
                x_train = np.concatenate(dlc.data['x_train'])
                y_train = np.concatenate(dlc.data['y_train'])
                model = Analysis(model_type=model_type, classes=classes)
                print('model training...')
                model.train(x_train,y_train)
                res.extend(model.analysis(x_train, y_train))
                print('model testing...')
                x_test = np.concatenate(dlc.data['x_test'])
                y_test = np.concatenate(dlc.data['y_test'])
                res.extend(model.analysis(x_test, y_test))
                print('model testing...')
                x_val = np.concatenate(dlc.data['x_val'])
                y_val = np.concatenate(dlc.data['y_val'])
                res.extend(model.analysis(x_val, y_val))

                file = open(r'C:\Users\x\Desktop\final_data/analysis2.csv',mode='a', newline='')
                writer = csv.writer(file)
                line = [exp,i]
                line.extend(res)
                writer.writerow(line)
                file.close()
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
# cluster
from sklearn.decomposition import PCA
import umap
import hdbscan
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

# from data_process import *

class myEmbeder:
    def __init__(self, sel):
        self.sel = sel
    def transform(self, x):
        return x[:,self.sel]

def embedfeat(feat, num_dimensions=None, sel=[], savename=None):
    '''
    Dimension reduction
        num_dimensions: to certain dimension by UMAP
        sel: select certain features
    '''
    if len(sel)>0: # select feature manually
        embeddings = feat[:,sel]
        embeder = myEmbeder(sel)
    else: # select feature by UMAP
        if not num_dimensions:
            pca = PCA()
            pca.fit(feat)
            num_dimensions = np.argwhere(np.cumsum(pca.explained_variance_ratio_) >= 0.7)[0][0] + 1
        sampled_input_feats = feat[np.random.choice(feat.shape[0], feat.shape[0], replace=False)]
        embeder = umap.UMAP(n_neighbors=60, n_components=num_dimensions, min_dist=0.0, random_state=42).fit(sampled_input_feats)
        embeddings = embeder.embedding_
    if savename:
        joblib.dump(embeder, savename)
    return embeder, embeddings

def motion_cluster(feat, k=None, cls_type='hdbscan'):
    if cls_type=='hdbscan':
        mcls=None
        min_c = k
        print("min cluster size: ", int(round(min_c * 0.01 * feat.shape[0])))
        learned_hierarchy = hdbscan.HDBSCAN(
                            prediction_data=True, min_cluster_size=int(round(min_c * 0.01 * embeddings.shape[0])),
                            min_samples=1).fit(feat)
        labels = learned_hierarchy.labels_
        assign_prob = hdbscan.all_points_membership_vectors(learned_hierarchy)
        assignments = np.argmax(assign_prob, axis=1)
    elif cls_type=='spec':
        mcls = SpectralClustering(n_clusters=k, assign_labels='discretize', random_state=0).fit(embeddings)
        assignments = mcls.labels_
    elif cls_type=='km':
        mcls = KMeans(n_clusters=k).fit(feat)
        assignments = mcls.labels_
    elif cls_type=='af':
        mcls =  AffinityPropagation().fit(feat)
        assignments = mcls.labels_
    print("motions num: ", len(np.unique(assignments)))
    return assignments, mcls

def motion_clf(x, y, test_part=0.1, score=True, savename=None, clf_type='svm'):
    if test_part:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    else:
        x_train, y_train = x, y
    if clf_type=='rf':
        validate_clf = RandomForestClassifier(random_state=0)#
        clf = RandomForestClassifier(random_state=42)#
    else:
        validate_clf = SVC(kernel='rbf', C=100)
        clf = SVC(kernel='rbf', C=100)
    validate_clf.fit(x_train, y_train)
    clf.fit(x, y)
    if(score):
        print(cross_val_score(validate_clf, x_test, y_test, cv=5, n_jobs=-1))
    if savename:
        joblib.dump(clf, savename)
    return clf

def motion_predict(feat, clf, embeder=None):
    if embeder:
        test_embedding = embeder.transform(feat)
    else:
        test_embedding = feat
    labels = clf.predict(test_embedding)
    return labels

### pose analysis on data ##########################################################################
def motion_score(motionB, motionT, score_type='clf', show=False):
    motion_num = len(motionB)
    diff = np.array(motionT)-np.array(motionB)
    sum2 = np.array(motionT)+np.array(motionB)
    ratio = np.zeros((motion_num), dtype=float)
    for i in range(motion_num):
        if (motionB[i]+motionT[i])>0:
            ratio[i] = motionT[i]/(motionB[i]+motionT[i])
    motion_score = np.zeros((motion_num), dtype=float)
    if score_type=='clf':
        th = 0.4
        if show:
            print(abs(diff)/sum(abs(diff)))
            print(np.array(sum2)/sum(sum2))
        motion_score[(ratio<=th) | (ratio>=1-th)] = 1
        motion_score[(ratio>th) & (ratio<1-th)] = -1
    return motion_score
# def cluster_micedata(miceD, sel=['random'], sel_num=20, 
#                     embed=False, k=10, cls_type='km', clf_type='svm'):
#     # miceD : DataSet class object
#     # miceF : miceFeature class object
#     # get feature
#     feat = []
#     if sel[0]=='random':
#         miceFs = miceD.mice_feat
#         ind = np.random.choice(np.arange(len(miceFs)), sel_num, replace=False)
#         for i in ind:
#             feat.append(miceFs[i].feature)
#     else:
#         miceFs = []
#         for s in sel:
#             miceFs.entend(miceD.sel_feat(s))
#         for miceF in miceFs:
#             feat.append(miceF.feature)
#     feat = np.concatenate(feat)
#     # cluster
#     if embed:
#         embeder, feat = embedfeat(feat)
#     motions = motion_cluster(feat, k, cls_type)
#     motion_num = len(np.unique(motions))
#     mclf = motion_clf(feat, motions, clf_type=clf_type)
#     # cluster predict and save result
#     motionsB = [0]*motion_num
#     motionsT = [0]*motion_num
#     miceFsB, miceFsT = miceD.sel_feat('Capbasal'), miceD.sel_feat('Cap')
#     for i in range(len(miceFsB)):
#         miceFB = miceFsB[i]
#         miceFT = miceFsT[i]
#         if embed:
#             motionB = motion_predict(miceFB.feature, mclf, embeder)
#             motionT = motion_predict(miceFT.feature, mclf, embeder)
#         else:    
#             motionB = motion_predict(miceFB.feature, mclf)
#             motionT = motion_predict(miceFT.feature, mclf)
#         for i in np.unique(motions):
#             motionsB[i]+= len(np.where(motionB==i)[0])
#             motionsT[i]+= len(np.where(motionT==i)[0])
#     # plot 
#     x = np.arange(motion_num)
#     width = 0.3
#     plt.bar(x, motionsB, width, color='green', label='basal')
#     plt.bar(x + width, motionsT, width, color='red', label='treat')
#     plt.xticks(x + width / 2, x)
#     plt.legend(bbox_to_anchor=(1,1), loc='upper left')
#     plt.show()
#     return motionsB,motionsT
    # cluster basal/treat statistic
    # diff = np.array(motionsT)-np.array(motionsB)
    # sum2 = np.array(motionsT)+np.array(motionsB)
    # ratio = np.zeros((motion_num), dtype=float)
    # for i in range(motion_num):
    #     ratio[i] = motionsT[i]/(motionsB[i]+motionsT[i])
    # count motion score
    # motion_score = np.zeros((motion_num), dtype=float)
    # if score_type=='clf':
    #     th = 0.4
    #     if show:
    #         print(abs(diff)/sum(abs(diff)))
    #         print(np.array(sum2)/sum(sum2))
    #     motion_score[(ratio<=th) | (ratio>=1-th)] = 1
    #     motion_score[(ratio>th) & (ratio<1-th)] = -1
    #     # motion_score[np.where(abs(diff)/sum(abs(diff))>0.04)] = 1
    #     # motion_score[np.where(abs(diff)/sum(abs(diff))<=0.04)] = -1
    # elif score_type=='reg1':
    #     motion_score = ratio
    # else: # score by pose
    #     motion_score = np.ones((motion_num), dtype=float)
    # return motion_score ,motion_num
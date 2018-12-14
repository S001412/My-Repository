# -*- coding: utf-8 -*-

from sklearn.cluster import hierarchical
# import matplotlib.pyplot as plt
from sklearn import metrics
from numpy.linalg import norm
import numpy as np
# from mysql_data_precessing import test_vector
from operator import itemgetter
from data_preprocessing import label_data, comment_data_matrix, label_comment_matrix
from keras.models import load_model

# km = KMeans(n_clusters=10, max_iter=600, n_init=50)
# y_pred = km.fit_predict(test_vector)
# c = metrics.calinski_harabaz_score(test_vector, y_pred)
# # plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# a = metrics.calinski_harabaz_score(X, y_pred)
# label_pred = km.labels_ #获取聚类标签
# centroids = km.cluster_centers_ #获取聚类中心
# inertia = km.inertia_ # 获取聚类准则的总和
# # plt.show()
# bh = Birch(n_clusters=10)
# y_p2 = bh.fit_predict(test_vector)
# b = metrics.calinski_harabaz_score(test_vector, y_p2)


def cosine_similarity(m1, m2):
    cos_sim = np.dot(m1, m2) / (norm(m1) * norm(m2))
    return 1 - cos_sim


# 建立相似度矩阵，输入一个字典，输出字典中元素两两之间的相似度,由高到低排序
def cosine_sim_matrix(set1, matrix):
    sim_dict = {}
    for i in set1:
        sim_dict[i] = {}
        for j in set1:
            if i == j:
                continue
            else:
                sim_dict[i][j] = cosine_similarity(matrix[i, :], matrix[j, :])
        sim_dict[i] = sorted(sim_dict[i].items(), key=itemgetter(1), reverse=True)
    return sim_dict


def cluster(test_vector):
    hi = hierarchical.AgglomerativeClustering(n_clusters=10)
    y_pred = hi.fit_predict(test_vector)
    a = metrics.calinski_harabaz_score(test_vector, y_pred)
    # 建立类别字典
    labels = set(y_pred)
    class_dict = {}
    for label in labels:
        class_dict[label] = set()
    # 将同一类别对应的编号放在同一个键中
    for y_index in range(len(y_pred)):
        for label in class_dict:
            if y_pred[y_index] == label:
                class_dict[label].add(y_index)
    return class_dict


def distEclud(vecA, vecB):
    '''
    输入：向量A和B
    输出：A和B间的欧式距离
    '''
    return np.sqrt(sum(np.power(vecA - vecB, 2)))


def newCent(L, label_list):
    '''
    输入：有标签数据集L
    输出：根据L确定初始聚类中心
    '''
    centroids = []
    label_list = np.unique(label_list)
    for i in label_list:
        L_i = L[i: i*10]
        cent_i = np.mean(L_i, 0)
        centroids.append(cent_i)
    return np.array(centroids)


def semi_kMeans(L, U, label_list, distMeas=distEclud, initial_centriod=newCent):
    '''
    输入：有标签数据集L（最后一列为类别标签）、无标签数据集U（无类别标签）
    输出：聚类结果
    '''
    # 合并L和U
    dataSet = np.vstack((L, U))
    label_list = np.unique(label_list)
    # L中类别个数
    k = len(label_list)
    m = np.shape(dataSet)[0]
    # 初始化样本的分配
    clusterAssment = np.zeros(m)
    # 确定初始聚类中心
    centroids = initial_centriod(L, label_list)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # 将每个样本分配给最近的聚类中心
        for i in range(m):
            minDist = np.inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i] != minIndex: clusterChanged = True
            clusterAssment[i] = minIndex
    return clusterAssment[100:]


label_ = label_data[:, 0]
encoder = load_model('encoder_label1.h5')
x_train_encoded = encoder.predict(comment_data_matrix)
x_test_encoded = encoder.predict(label_comment_matrix)
x_label = semi_kMeans(x_test_encoded, x_train_encoded, label_, distMeas=distEclud, initial_centriod=newCent)

if __name__ == '__main__':
    b = metrics.calinski_harabaz_score(x_train_encoded, x_label)
    # cluster_dict = cluster()
    # # 计算某个类别中元素之间的相似度
    # sim_matrix = cosine_sim_matrix(cluster_dict[0], test_vector)

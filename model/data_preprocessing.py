# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from jieba import analyse
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split


def label_onehot(array, vocab_size, length):
    vocab_size = vocab_size
    array_onehot_encoder = [one_hot(d, vocab_size) for d in array]
    # 扩展至同一长度
    max_length = length
    onehot_result_array = pad_sequences(array_onehot_encoder, maxlen=max_length, padding='post')
    onehot_result_array = preprocessing.normalize(onehot_result_array)
    return onehot_result_array


# 输入数值数据，使用sklearn返回数据的归一化结果
def min_max_scale(array, row):
    array = preprocessing.minmax_scale(array)
    array = array.reshape(row, 1)
    return array


# 输入带汉字的标签，返回onehot结果
def onehot_encoder(data):
    # integer encode, 标签转化为onehot
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(data)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    onehot_result = onehot_encoded
    return onehot_result


# 将分词后的结果转变为词向量，输入 array，输出array
def word_vectorization(word_model, array, row_array, limit):
    # 用来存储单个词语向量化后的结果，长度由生成词模型时的设置解果决定
    size = 100
    res_array = np.zeros(size * limit)
    for row_number in range(row_array):
        if len(array[row_number]) == 1:
            res_array = np.vstack((res_array, np.zeros(size * limit)))
            continue
        else:
            try:
                split_list = array[row_number].split(' ')
                tem_array = word_model[split_list[0]]
            except KeyError:
                tem_array = np.zeros(size)
        for index in range(1, limit):
            if index < len(split_list):
                try:
                    word_vec = word_model[split_list[index]]
                    tem_array = np.concatenate((tem_array, word_vec))
                except KeyError:
                    tem_array = np.concatenate((tem_array, np.zeros(size)))
            else:
                tem_array = np.concatenate((tem_array, np.zeros(size)))
        res_array = np.vstack((res_array, tem_array))
    res_array = np.delete(res_array, 0, axis=0)
    return res_array


user_data = pd.read_excel('评论信息+用户信息.xlsx'.encode('utf-8'))
# user_data = user_data[['有用数', '感谢次数', '总分', '评论数', '读者数', '景点名称', '评分', '标题', '内容']]
user_data = user_data[['标题'.encode('utf-8'), '内容'.encode('utf-8')]]
user_data = pd.DataFrame(user_data).fillna({'标题'.encode('utf-8'): '0'.encode('utf-8'), '内容'.encode('utf-8'): '0'.encode('utf-8')})
user_data = np.array(user_data)
row_user_data = user_data.shape[0]
column_user_data = user_data.shape[1]

label_data = pd.read_excel('帖子标签.xlsx'.encode('utf-8'))
label_data = np.array(label_data)
row_label_data = label_data.shape[0]

# user_data_type = user_data[:, 5]
# input_vec = onehot_encoder(user_data_type)

# user_rating_score = min_max_scale(user_data[:, 6], row_user_data)
# input_vec = user_rating_score

tfidf = analyse.extract_tags
# comment_data = copy.deepcopy(user_data[:, 0])
comment_data = user_data[:, 0]
for i in range(row_user_data):
    if type(user_data[i, 0]) != str:
        user_data[i, 0] = '0'
    title_list = tfidf(user_data[i, 0])
    comment_list = tfidf(user_data[i, 1])
    comment_data[i] = ' '.join(title_list)
    comment_data[i] = ' '.join(comment_list)

label_comment = label_data[:, 1]
for l in range(row_label_data):
    label_comment_list = tfidf(label_data[l, 1])
    label_comment[l] = ' '.join(label_comment_list)

# 读取模型，构建词向量矩阵
word_model = KeyedVectors.load_word2vec_format("comment.model.bin", binary=True)
comment_data_matrix = word_vectorization(word_model, comment_data, row_user_data, 10)
label_comment_matrix = word_vectorization(word_model, label_comment, row_label_data, 10)

# label_comment_matrix = np.hstack((label_comment_array, np.zeros((row_label_data, 1))))
#
# for i in range(row_label_data):
#     label_comment_matrix[i, -1] = label_data[i, 0]

# x_train, x_test = train_test_split(comment_data_matrix, test_size=0.2)

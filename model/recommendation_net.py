import random
import numpy as np
from keras.models import load_model
from mysql_data_precessing import test_vector
from mysql_data_precessing import mysql_data

recommend_model = load_model('mysql_recommendation.h5')
y_pred = recommend_model.predict(test_vector)
# 将onehot编码转换为对应的数值
y_label = np.zeros(y_pred.shape[0])
for i in range(len(y_pred)):
    max_num = y_pred[i][0]
    for j in range(1, y_pred.shape[1]):
        if y_pred[i][j] > max_num:
            max_num = y_pred[i][j]
            y_label[i] = j

y_set = set(y_label)
class_dict = {}
for y in y_set:
    class_dict[y] = []

# 景点序号放在字典对应的列表中
spot_id = 0
for label in y_label:
    class_dict[label].append(spot_id)
    spot_id += 1

rand_type = random.randint(0, len(y_set)-1)
while len(class_dict[rand_type]) < 3:
    rand_type = random.randint(0, len(y_set))

# 建立推荐景点列表，每次随机推荐3个景点，
# TODO 根据用户标签推荐对应类别的景点
recommendation_list = []
for i in class_dict[rand_type]:
    recommendation_list.append(mysql_data[i, :])
    if len(recommendation_list) > 2:
        break

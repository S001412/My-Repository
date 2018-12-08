# -*- coding: utf-8 -*-

from keras.layers import Dense, Dropout, Input
from keras.models import Sequential, Model
from keras.utils import to_categorical
from keras import regularizers
# from keras.models import load_model
# from vae_keras import num_classes
from data_preprocessing import comment_data_matrix
from K_means import x_label
from sklearn.model_selection import train_test_split

num_classes = 10


def fit_linear_model(X_train, Y_train, node1, node2, batch_size, epoch):
    # create model
    model = Sequential()
    model.add(Dense(node1, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(node2, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(16, activation='relu'))
    model.add(Dense(32, input_dim=128,
                    kernel_regularizer=regularizers.l2(0.1)))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # model.add(Dense(1, init='normal'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, nb_epoch=epoch, verbose=1, batch_size=batch_size)
    return model


# 超参数寻优函数
# def train_linear_model(args):
#     # create model
#     model = Sequential()
#     model.add(Dense(int(args["node1"]), input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu'))
#     model.add(Dropout(0.25))
#     model.add(Dense(int(args["node2"]), activation='relu'))
#     model.add(Dropout(0.25))
#     model.add(Dense(32, input_dim=128, kernel_regularizer=regularizers.l2(0.1)))
#     # model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
#     model.add(Dense(1, init='normal'))
#     # Compile model
#     model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#     model.fit(x_train, y_train, epochs=500, verbose=2, batch_size=int(args["batch_size"]))
#     # 优化函数时会寻找返回值的最小值，所以返回-accuracy
#     accuracy = model.evaluate(x_test, y_test)[1]
#     return -accuracy


# def model(x_train, x_test, y_train, y_test):
#     x = Input(shape=(x_train.shape[1],))
#     h1 = Dense(512, activation='relu')(x)
#     h2 = Dense(256, activation='relu')(h1)
#     h3 = Dense(128, activation='relu')(h2)
#     y = Dense(1, init='normal')(h3)
#     model = Model(x, y)
#     model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#     model.fit(x_train, y_train, nb_epoch=200, verbose=1, batch_size=64)
#     accuracy = model.evaluate(x_test, y_test)
#     print(accuracy)
#     return model


# 由特征词生成数值向量
# vocab_size = 2000
# max_length = 23
# comment_onehot_array = label_onehot(comment_data, vocab_size, max_length)
# input_vec = comment_onehot_array

# # 标签生成，综合考虑五个因素，重要性依次为有用数, 感谢次数, 总分, 评论数, 读者数
# Y = np.zeros((row_user_data, 1))
# useful = user_data[:, 0]
# total = user_data[:, 2]
# comment = user_data[:, 3]
# reader = user_data[:, 4]
# # 滤除非数值的数据
# for i in range(row_user_data):
#     if type(comment[i]) != int:
#         comment[i] = 1
#     if type(useful[i]) != float:
#         useful[i] = 0
#     if type(total[i]) != float:
#         total[i] = 1
#     if type(reader[i]) != float:
#         reader[i] = 1
# # 取平均值作为判断条件，大于平均值的推荐，小于则不推荐
# useful_mean = useful.mean()
# total_mean = total.mean()
# comment_mean = comment.mean()
# reader_mean = reader.mean()
# # 小于所有平均值的帖子标签设置为0，其他的分别设置为3，4，6，8，9，重要程度依次递增
# for i in range(row_user_data):
#     if user_data[i, 0] > useful_mean:
#         Y[i] = 8
#         continue
#     if user_data[i, 1] != 0:
#         Y[i] = 9
#         continue
#     if user_data[i, 2] > total_mean:
#         Y[i] = 6
#         continue
#     if user_data[i, 3] > comment_mean:
#         Y[i] = 4
#         continue
#     if user_data[i, 4] > reader_mean:
#         Y[i] = 3
#         continue
# Y = preprocessing.minmax_scale(Y)
# y = to_categorical(Y, 6)


if __name__ == '__main__':

    # model1 = model(x_train, x_test, y_train, y_test)
    # pre = model1.predict(x_test)
    # global train_x, test_x, train_y, test_y
    # train_x, test_x, train_y, test_y = train_test_split(input_vec, Y, test_size=0.2)
    # space = {"node1": hp.choice("node1", range(256, 512)),
    #          "node2": hp.choice("node2", range(64, 256)),
    #          "batch_size": hp.choice("batch_size", range(16, 64))}
    # algo = partial(tpe.suggest, n_startup_jobs=10)
    # best = fmin(train_linear_model, space, algo=algo, max_evals=100)
    # encoder = load_model('encoder_label.h5')
    # x_train_encoded = encoder.predict(comment_data_matrix)
    x_label = to_categorical(x_label, num_classes)
    x_train, x_test, y_train, y_test = train_test_split(comment_data_matrix, x_label, test_size=0.2)

    model = fit_linear_model(X_train=x_train,
                             Y_train=y_train,
                             epoch=100,
                             node1=256,
                             node2=128,
                             batch_size=32)
    model.save('mysql_recommendation.h5')
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # predict = model.predict(test_x)

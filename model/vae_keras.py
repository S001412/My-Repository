# -*- coding: utf-8 -*-

# import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.layers import *
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from data_preprocessing import label_comment_matrix, comment_data_matrix
# from hyperopt import fmin, tpe, hp, partial

x_test = label_comment_matrix
x_train = comment_data_matrix
batch_size = 128
original_dim = x_train.shape[1]
# 隐变量取2维只是为了方便后面画图
latent_dim = 64
intermediate_dim = 512
intermediate_dim1 = 256
intermediate_dim2 = 128
epochs = 400
epsilon_std = 0.1
num_classes = 10


class Gaussian(Layer):
    """这是个简单的层，只为定义q(z|y)中的均值参数，每个类别配一个均值。
    输出也只是把这些均值输出，为后面计算loss准备，本身没有任何运算。
    """

    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        super(Gaussian, self).__init__(**kwargs)

    def build(self, input_shape):
        latent_dim = input_shape[-1]
        self.mean = self.add_weight(name='mean',
                                    shape=(self.num_classes, latent_dim),
                                    initializer='zeros')

    def call(self, inputs):
        z = inputs  # z.shape=(batch_size, latent_dim)
        z = K.expand_dims(z, 1)
        return z * 0 + K.expand_dims(self.mean, 0)

    def compute_output_shape(self, input_shape):
        return (None, self.num_classes, input_shape[-1])


def train_vae_model():
    # latent_dim = int(args["latent_dim"])
    x = Input(shape=(original_dim,))
    h1 = Dense(intermediate_dim, activation='relu')(x)
    h2 = Dense(intermediate_dim1, activation='relu')(h1)
    h3 = Dense(intermediate_dim2, activation='relu')(h2)
    # 算p(Z|X)的均值和方差
    z_mean = Dense(latent_dim)(h3)
    z_log_var = Dense(latent_dim)(h3)
    encoder = Model(x, z_mean)

    # 重参数技巧
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # 重参数层，相当于给输入加入噪声
    # z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # 解码层，也就是生成器部分
    decoder_input = Input(shape=(latent_dim,))
    decoder_h1 = Dense(intermediate_dim2, activation='relu')(decoder_input)
    decoder_h2 = Dense(intermediate_dim1, activation='relu')(decoder_h1)
    decoder_h3 = Dense(intermediate_dim, activation='relu')(decoder_h2)
    decoder_mean = Dense(original_dim, activation='sigmoid')(decoder_h3)
    # h_decoded = decoder_h(z)
    # x_decoded_mean = decoder_mean(h_decoded)
    decoder = Model(decoder_input, decoder_mean)

    z = Input(shape=(latent_dim,))
    y = Dense(intermediate_dim, activation='relu')(z)
    y = Dense(num_classes, activation='softmax')(y)

    # 隐变量分类器
    classfier = Model(z, y)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    x_recon = decoder(z)
    y = classfier(z)

    gaussian = Gaussian(num_classes)
    z_prior_mean = gaussian(z)

    # 建立模型
    vae = Model(x, [x_recon, z_prior_mean, y])

    # 下面一大通都是为了定义loss
    z_mean = K.expand_dims(z_mean, 1)
    z_log_var = K.expand_dims(z_log_var, 1)

    # 这是重构误差的权重，它的相反数就是重构方差，越大意味着方差越小。
    lamb = 5
    xent_loss = 0.5 * K.mean((x - x_recon)**2, 0)
    kl_loss = - 0.5 * (1 + z_log_var - K.square(z_mean - z_prior_mean) - K.exp(z_log_var))
    kl_loss = K.mean(K.batch_dot(K.expand_dims(y, 1), kl_loss), 0)
    cat_loss = K.mean(y * K.log(y + K.epsilon()), 0)
    vae_loss = lamb * K.sum(xent_loss) + K.sum(kl_loss) + K.sum(cat_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')

    vae.fit(x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, None),
            callbacks=[early_stopping])
    means = K.eval(gaussian.mean)
    # print(means.shape)
    # acc = vae.evaluate(x_test, batch_size, verbose=2)
    # print(acc)
    # encoder.save('encode_model1.h5')
    # classfier.save('classfier_model1.h5')

    return encoder, classfier


def encoder_model():
    x = Input(shape=(original_dim,))
    h1 = Dense(intermediate_dim, activation='relu')(x)
    h = Dense(intermediate_dim1, activation='relu')(h1)

    # 算p(Z|X)的均值和方差
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    # 构建encoder，然后观察各个数字在隐空间的分布
    encoder = Model(x, z_mean)

    # 重参数技巧
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # 重参数层，相当于给输入加入噪声
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # 解码层，也就是生成器部分
    decoder_input = Input(shape=(latent_dim,))
    decoder_h = Dense(intermediate_dim1, activation='relu')(decoder_input)
    decoder_h1 = Dense(intermediate_dim, activation='relu')(decoder_h)
    decoder_mean = Dense(original_dim, activation='sigmoid')(decoder_h1)
    decoder = Model(decoder_input, decoder_mean)
    x_decoder_mean = decoder(z)
    # h_decoded = decoder_h(z)
    # x_decoded_mean = decoder_mean(h_decoded)

    # 建立模型
    vae = Model(x, x_decoder_mean)

    # xent_loss是重构loss，kl_loss是KL loss
    xent_loss = original_dim * metrics.mean_squared_error(x, x_decoder_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)

    # add_loss是新增的方法，用于更灵活地添加各种loss
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')

    vae.fit(x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, None))
            # callbacks=[early_stopping])

    # # 构建生成器
    # decoder_input = Input(shape=(latent_dim,))
    # _h_decoded = decoder_h(decoder_input)
    # _x_decoded_mean = decoder_mean(_h_decoded)
    # generator = Model(decoder_input, _x_decoded_mean)
    return encoder


# # 观察隐变量的两个维度变化是如何影响输出结果的
# n = 15  # figure with 15x15 digits
# digit_size = 28
# figure = np.zeros((digit_size * n, digit_size * n))
#
# #用正态分布的分位数来构建隐变量对
# grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
# grid_y = norm.ppf(np.linspace(0.05, 0.95, n))


if __name__ == '__main__':
    encoder = encoder_model()
    encoder.save('encoder_label.h5')
    # x_train_encoded = encoder.predict(x_train)
    # x_test_encoded = encoder.predict(x_test)

    # space = {"latent_dim": hp.choice("node1", range(10, 128))}
    # algo = partial(tpe.suggest, n_startup_jobs=10)
    # best = fmin(train_vae_model, space, algo=algo, max_evals=100)
    # encoder, classfier = train_vae_model()
    # x_train_encoded = encoder.predict(x_train)
    # y_train_pred = classfier.predict(x_train_encoded).argmax(axis=1)
    # x_test_encoded = encoder.predict(x_test)
    # y_test_pred = classfier.predict(x_test_encoded).argmax(axis=1)
    # encoder = encoder_model()
    # x_train_encoded = encoder.predict(x_train)
    # x_test_encoded = encoder.predict(x_test)

    # y_test = encoder.predict(x_test)
    # from K_means import label_
    # import matplotlib.pyplot as plt
    # plt.scatter(y_test[:, 0], y_test[:, 1], c=np.squeeze(label_), s=3)
    # plt.colorbar()
    # plt.show()

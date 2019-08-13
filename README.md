任务一：基于机器学习的文本分类
实现基于logistic/softmax regression的文本分类

参考

文本分类
《神经网络与深度学习》 第2/3章
数据集：Classify the sentiment of sentences from the Rotten Tomatoes dataset

实现要求：NumPy

需要了解的知识点：

文本特征表示：Bag-of-Word，N-gram
分类器：logistic/softmax regression，损失函数、（随机）梯度下降、特征选择
数据集：训练集/验证集/测试集的划分
实验：

分析不同的特征、损失函数、学习率对最终分类性能的影响
shuffle 、batch、mini-batch
时间：两周

任务二：基于深度学习的文本分类
熟悉Pytorch，用Pytorch重写《任务一》，实现CNN、RNN的文本分类；

参考

https://pytorch.org/
Convolutional Neural Networks for Sentence Classification https://arxiv.org/abs/1408.5882
https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
word embedding 的方式初始化

随机embedding的初始化方式

用glove 预训练的embedding进行初始化 https://nlp.stanford.edu/projects/glove/

知识点：

CNN/RNN的特征抽取
词嵌入
Dropout
时间：两周

任务三：基于注意力机制的文本匹配
输入两个句子判断，判断它们之间的关系。参考ESIM（可以只用LSTM，忽略Tree-LSTM），用双向的注意力机制实现。

参考
《神经网络与深度学习》 第7章
Reasoning about Entailment with Neural Attention https://arxiv.org/pdf/1509.06664v1.pdf
Enhanced LSTM for Natural Language Inference https://arxiv.org/pdf/1609.06038v3.pdf
数据集：https://nlp.stanford.edu/projects/snli/
实现要求：Pytorch
知识点：
注意力机制
token2token attetnion
时间：两周
任务四：基于LSTM+CRF的序列标注
用LSTM+CRF来训练序列标注模型：以Named Entity Recognition为例。

参考
《神经网络与深度学习》 第6、11章
https://arxiv.org/pdf/1603.01354.pdf
https://arxiv.org/pdf/1603.01360.pdf
数据集：CONLL 2003，https://www.clips.uantwerpen.be/conll2003/ner/
实现要求：Pytorch
知识点：
评价指标：precision、recall、F1
无向图模型、CRF
时间：两周
任务五：基于神经网络的语言模型
用LSTM、GRU来训练字符级的语言模型，计算困惑度

参考
《神经网络与深度学习》 第6、15章
数据集：poetryFromTang.txt
实现要求：Pytorch
知识点：
语言模型：困惑度等
文本生成
时间：两周

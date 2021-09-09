import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,XG_NB
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score,recall_score,f1_score,auc,roc_curve
import math

from itertools import cycle
from scipy import interpolate as interp

#导入数据
df1 = pd.read_csv(r'C:\Users\DaBiao\Downloads\pycharm\data_test\cluster_data.csv', usecols=[1, 2, 3])
df2 = pd.read_csv(r'C:\Users\DaBiao\Downloads\pycharm\data_test\cluster_data.csv', usecols=[4])
X = df1.values
y1 = df2.values
y = y1[:, 0]
class NaiveBayes():

    def __init__(self):
        self.parameters = []  # 保存每个特征针对每个类的均值和方差
        self.y = None
        self.classes = None

    def fit(self, X, y):
        self.y = y
        self.classes = np.unique(y)  # 类别
        # 计算每个特征针对每个类的均值和方差
        for i, c in enumerate(self.classes):
            # 选择类别为c的X
            X_where_c = X[np.where(self.y == c)]
            self.parameters.append([])
            # 添加均值与方差
            for col in X_where_c.T:
                parameters = {"mean": col.mean(), "var": col.var()}
                self.parameters[i].append(parameters)

    def _calculate_prior(self, c):
        """
        先验函数。
        """
        frequency = np.mean(self.y == c)
        return frequency

    def _calculate_likelihood(self, mean, var, X):
        """
        似然函数。
        """
        # 高斯概率
        eps = 1e-4  # 防止除数为0
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
        exponent = math.exp(-(math.pow(X - mean, 2) / (2 * var + eps)))
        return coeff * exponent

    def _calculate_probabilities(self, X):
        posteriors = []
        for i, c in enumerate(self.classes):
            posterior = self._calculate_prior(c)
            for feature_value, params in zip(X, self.parameters[i]):
                # 独立性假设
                # P(x1,x2|Y) = P(x1|Y)*P(x2|Y)
                likelihood = self._calculate_likelihood(params["mean"], params["var"], feature_value)
                posterior *= likelihood
            posteriors.append(posterior)
        # 返回具有最大后验概率的类别
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        y_pred = [self._calculate_probabilities(sample) for sample in X]
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred, axis=0) / len(y)
        return accuracy


score_all=[]
# 进行交叉运算
for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model =GaussianNB ()
    model.fit(X_test, y_test)
    y_prec = model.predict(X_train)
    score=f1_score(y_train,y_prec,average="micro")
    score_all.append(score)
print(score_all)
#准确率
# xg_score=[92.30769230769231, 88.88888888888888, 92.73504273504274, 90.5982905982906, 92.30769230769231]
# GaussianNB_s=[87.60683760683761, 74.35897435897436, 87.17948717948718, 81.62393162393162, 82.47863247863247]
# AOED=[67.60683760683761,54.35897435897436,66.75213675213675,72.30769230769231,57.17948717948718]
# navi_bye=[77.35042735042735, 86.75213675213675, 87.60683760683761, 86.32478632478633, 84.18803418803419]
# ave_prec=[91.36752136752136, 82.64957264957266 ,84.44444444444444 ,63.64102564102565]
#
#
# x=np.arange(5)
# plt.plot(x,xg_score,label='XG-NB')
# plt.plot(x,GaussianNB_s,label='GaussianNB')
# plt.plot(x,AOED,label='AODE')
# plt.plot(x,navi_bye,label='NB')
# plt.legend(loc="lower right")
# ax = plt.gca()
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# plt.title("不同贝叶斯算法的准确率")
# ax.set_ylim(0, 100)
# ax.set_xticks([0,1,2,3,4])
# plt.xlabel("次数")
# plt.ylabel("准确率%")

# # #召回率
# navi=[77.33479331781473, 90.49140049140049, 87.50080781585601, 83.09091744082298, 74.29905867405868]
# GAUSS=[91.02564102564102, 84.18803418803419, 87.17948717948718, 91.02564102564102, 90.17094017094017]
# XG_NB=[87.17948717948718, 90.17094017094017, 89.31623931623932, 88.03418803418803, 90.17094017094017]
# aoed=[76.78549705662175, 62.08493768004335, 64.07211209842789, 68.91451035615795, 59.85463720395809]
# ave_recall=[88.97435897435898, 88.71794871794873,82.54339554799058 ,66.34233887904182]
#
# x=np.arange(5)
# plt.plot(x,XG_NB,label='XG-NB')
# plt.plot(x,GAUSS,label='GaussianNB')
# plt.plot(x,aoed,label='AODE')
# plt.plot(x,navi,label='NB')
# plt.legend(loc="lower right")
# ax = plt.gca()
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# plt.title("不同贝叶斯算法的召回率")
# ax.set_ylim(0, 100)
# ax.set_xticks([0,1,2,3,4])
# plt.xlabel("次数")
# plt.ylabel("召回率%")

#f1
f1_xg=[87.56035631035631, 95.24867612418751, 88.74782585028339, 86.91280845825958, 83.08597657166109]
f1_gaos=[89.74358974358975, 91.02564102564102, 92.44444444444444, 77.35042735042735, 89.74358974358975]
f1_navi=[79.11389627896128, 73.39517592880525, 79.53936275364846, 87.73553418591586, 85.83453944716529]
f1_aoed=[69.86471773929983,73.167408227066,75.11389627896128,65.24867612418751,70.4642245217092]
ave_f1=[88.31112866294959 ,88.06153846153848 ,81.12370171889923 ,70.77178457824476]
s1=sum(f1_xg)/5
s2=sum(f1_gaos)/5
s3=sum(f1_navi)/5
s4=sum(f1_aoed)/5
print(s1,s2,s3,s4)

x=np.arange(5)
plt.plot(x,f1_xg,label='XG-NB')
plt.plot(x,f1_gaos,label='GaussianNB')
plt.plot(x,f1_aoed,label='AODE')
plt.plot(x,f1_navi,label='NB')
plt.legend(loc="lower right")
ax = plt.gca()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.title("不同贝叶斯算法的F1")
ax.set_ylim(0, 100)
ax.set_xticks([0,1,2,3,4])
plt.xlabel("次数")
plt.ylabel("F1值%")
plt.show()
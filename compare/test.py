import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from numpy import interp
import pandas as pd
from sklearn.naive_bayes import GaussianNB,XG_NB,BernoulliNB
#导入数据
df1 = pd.read_csv(r'C:\Users\DaBiao\Downloads\pycharm\data_test\cluster_data.csv', usecols=[1, 2, 3])
df2 = pd.read_csv(r'C:\Users\DaBiao\Downloads\pycharm\data_test\cluster_data.csv', usecols=[4])
X = df1.values
y1 = df2.values
y = y1[:, 0]

# 将标签二值化
y = label_binarize(y, classes=[0, 1, 2,3])

# 设置种类
n_classes = y.shape[1]

# 训练模型并预测
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.8,random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(BernoulliNB())
classifier.fit(X_train, y_train)
y_score=classifier.predict(X_test)
# 计算每一类的ROC
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#
# # Compute micro-average ROC curve and ROC area（方法二）
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
# # Compute macro-average ROC curve and ROC area（方法一）
# # First aggregate all false positive rates
# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#
# # Then interpolate all ROC curves at this points
# mean_tpr = np.zeros_like(all_fpr)
# for i in range(n_classes):
#     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#
# # Finally average it and compute AUC
# mean_tpr /= n_classes
# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
fpr1={0: np.array([0., 0.15, 1.]), 1: np.array([0.        , 0.06432749, 1.        ]), 2: np.array([0.        , 0.34479167, 1.        ]), 3: np.array([0., 1.]), 'micro': np.array([0.        , 0.07702523, 1.        ]), 'macro': np.array([0.        , 0.06732749, 0.204479167, 1.        ])}
tpr1={0: np.array([0.        , 0.6739726, 1.        ]), 1: np.array([0.   , 0.59, 1.   ]), 2: np.array([0.        , 0.58881356, 1.        ]), 3: np.array([0., 1.]), 'micro': np.array([0.        , 0.20318725, 1.        ]), 'macro': np.array([0.00684932, 0.64520271, 0.70312332, 1.        ])}
roc_auc1={0: 0.6136986301369864, 1: 0.5353362573099415, 2: 0.7420109463276836, 3: 0.5, 'micro': 0.5630810092961488, 'macro': 0.6652614584436528}

# Plot all ROC curves
lw=2
plt.figure()
plt.plot(fpr1["micro"], tpr1["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc1["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr1["macro"], tpr1["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc1["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr1[i], tpr1[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc1[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AODE的ROC曲线')
plt.legend(loc="lower right")
plt.show()

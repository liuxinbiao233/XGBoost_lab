import pandas as pd
import xgboost
from xgboost import plot_importance
from matplotlib import pyplot as plt
from xgboost import plot_tree
import xgboost as xgb
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
# load data
DATA = pd.read_csv(r'C:\Users\DaBiao\Downloads\pycharm\data_test\cluster_data.csv', usecols=[1,2,3])
# split data into X and y
X = DATA.values
DATA1 = pd.read_csv(r'C:\Users\DaBiao\Downloads\pycharm\data_test\cluster_data.csv', usecols=[4])
Y = DATA1.values


# # # # split data into train and test sets
# seed = 7
# test_size = 0.2
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# dtest=xgb.DMatrix(X_test)
#
#
# params1 = {'learning_rate': 0.1,
#                    'max_depth': 6,  # 构建树的深度，越大越容易过拟合
#                    'num_boost_round': 20,
#                    'min_child_weight': 1,
#                    'max_leaf_nodes': 4,
#                    'objective': 'multi:softprob',  # 多分类的问题
#                    'random_state': 7,
#                    'num_class': 4,  # 类别数，与 multisoftmax 并用
#                    'eta': 0.1  # 为了防止过拟合，更新过程中用到的收缩步长。eta通过缩减特征 的权重使提升计算过程更加保守。缺省值为0.3，取值范围为：[0,1]
#                    }
# eval_rst = {}
# model = xgb.train(params1,X_train, y_train,early_stopping_rounds=5, evals_result=eval_rst, verbose_eval=True)
#
# y_pred=model.predict(xgb.DMatrix(X_test))
#
# model.save_model('testXGboostClass.model')  # 保存训练模型
#
# yprob = np.argmax(y_pred, axis=1)  # return the index of the biggest pro
#
# predictions = [round(value) for value in yprob]
#
# # evaluate predictions
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
# y_pred_leaf=model.predict(dtest,pred_leaf=True)
#
# #特征重要度
# ypred_contribs = model.predict(dtest, pred_contribs=True) #最后一列是bias, 前面的四列分别是每个特征对最后打分的影响因子
# # 显示重要特征
# plot_importance(model)
# plt.show()
# # print('booster attributes:', model.attributes())
#



from xgboost import XGBClassifier
import xgboost

# model = XGBClassifier()
# model.fit(X, Y)
# # 如果输入是没有表头的array,会自动以f1,f2开始,需要更换表头
# # 画树结构图的时候也需要替换表头
# model.get_booster().feature_names = DATA.axes[1].values.tolist()
# # max_num_features指定排名最靠前的多少特征
# # height=0.2指定柱状图每个柱子的粗细,默认是0.2
# # importance_type='weight'默认是用特征子树中的出现次数(被选择次数),还有"gain"和"cover"
# xgboost.plot_importance(model, max_num_features=5)
# ax = plt.gca()
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# plt.title("属性特征重要度")
# ax.set_xlabel("F值")
# ax.set_ylabel("属性名称")
# scr = xgboost.to_graphviz(model, num_trees=-6,leaf_node_params={'shape': 'plaintext'})
# # src.format = "jpg"
#
# plt.show()





from xgboost import XGBClassifier
#进行交叉验证
import xgboost as xgt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
#
#
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3,random_state=1)
_train_matrix = xgt.DMatrix(data=train_X, label=train_Y,)
_validate_matrix = xgt.DMatrix(data=test_X, label=test_Y,)
#
# params = {
#     'booster': 'gbtree',
#     'eta': 0.3,
#     'max_depth': 5,
#     'tree_method': 'exact',
#     'objective': 'multi:softprob',  # 多分类的问题
#     'random_state': 7,
#     'num_class': 4,
#
# }
# eval_rst = {}
# booster = xgt.train(params, _train_matrix, num_boost_round=100,
#                     evals=([(_train_matrix, 'valid1'), (_validate_matrix, 'valid2')]),
#                     early_stopping_rounds=5, evals_result=eval_rst, verbose_eval=True)
# ## 训练输出
# # Multiple eval metrics have been passed: 'valid2-auc' will be used for early stopping.
# # Will train until valid2-auc hasn't improved in 5 rounds.
# # [0]   valid1-logloss:0.685684 valid1-error:0.042857   valid1-auc:0.980816 valid2-logloss:0.685749 valid2-error:0.066667   valid2-auc:0.933333
# # ...
# # Stopping. Best iteration:
# # [1]   valid1-logloss:0.678149 valid1-error:0.042857   valid1-auc:0.99551  valid2-logloss:0.677882 valid2-error:0.066667   valid2-auc:0.966667
# print('booster attributes:', booster.attributes())
# # booster attributes: {'best_iteration': '1', 'best_msg': '[1]\tvalid1-logloss:0.678149\tvalid1-error:0.042857\tvalid1-auc:0.99551\tvalid2-logloss:0.677882\tvalid2-error:0.066667\tvalid2-auc:0.966667', 'best_score': '0.966667'}
# print('fscore:', booster.get_fscore())
# # fscore: {'Petal Length': 8, 'Petal Width': 7}
# print('eval_rst:', eval_rst)
# # eval_rst: {'valid1': {'logloss': [0.685684, 0.678149, 0.671075, 0.663787, 0.656948, 0.649895], 'error': [0.042857, 0.042857, 0.042857, 0.042857, 0.042857, 0.042857], 'auc': [0.980816, 0.99551, 0.99551, 0.99551, 0.99551, 0.99551]}, 'valid2': {'logloss': [0.685749, 0.677882, 0.670747, 0.663147, 0.656263, 0.648916], 'error': [0.066667, 0.066667, 0.066667, 0.066667, 0.066667, 0.066667], 'auc': [0.933333, 0.966667, 0.966667, 0.966667, 0.966667, 0.966667]}}
# epochs = len(eval_rst['valid1']['mlogloss'])
# x_axis = range(0, epochs)
# # plot log loss
# fig, ax = plt.subplots()
# ax.plot(x_axis, eval_rst['valid1']['mlogloss'], label='Train')
# ax.plot(x_axis, eval_rst['valid2']['mlogloss'], label='Test')
# ax.legend()
# plt.ylabel('Log Loss')
# plt.title('XGBoost Log Loss')
# plt.show()

#

#
# #选择多少个测试器。找到一个标准
# train_X, test_X, train_Y, test_Y = train_test_split(X, Y)
# model = XGBClassifier(n_estimators=30,learning_rate=0.36,gamma=0,max_depth=2,min_child_weight=2)
# eval_set = [(train_X, train_Y), (test_X, test_Y)]
# model.fit(train_X, train_Y, eval_set=eval_set, verbose=True)
# # make predictions for test data
# y_pred = model.predict(test_X)
# predictions = [round(value) for value in y_pred]
# # evaluate predictions
# accuracy = accuracy_score(test_Y, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
# # retrieve performance metrics
# results = model.evals_result()
# epochs = len(results['validation_0']['mlogloss'])
# x_axis = range(0, epochs)
# # plot log loss
# fig, ax = plt.subplots()
# ax.plot(x_axis, results['validation_0']['mlogloss'],)
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# plt.ylabel('损失函数值')
# plt.title('XGBoost 损失函数')
# plt.xlabel("迭代次数")
# plt.show()


#
# #网格验证
# params = {
#     # 'eta': list(np.linspace(0.01,0.5,100)),
#     'max_depth': list(range(2,15,1)),
#     # 'n_estimators':list(range(20,70,10)),
#     'gamma':list(np.linspace(0,0,5,10)),
#     'min_child_weight':list(range(1,9,1)),
# }
# cf1=xgb.XGBClassifier(n_estimators=30,learning_rate=0.36)
# grid=GridSearchCV(cf1,params,cv=3,scoring='neg_log_loss',n_jobs=-1)
# test_Y1=test_Y.ravel()
# grid.fit(test_X,test_Y1)
#
#
# best_estimtor=grid.best_estimator_
# print(best_estimtor)
# print('------')

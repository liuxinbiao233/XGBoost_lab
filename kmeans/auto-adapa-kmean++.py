import numpy as np
import random
import math
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
import numbers
import pandas as pd
import matplotlib.pyplot as plt

def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)
def distance(point1, point2):
    return np.sqrt(np.sum(np.square(point1 - point2), axis=1))

def InitCentroids(X, K):
    n = np.size(X, 0)
    rands_index = np.array(random.sample(range(1, n), K))
    centriod = X[rands_index, :]
    return centriod


def findClosestCentroids(X, w, centroids):
    K = np.size(centroids, 0)
    idx = np.zeros((np.size(X, 0)), dtype=int)
    n = X.shape[0]  # n 表示样本个数
    for i in range(n):
        subs = centroids - X[i, :]
        dimension2 = np.power(subs, 2)
        w_dimension2 = np.multiply(w, dimension2)
        w_distance2 = np.sum(w_dimension2, axis=1)
        if math.isnan(w_distance2.sum()) or math.isinf(w_distance2.sum()):
            w_distance2 = np.zeros(K)
            # print 'the situation that w_distance2 is nan or inf'
        idx[i] = np.where(w_distance2 == w_distance2.min())[0][0]
    return idx


def computeCentroids(X, idx, K):
    n, m = X.shape
    cc=check_random_state(4) #4 13 16 20 27 34
    centers = np.empty((4, m))
    # n_local_trials是每次选择候选点个数
    n_local_trials = None
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(4))

    # 第一个随机点
    # center_id = self.random_state.randint(n_samples)
    center_id = cc.randint(n)
    centers[0] = X[center_id]

    # closest_dist_sq是每个样本，到所有中心点最近距离
    # 假设现在有3个中心点，closest_dist_sq = [min(样本1到3个中心距离),min(样本2到3个中心距离),...min(样本n到3个中心距离)]
    closest_dist_sq = distance(centers[0, np.newaxis], X)

    # current_pot所有最短距离的和
    current_pot = closest_dist_sq.sum()

    for c in range(K):
        index = np.where(idx == c)[0]
        # 选出n_local_trials随机址，并映射到current_pot的长度
        rand_vals = cc.random_sample(n_local_trials) * current_pot
        # np.cumsum([1,2,3,4]) = [1, 3, 6, 10]，就是累加当前索引前面的值
        # np.searchsorted搜索随机出的rand_vals落在np.cumsum(closest_dist_sq)中的位置。
        # candidate_ids候选节点的索引
        candidate_ids = np.searchsorted(np.cumsum(closest_dist_sq), rand_vals)

        # best_candidate最好的候选节点
        # best_pot最好的候选节点计算出的距离和
        # best_dist_sq最好的候选节点计算出的距离列表
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # 计算每个样本到候选节点的欧式距离
            distance_to_candidate = distance(X[candidate_ids[trial], np.newaxis], X)

            # 计算每个候选节点的距离序列new_dist_sq， 距离总和new_pot
            new_dist_sq = np.minimum(closest_dist_sq, distance_to_candidate)
            new_pot = new_dist_sq.sum()

            # 选择最小的new_pot
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        centers[c] = X[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers



def computeWeight(X, centroid, idx, K, belta):
    n, m = X.shape
    weight = np.zeros((1, m), dtype=float)
    D = np.zeros((1, m), dtype=float)
    for k in range(K):
        index = np.where(idx == k)[0]
        temp = X[index, :]  # 取第k个簇的所有样本
        distance2 = np.power((temp - centroid[k, :]), 2)  # ? by m
        D = D + np.sum(distance2, axis=0)
    e = 1 / float(belta - 1)
    for j in range(m):
        temp = D[0][j] / D[0]
        weight[0][j] = 1 / np.sum((np.power(temp, e)), axis=0)
    return weight


def costFunction(X, K, centroids, idx, w, belta):
    n, m = X.shape
    D = np.zeros((1, m), dtype=float)
    for k in range(K):
        index = np.where(idx == k)[0]
        temp = X[index, :]
        distance2 = np.power((temp - centroids[k, :]), 2)  # ? by m
        D = D + np.sum(distance2, axis=0)
    cost = np.sum(w ** belta * D)
    return cost


def isConvergence(costF, max_iter):
    if math.isnan(np.sum(costF)):
        return False
    index = np.size(costF)
    for i in range(index - 1):
        if costF[i] < costF[i + 1]:
            return False
    if index >= max_iter:
        return True
    elif costF[index - 1] == costF[index - 2] == costF[index - 3]:
        return True
    return 'continue'


def wkmeans(X, K, belta, max_iter):
    n, m = X.shape
    costF = []
    r = np.random.rand(1, m)
    w = np.divide(r, r.sum())
    centroids = InitCentroids(X, K)
    for i in range(max_iter):
        idx = findClosestCentroids(X, w, centroids)
        centroids = computeCentroids(X, idx, K)
        w = computeWeight(X, centroids, idx, K, belta)
        c = costFunction(X, K, centroids, idx, w, belta)
        costF.append(round(c, 4))
        if i < 2:
            continue
        flag = isConvergence(costF, max_iter)
        if flag == 'continue':
            continue
        elif flag:
            best_labels = idx
            best_centers = centroids
            isConverge = True
            return isConverge, best_labels, best_centers, costF
        else:
            isConverge = False
            return isConverge, None, None, costF


class WKMeans:

    def __init__(self, n_clusters=3, max_iter=20, belta=7.0,n_init = 30, tol = 1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.belta = belta
        self.n_init = n_init
        self.tol = tol

    def fit(self, X):
        self.isConverge, self.best_labels, self.best_centers, self.cost = wkmeans(
            X=X, K=self.n_clusters, max_iter=self.max_iter, belta=self.belta
        )
        return self

    def fit_predict(self, X, y=None):
        if self.fit(X).isConverge:
            return self.best_labels
        else:
            return 'Not convergence with current parameter ' \
                   'or centroids,Please try again'

    def get_params(self):
        return self.isConverge, self.n_clusters, self.belta, 'WKME'

    def get_cost(self):
        return self.cost


def load_data():
    df1 = pd.read_csv(r'C:\Users\DaBiao\Downloads\pycharm\data_test\cluster_data.csv', usecols=[1, 2, 3])
    df2 = pd.read_csv(r'C:\Users\DaBiao\Downloads\pycharm\data_test\cluster_data.csv', usecols=[4])
    x = df1.values
    y1 = df2.values
    y=y1[:,0]
    return x, y


if __name__ == '__main__':
    x, y = load_data()

    model = KMeans(n_clusters=4)
    model.fit(x)
    y_pred = model.predict(x)
    nmi = normalized_mutual_info_score(y, y_pred)
    print("NMI by sklearn: ", nmi)

    model1 = WKMeans(n_clusters=4, belta=3)
    while True:
        y_pred = model1.fit_predict(x)
        if model1.isConverge == True:
            nmi = normalized_mutual_info_score(y, y_pred)
            print("NMI by wkmeans: ", nmi)
            break

#     进行画图操作
    times = pd.read_csv(r'C:\Users\DaBiao\Downloads\pycharm\data_test\cluster_data.csv', usecols=[3])
    a = pd.read_csv(r'C:\Users\DaBiao\Downloads\pycharm\data_test\cluster_data.csv', usecols=[2])
    v = pd.read_csv(r'C:\Users\DaBiao\Downloads\pycharm\data_test\cluster_data.csv', usecols=[1])
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    ax = plt.subplot(projection='3d')
    ax.set_xlabel("平均速度")
    ax.set_ylabel("平均绝对加速度")
    ax.set_zlabel("怠速时间比")

    colors = ['#4EACC5', '#4E9A06', 'm', '#FF9C34']

    ax.scatter(v.values[y_pred == 0][:, 0], a.values[y_pred == 0][:, 0],times.values[y_pred == 0][:, 0], c='#4EACC5',
               marker='.', alpha=0.7, s=10)
    ax.scatter(v.values[y_pred== 1][:, 0], a.values[y_pred == 1][:, 0],times.values[y_pred == 1][:, 0], c='#4E9A06',
               marker='.', alpha=0.7, s=10)
    ax.scatter(v.values[y_pred == 2][:, 0], a.values[y_pred == 2][:, 0],times.values[y_pred== 2][:, 0], c='#FF9C34',
               marker='.', alpha=0.7, s=10)
    ax.scatter(v.values[y_pred == 3][:, 0], a.values[y_pred == 3][:, 0],times.values[y_pred == 3][:, 0], c='m',
               marker='.', alpha=0.7, s=10)

    ax.scatter(model1.best_centers[:, 0], model1.best_centers[:, 1], model1.best_centers[:, 2], c='r', s=35,marker='*')
    plt.show()


# result:
# NMI by sklearn:  0.7581756800057784
# NMI by wkmeans:  0.8130427037493443


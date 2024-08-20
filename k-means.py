import numpy as np
import matplotlib.pyplot as plt
import time

# 加载数据
def loadDataSet(fileName):
    data = np.loadtxt(fileName,delimiter='\t')
    return data

# 欧氏距离计算
def distEclud(x,y):
    return np.sqrt(np.sum((x-y)**2))  # 计算欧氏距离

# randCent 函数通过从数据集中随机选择 k 个数据点来初始化 k 个中心点
def randCent(dataSet,k):
    m,n = dataSet.shape  # m行n列
    centroids = np.zeros((k,n))  # k行n列的全0矩阵
    for i in range(k):
        index = int(np.random.uniform(0,m)) # 随机产生一个0到m之间的整数
        centroids[i,:] = dataSet[index,:]  # 列是所有的列
    return centroids

'''
初始化： 它使用 randCent 函数初始化 k 个中心点，并准备一个 clusterAssment 矩阵来存储每个数据点的聚类索引和误差
分配步骤： 根据欧氏距离将每个数据点分配给最近的中心点。 更新步骤： 通过计算分配给每个簇的点的平均值来更新中心点
收敛检查： 循环继续进行，直到没有点改变聚类(即 clusterChange 变为 False)
'''
def KMeans(dataSet,k):

    m = np.shape(dataSet)[0]  #行的数目
    clusterAssment = np.mat(np.zeros((m,2)))  # m行2列的全0矩阵，即一开始均所属第0个质心
    # 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差（距离的平方）
    clusterChange = True

    # 第1步 初始化centroids 中心
    centroids = randCent(dataSet,k)
    while clusterChange:
        clusterChange = False

        # 遍历所有的样本点（行数）
        for i in range(m):
            minDist = 100000.0  # 初始化
            minIndex = -1

            #第2步 遍历所有的质心，找出最近的质心
            for j in range(k):
                # 计算该样本到质心的欧式距离
                distance = distEclud(centroids[j,:],dataSet[i,:])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 第 3 步：更新每一行样本所属的簇
            if clusterAssment[i,0] != minIndex:
                clusterChange = True
                clusterAssment[i,:] = minIndex,minDist**2
        #第 4 步：更新质心
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:,0].A == j)]  # 获取簇类所有的点
            # .A: Converts the matrix to an array for easier comparison.
            centroids[j,:] = np.mean(pointsInCluster,axis=0)   # 对矩阵的行求均值 axis=0跨行

    print("Congratulations,cluster complete!")
    return centroids,clusterAssment

def showCluster(dataSet,k,centroids,clusterAssment):  # 展示簇
    m,n = dataSet.shape
    if n != 2:
        print("数据不是二维的")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("k值太大了")
        return 1

    # 绘制所有的样本
    for i in range(m):
        markIndex = int(clusterAssment[i,0])
        plt.plot(dataSet[i,0], dataSet[i,1], mark[markIndex])
        print(f'成功绘制第{i}个样本点, 标记是{mark[markIndex]}')
        time.sleep(1)

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 绘制质心
    for i in range(k):
        plt.plot(centroids[i,0], centroids[i,1], mark[i])
        print(f'成功绘制第{i}个质心, 标记是{mark[i]}')
        time.sleep(1)

    plt.show()


dataSet = loadDataSet("data.txt")
k = 4
centroids,clusterAssment = KMeans(dataSet,k)

showCluster(dataSet,k,centroids,clusterAssment)
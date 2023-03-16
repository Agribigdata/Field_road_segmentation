import loaddata as lddata
from sklearn.cluster import DBSCAN
import parameter as parm
x = lddata.clean_x
y = lddata.clean_y
# 设置聚类函数所需的数据格式
dataSet = [[0 for j in range(2)] for h in range(len(x))]
for j in range(len(x)):
    dataSet[j][0] = x[j]
    dataSet[j][1] = y[j]
# dbscan聚类方式 clustering为聚类结果
clustering = DBSCAN(eps=parm.eps, min_samples=parm.min_samples).fit(dataSet)
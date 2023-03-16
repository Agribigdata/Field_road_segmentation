'''
create by zxq on 2020/11/1
'''
from sklearn import linear_model
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
import math
import random
from sklearn.metrics import precision_score, recall_score, f1_score
import multiprocessing
from time import time
from datetime import datetime, date
import heapq
import getaveragespeed as avespeed
import gettime as gtime

#求取每个轨迹段的方向、速度、斜率、时间、距离
def getfirstsegment(q, clusteringcopy, cleanx, cleany, direct, origindatatime, origindataid, origindatax, origindatay,
                    origindataspeed):
    dir = direct
    clustering = clusteringcopy
    # 统计方向
    x = cleanx
    y = cleany
    # 计算分段轨迹
    allsegment = []
    onesegment = []
    onesegment.append(0)
    flag = 0
    for i in range(1, len(clustering.labels_)):
        if clustering.labels_[i] == clustering.labels_[i - 1]:
            onesegment.append(i)
            flag = 1
        else:
            allsegment.append(onesegment)
            onesegment = []
            onesegment.append(i)
            flag = 0
    if flag == 1:
        allsegment.append(onesegment)

    alldir = []#方向
    allspeed = []#速度
    allerrorpoint = []#异常点数
    allslope = []#斜率

    #统计方向
    for i in range(len(allsegment)):
        dircount = [0 for h in range(37)]
        for j in range(len(allsegment[i])):
            if dir[allsegment[i][j]] <= 360:
                dircun = int(dir[allsegment[i][j]] / 10)
                dircount[dircun] = dircount[dircun] + 1
        numdir0 = 0
        dirsave = []
        for j in range(len(dircount)):
            if dircount[j] > 0:
                numdir0 = numdir0 + 1
        if numdir0 >= 3:
            re1 = map(dircount.index, heapq.nlargest(1, dircount))
            a = list(re1)
            dirsave.append(a[0])
            dircount[a[0]] = -1
            re1 = map(dircount.index, heapq.nlargest(1, dircount))
            a = list(re1)
            dirsave.append(a[0])
            dircount[a[0]] = -1
            re1 = map(dircount.index, heapq.nlargest(1, dircount))
            a = list(re1)
            dirsave.append(a[0])
        if numdir0 == 2:
            re1 = map(dircount.index, heapq.nlargest(1, dircount))
            a = list(re1)
            dirsave.append(a[0])
            dircount[a[0]] = -1
            re1 = map(dircount.index, heapq.nlargest(1, dircount))
            a = list(re1)
            dirsave.append(a[0])
        if numdir0 == 1:
            re1 = map(dircount.index, heapq.nlargest(1, dircount))
            dirsave = list(re1)
        alldir.append(dirsave)

        # 获取原始数据中的平具速度错误点数
        firstid = origindataid[allsegment[i][0]]
        finalid = origindataid[allsegment[i][len(allsegment[i]) - 1]]
        counttime, errornum = gtime.gettime(q, origindataid, origindatatime, firstid, finalid)
        #统计该轨迹段的平均速度
        averspeed = avespeed.getaverspeed(q, origindataid, origindataspeed, firstid, finalid)
        allspeed.append(averspeed)
        allerrorpoint.append(errornum)
        # 利用最小二乘法拟合来获取每段轨迹的斜率
        a = []
        b = []
        for j in range(len(allsegment[i])):
            a.append([x[allsegment[i][j]]])
            b.append(y[allsegment[i][j]])
        a = np.array(a)
        b = np.array(b)
        reg = linear_model.LinearRegression()
        reg.fit(a, b)
        reg.coef_ = list(reg.coef_)
        allslope.append(reg.coef_[0])

    return allsegment, alldir, allspeed, allerrorpoint, allslope

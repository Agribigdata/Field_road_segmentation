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


# 第四步算法：对多个距离相同，长度相似，方向相似的道路进行修正
def getsameroad(clusteringcopy, q, allsegment, alldir, allspeed, allerrorpoint, allslope, speedthreshold, dirthreshold,
                dirnumthreshold):
    # 对多个距离相同，长度相似，方向相似的道路进行修正
    clustering = clusteringcopy
    # 统计方向
    # 使用长距离segment计算
    newallsegment = []
    newalldir = []
    newsegmentlabal = []
    newallspeed = []
    newallerrorpoint = []
    newallslope = []
    for i in range(len(allsegment)):
        if len(allsegment[i]) >= 20:
            newallsegment.append(allsegment[i])
            newalldir.append(alldir[i])
            newallspeed.append(allspeed[i])
            newallerrorpoint.append(allerrorpoint[i])
            newallslope.append(allslope[i])
    for i in range(len(newallsegment)):
        numtian = 0 #统计某段轨迹中为田的点数 ，做轨迹标签使用
        numroad = 0 #统计某段轨迹中为路的点数
        for j in range(len(newallsegment[i])):
            if clustering.labels_[newallsegment[i][j]] == 0:
                numroad = numroad + 1
            else:
                numtian = numtian + 1
        if numtian >= numroad:
            newsegmentlabal.append(1)
        else:
            newsegmentlabal.append(0)
    for i in range(1, len(newallsegment) - 1):
        #0标签代表道路
        #dirthreshold、dirnumthreshold、speedthreshold为阈值
        if newsegmentlabal[i] == 0 and newsegmentlabal[i - 1] == 0 and newsegmentlabal[i + 1] == 0:
            onespeed = newallspeed[i - 1]
            twospeed = newallspeed[i]
            threespeed = newallspeed[i + 1]
            errandspeed = abs(onespeed - twospeed) + abs(threespeed - twospeed)

            if errandspeed <= speedthreshold:
                roadsamedir = 0
                for j in newalldir[i]:
                    for h1 in newalldir[i - 1]:
                        if j >= h1 - dirthreshold and j <= h1 + dirthreshold:
                            roadsamedir += 1
                    for h1 in newalldir[i + 1]:
                        if j >= h1 - dirthreshold and j <= h1 + dirthreshold:
                            roadsamedir += 1
                if roadsamedir >= dirnumthreshold:
                    for j in range(len(newallsegment[i - 1])):
                        clustering.labels_[newallsegment[i - 1][j]] = 1
                    for j in range(len(newallsegment[i])):
                        clustering.labels_[newallsegment[i][j]] = 1
                    for j in range(len(newallsegment[i + 1])):
                        clustering.labels_[newallsegment[i + 1][j]] = 1

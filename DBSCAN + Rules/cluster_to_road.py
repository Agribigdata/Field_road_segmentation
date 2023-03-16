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
import get_score as one
from time import time
from datetime import datetime, date
import heapq

def cluster2road(q, jishu, bilv, chazhi, dirnum, direct, alltian, clusteringcopy):

    clustering = clusteringcopy
    #统计轨迹中每块农田的具体方向分布
    segment_field = [0 for j in range(len(clustering.labels_))]
    for j in range(len(clustering.labels_)):
        segment_field[j] = []
    for j in range(len(clustering.labels_) - 1):
        if clustering.labels_[j] >= 0:
            segment_field[clustering.labels_[j]].append(j)
    dir = direct
    for j in range(len(segment_field)):
        if len(segment_field[j]) > 0:
            count0 = 0
            dircount = [0 for j in range(37)]
            for h in segment_field[j]:
                if dir[h] <= 360:
                    dircun = int(dir[h] / 10)
                    dircount[dircun] = dircount[dircun] + 1
            for h in range(len(dircount)):
                if dircount[h] <= jishu:
                    count0 = count0 + 1
            if count0 >= bilv:
                for m in range(len(segment_field[j])):
                    clustering.labels_[segment_field[j][m]] = -1
                segment_field[j] = []
    for j in range(len(clustering.labels_)):
        if (clustering.labels_[j] >= 0):
            clustering.labels_[j] = 1
        else:
            clustering.labels_[j] = 0

    record_field_road_border = []
    for i in range(chazhi, len(clustering.labels_) - chazhi - 1):
        tianroad = []
        if clustering.labels_[i] != clustering.labels_[i + 1]:
            for j in range(i - chazhi, i + chazhi):
                tianroad.append(j)
            record_field_road_border.append(tianroad)
    for i in range(len(record_field_road_border)):
        for j in range(len(record_field_road_border[i]) - 1):
            flag = 0
            ind = record_field_road_border[i][j]
            #判断交界处的每一个点是否在农田的方向分布阈值内，如果交界点方向分布在合理范围内，flag=1不做任何操作，flag=0 将标签设置为道路
            for h in range(len(alltian)):
                if (float(dir[ind]) > (float(alltian[h]) - dirnum) and float(dir[ind]) < (float(alltian[h]) + dirnum)):
                    flag = 1
            if flag == 0:
                clustering.labels_[record_field_road_border[i][j]] = 0

    clusteringcopy = clustering
    return clusteringcopy

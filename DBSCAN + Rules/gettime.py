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

#求取轨迹段中任意两点之间的时间差
def gettime(q, origindataid, origindatatime, fistid, finalid):
    timed = 0 #统计时间间隔
    errornum = 0 #统计采样错误点的个数
    for j in range(len(origindataid)):
        if origindataid[j] == fistid:
            fist_time = origindatatime[j]
            h = j
            break
    for j in range(h, len(origindataid)):
        if origindataid[j] == finalid:
            final_time = origindatatime[j]
            h2 = j + 1
            break
    #将不符合处理方式的数据进行相应的转换
    for j in range(h, h2 - 1):
        if q <= 29:
            onefisttime = str(origindatatime[j]).replace("/", "-")
            onefinaltime = str(origindatatime[j + 1]).replace("/", "-")
        else:
            onefisttime = str(origindatatime[j])
            onefinaltime = str(origindatatime[j + 1])
        time_1_struct = datetime.strptime(str(onefisttime), "%Y-%m-%d %H:%M:%S")
        time_2_struct = datetime.strptime(str(onefinaltime), "%Y-%m-%d %H:%M:%S")
        timed = (time_2_struct - time_1_struct).seconds
        if timed != 1:
            errornum = errornum + 1
    if q <= 29:
        fist_time = str(fist_time).replace("/", "-")
        final_time = str(final_time).replace("/", "-")
    time_1_struct = datetime.strptime(str(fist_time), "%Y-%m-%d %H:%M:%S")
    time_2_struct = datetime.strptime(str(final_time), "%Y-%m-%d %H:%M:%S")
    timed = (time_2_struct - time_1_struct).seconds
    return timed, errornum
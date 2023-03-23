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
import getdistence
from sklearn.metrics import precision_score, recall_score, f1_score
import multiprocessing
from time import time
import getRRR as distec
from datetime import datetime, date
import heapq
import main as ma

def getroad2field(clusteringcopy, q, allsegment, alldir, allspeed, allerrorpoint, allslope,cleanx,cleany,newrowtag,speedcha,dirthreshold,dirnumthreshold):

    # 第一次田路田segment修改参数
    clustering = clusteringcopy.copy()
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
        numtian = 0
        numroad = 0
        for j in range(len(newallsegment[i])):
            if clustering[newallsegment[i][j]] == 0:
                numroad = numroad + 1
            else:
                numtian = numtian + 1
        if numtian >= numroad:
            newsegmentlabal.append(1)
        else:
            newsegmentlabal.append(0)
    for i in range(1, len(newallsegment) - 1):

        if newsegmentlabal[i] == 0 and newsegmentlabal[i - 1] == 1 and newsegmentlabal[i + 1] == 1:

            averagespeed = (newallspeed[i - 1] + newallspeed[i + 1]) / 2
            if newallspeed[i] >= (averagespeed - speedcha) and newallspeed[i] <= (averagespeed + speedcha):

                dirsamenum = 0
                for j in newalldir[i]:
                    for h1 in newalldir[i - 1]:
                        if j >= h1 - dirthreshold and j <= h1 + dirthreshold:
                            dirsamenum += 1
                    for h1 in newalldir[i + 1]:
                        if j >= h1 - dirthreshold and j <= h1 + dirthreshold:
                            dirsamenum += 1
                if dirsamenum >= dirnumthreshold:
                    if newallerrorpoint[i] != 0:
                        for j in range(len(newallsegment[i])):#将符合条件的进行修正标签
                            clustering[newallsegment[i][j]]=1
    return clusteringcopy
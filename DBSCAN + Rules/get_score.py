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
import getdistence as dis

#计算经过多次算法步骤之后的结果 标签需为 0 or 1
def getscore(picindex,clusteringcopy, turnnewrowtag, newrowtagcopy):
    q=picindex
    newrowtag = newrowtagcopy
    clustering2 = list(clusteringcopy.labels_).copy()
    dbptian = (precision_score(newrowtag, clustering2, average='binary')) * 100
    dbrtian= (recall_score(newrowtag, clustering2, average='binary')) * 100
    dbf1scoretian = (f1_score(newrowtag, clustering2, average='binary')) * 100
    for j in range(len(clustering2)):
        if (clustering2[j] == 1):
            clustering2[j] = 0
        else:
            clustering2[j] = 1
    dbproad = (precision_score(turnnewrowtag, clustering2, average='binary')) * 100
    dbrroad = (recall_score(turnnewrowtag, clustering2, average='binary')) * 100
    dbf1scoreroad = (f1_score(turnnewrowtag, clustering2, average='binary')) * 100
    if dbptian == 0 and dbrtian == 0 and dbf1scoretian == 0:
        total30p = dbproad
        total30r = dbrroad
        total30f = dbf1scoreroad
    else:
        total30p = (dbptian + dbproad) / 2
        total30r = (dbrtian + dbrroad) / 2
        total30f = (dbf1scoretian + dbf1scoreroad) / 2

    print("f1score:", total30f)
    print("precision:", total30p)
    print("recall:", total30r)
    print("field_f1score:", dbf1scoretian)
    print("field_precision:", dbptian)
    print("field_recall:", dbrtian)
    print("road_f1score:", dbf1scoreroad)
    print("road_precision:", dbproad)
    print("road_recall:", dbrroad)

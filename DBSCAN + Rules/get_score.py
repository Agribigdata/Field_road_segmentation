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
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import multiprocessing
from time import time
from datetime import datetime, date
import heapq
import getdistence as dis

#计算经过多次算法步骤之后的结果 标签需为 0 or 1
def getscore(picindex,clusteringcopy, turnnewrowtag, newrowtagcopy):
    q=picindex
    newrowtag = newrowtagcopy
    clustering2 = clusteringcopy.tolist()
    for i in range(len(clustering2)):
        if clustering2[i] >= 0:
            clustering2[i] = 1
        else:
            clustering2[i] = 0
    print(classification_report(newrowtagcopy, clustering2, digits=4))

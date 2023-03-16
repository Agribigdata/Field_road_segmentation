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
#求取两点之间、任意两点之间的距离
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r * 1000

def getdist(q, origindataid, origindataxy, origindatayx, fistid, finalid):
    onesegdist = 0
    onexy = origindataxy[q]
    oneyx = origindatayx[q]
    for j in range(len(origindataid[q])):
        if origindataid[q][j] == fistid:
            h1 = j
            break
    for j in range(h1, len(origindataid[q])):
        if origindataid[q][j] == finalid:
            h2 = j
            break
    for j in range(h1, h2):
        adi = haversine(onexy[j], oneyx[j], onexy[j + 1], oneyx[j + 1])
        onesegdist += adi
    return onesegdist
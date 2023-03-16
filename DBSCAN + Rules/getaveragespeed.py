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

def getaverspeed(q, origindata1, origindatas, fistid, finalid):
    onesegspe = 0
    for j in range(len(origindata1)):
        if origindata1[j] == fistid:
            h1 = j
            break
    for j in range(h1, len(origindata1)):
        if origindata1[j] == finalid:
            h2 = j + 1
            break
    for j in range(h1, h2):
        onesegspe = onesegspe + origindatas[j]
    return onesegspe / (h2 - h1)
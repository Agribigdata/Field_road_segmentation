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

print('loading data...\n')
#加载清理好的数据
#data1作为pands读取数据的中间转换
picture_index=0 #加载轨迹的index
data1 = pd.read_excel("agricultural machinery/data/cleandata" + str(picture_index + 1) + ".xlsx", header=0)
cleandata = data1.loc[:,
               ['ID', 'GEARSTIME', 'LNG', 'LAT', 'SPEED', 'DIRECTION', 'HIGHT', 'GPSLOCATION', 'ACCSTATE',
                'GOOGLE_LNG', 'GOOGLE_LAT', 'BAIDU_LAT', 'BAIDU_LNG']]
#根据算法需要读取清洗好数据的属性
clean_x = cleandata['LNG']
clean_y = cleandata['LAT']
clean_speed = cleandata['SPEED']
clean_id = cleandata['ID']
direct = cleandata['DIRECTION']
#读取源数据属性
data4 = pd.read_excel("origindata/data/data" + str(picture_index + 1) + ".xlsx", header=0)
oridata = data4.loc[:, ['ID', 'GEARSTIME', 'LNG', 'LAT', 'SPEED']]
origindata_time = oridata['GEARSTIME']
origindata_id = oridata['ID']
origindata_x = oridata['LNG']
origindata_y = oridata['LAT']
origindata_speed = oridata['SPEED']

#读取标注数据标签 road=0 field=1
data3 = pd.read_excel("agricultural machinery/labal/" + "labal" + str(picture_index + 1) + ".xlsx", header=0)
datarow_Tag = data3.loc[:, ['ID', 'Tag']]
origin_id = datarow_Tag['ID']
origin_tag = datarow_Tag['Tag']
#做源数据与清洗数据的标签对应，origin_id代表原始数据的id，origin_tag原始数据的标签,newrow_tag代表对应成功之后新数据的标签
rowid = origin_id
rowtag = origin_tag
newrow_tag = []
newrow_id = []
rowid = list(rowid)
dbscantid = clean_id
dbscantid = list(dbscantid)
for j in range(len(rowid)):
    if rowid[j] in dbscantid:
        if rowid[j] not in newrow_id:
            newrow_tag.append(rowtag[j])
            newrow_id.append(rowid[j])
#做标签转换，用来计算道路的识别准确率 road=1 field=0
turn_newrowtag = []
for j in range(len(newrow_tag)):
    if (newrow_tag[j] == 1):
        turn_newrowtag.append(0)
    else:
        turn_newrowtag.append(1)
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
import os

print('loading data...\n')
#加载清理好的数据
#data1作为pands读取数据的中间转换
picture_index=0 #加载轨迹的index
# 读取数据文件
allfile = []
top_path = 'DBSCAN_test_不降维/'
for filepath, dirnames, filenames in os.walk(top_path):
    for filename in filenames:
        allfile.append(filename)
print('allfile:', allfile)
fileindex = 80
file_path = 'paddy_harvestor_76.xlsx'
data1 = pd.read_excel(top_path + file_path, header=0)
#lon	lat	time	dir	vin	speed	ap	jp	br	tag
cleandata = data1.loc[:, ['time', 'lon', 'lat', 'speed', 'dir', 'tag']]
#根据算法需要读取清洗好数据的属性
clean_x = cleandata['lon']
clean_y = cleandata['lat']
clean_speed = cleandata['speed']
direct = cleandata['dir']
newrow_tag = cleandata['tag'].tolist()
clean_id = []
for m in range(len(direct)):
    clean_id.append(m)
#读取源数据属性
data4 = pd.read_excel(top_path + file_path, header=0)
oridata = data4.loc[:, ['time', 'lon', 'lat', 'speed']]
origindata_time = oridata['time']
origindata_x = oridata['lon']
origindata_y = oridata['lat']
origindata_speed = oridata['speed']
origindata_id = clean_id

#读取标注数据标签 road=0 field=1
data3 = pd.read_excel(top_path + file_path, header=0)
origin_id = clean_id
origin_tag = newrow_tag
#做源数据与清洗数据的标签对应，origin_id代表原始数据的id，origin_tag原始数据的标签,newrow_tag代表对应成功之后新数据的标签
rowid = origin_id
rowtag = origin_tag
newrow_id = clean_id
dbscantid = clean_id
#做标签转换，用来计算道路的识别准确率 road=1 field=0
turn_newrowtag = []
for j in range(len(newrow_tag)):
    if (newrow_tag[j] == 1):
        turn_newrowtag.append(0)
    else:
        turn_newrowtag.append(1)
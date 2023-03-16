'''
create by zxq on 2020/11/1
'''
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import openpyxl
import numpy as np
import math
from datetime import datetime, date

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r * 1000

if __name__ == '__main__':


        picture_index=0
        totaldeltag=[]
        data1 = pd.read_excel("origindata/data/"+"data"+str(picture_index+1)+".xlsx", header=0)
        data2 = data1.loc[:,[ 'GEARSTIME', 'LNG', 'LAT', 'SPEED']]
        x=data2['LNG']
        y=data2['LAT']
        speed=data2['SPEED']
        onetime=[]
        oritime = data2['GEARSTIME'].tolist()
    #计算时间：
        if picture_index <= 29:
            for j in range(len(oritime)):
                oritime[j] = str(oritime[j]).replace("/", "-")

        for j in range(len(oritime) - 1):
            time_1_struct = datetime.strptime(str(oritime[j]), "%Y-%m-%d %H:%M:%S")
            time_2_struct = datetime.strptime(str(oritime[j + 1]), "%Y-%m-%d %H:%M:%S")
            timed = (time_2_struct - time_1_struct).seconds
            onetime.append(timed)
    #（1）	清除时间间隔为0，保留点为第一个
        waitdelete = []
        for j in range(len(speed)-1):
            if onetime[j] == 0:
                waitdelete.append(j+1)
        data2 = data2.drop(waitdelete)
        data2 = data2.reset_index()
        del data2['index']

    #（2）	清除经纬度相同、速度为0的点，保留点为第一个。（静止轨迹）
        x = data2['LNG']
        y = data2['LAT']
        speed = data2['SPEED']
        allpoint = []
        allpointindex = []
        waitdelete2 = []
        renum0 = 0
        rembspeed = -2
        frontpoint=[]
        frontpoint.append(x[0])
        frontpoint.append(y[0])
        for j in range(1,len(x)):
            point = []
            point.append(x[j])
            point.append(y[j])
            if point[0]==frontpoint[0] and  point[1]==frontpoint[1] and speed[j]==0:
                waitdelete2.append(j)
            frontpoint = []
            frontpoint.append(x[j])
            frontpoint.append(y[j])
        data2 = data2.drop(waitdelete2)
        data2 = data2.reset_index()
        del data2['index']

    #（3）	清除经纬度相同、速度相同（速度不为0）且连续的点

        x = data2['LNG']
        y = data2['LAT']
        speed = data2['SPEED']
        allpoint = []
        allpointindex = []
        waitdelete2 = []
        renum0 = 0
        rembspeed = -2
        for j in range(len(x)):
            point = []
            point.append(x[j])
            point.append(y[j])
            point.append(speed[j])
            if point in allpoint:
                if j - rembspeed == 1:
                    waitdelete2.append(j)
                rembspeed = j
            if point not in allpoint:
                allpoint.append(point)
                allpointindex.append(j)
        data2 = data2.drop(waitdelete2)
        data2 = data2.reset_index()
        del data2['index']

    #（4）	清除 经纬度不相同、速度为0且连续的点，保留点为第一个。（静态漂移）
        x = data2['LNG']
        y = data2['LAT']
        speed = data2['SPEED']
        allpoint = []
        allpointindex = []
        waitdelete2 = []
        rembspeed = -2
        for j in range(len(x)):
            if speed[j] == 0:
                if j - rembspeed == 1:
                    waitdelete2.append(j)
                rembspeed = j
        data2 = data2.drop(waitdelete2)
        data2 = data2.reset_index()

    #（5）清理经纬度异常的点
        x = data2['LNG']
        y = data2['LAT']
        waitdelete = []
        for k in range(len(data2)):
            if x[k] < 74 or y[k] < 17: #需根据自身数据进行适当调整。
                waitdelete.append(k)
        data2 = data2.drop(waitdelete)
        data2 = data2.reset_index()
        del data2['index']
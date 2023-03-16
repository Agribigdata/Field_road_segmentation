import pandas as pd
from math import radians, cos, sin, asin, sqrt
import numpy as np
import math
import os
import datetime
from geopy.distance import geodesic
import heapq
import time
#获取文件名
def geodistance(begin_lon_lat, end_lon_lat):
    lng1 = begin_lon_lat[0]
    lat1 = begin_lon_lat[1]
    lng2 = end_lon_lat[0]
    lat2 = end_lon_lat[1]
    distance=sqrt(pow((lng2-lng1),2)+pow((lat2-lat1),2))
    # lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
    # dlon = lng2 - lng1
    # dlat = lat2 - lat1
    # a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    # distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
    distance = round(distance, 10)
    return distance
def haversine(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000
def getrowtime(data2, p):
    rowtimes = data2['时间']
    rowtimes = rowtimes.tolist()
    if p == 1:
        for i in range(len(rowtimes)):
            rowtimes[i] = str(rowtimes[i])
    if p == 0:
        for i in range(len(rowtimes)):
            rowtimes[i] = rowtimes[i][10:18]
    else:
        for i in range(len(rowtimes)):
            rowtimes[i] = rowtimes[i][11:19]
            if int(rowtimes[i][0]) == 0:
                rowtimes[i] = rowtimes[i][1:8]
    for i in range(len(rowtimes)):
        rowtimes[i] = rowtimes[i].replace(":", "")
        rowtimes[i] = int(rowtimes[i])
    return rowtimes
def gettimedist(rowtimes, i):
    hour = int(rowtimes[i] / 10000)
    minute = int(rowtimes[i] % 10000 / 100)
    second = int(rowtimes[i] % 100)
    hour0 = int(rowtimes[i - 1] / 10000)
    minute0 = int(rowtimes[i - 1] % 10000 / 100)
    second0 = int(rowtimes[i - 1] % 100)
    timedist = (hour - hour0) * 3600 + (minute - minute0) * 60 + second - second0
    return timedist
def findfiles(files_path,result,cit_name):
    #查找文件代码
    files = os.listdir(files_path)
    for root, dirs, file in os.walk(files_path):
        cit_name.append(file)
    for s in files:
        #cit_name.append(s)
        s_path = os.path.join(files_path, s)
        if os.path.isdir(s_path):
            findfiles(s_path, result)
        elif os.path.isfile(s_path):
            result.append(s_path)
    return result,cit_name
def getfilename(path):
    result = []
    cit_name = []
    result_list,file_name=findfiles(path,result,cit_name)
    dict2020=[]
    kkk=[i for item in file_name for i in item]
    for i in range(len(result_list)):
        dict2020.append(result_list[i])
    return dict2020,kkk


class comobj():
    def __init__(self, key, distance):
        # 优先级
        self.key = key
        # 具体的属性值，可以用类封装起来
        self.distance = distance

    def __lt__(self, other):  # 这里的比较规则是：先根据key的大小判断，如果key相等则根据name判断
        # 最后都形成小根堆
        if self.distance < other.distance:
            return True
        else:
            return False
def cal_diff_main(path, result, allfileq):
    data = result
    col_name = data.columns.tolist()
    col_name.insert(13, 'lon_diff')
    col_name.insert(14, "lat_diff")
    #col_name.insert(15, 'height_diff')
    col_name.insert(15, "dir_diff")
    data_list_lon = data['lon'].values.flatten().tolist()
    if "lon" in data_list_lon:
        data_list_lon.remove("lon")
    data_list_lat = data['lat'].values.flatten().tolist()
    if "lat" in data_list_lat:
        data_list_lat.remove("lat")
    # data_list_height = data['height'].values.flatten().tolist()
    # if "height" in data_list_height:
    #     data_list_height.remove("height")
    data_list_dir = data['dir'].values.flatten().tolist()
    if "dir" in data_list_dir:
        data_list_dir.remove("dir")
    lon_diff = []
    lon_diff.append(0)
    lat_diff = []
    lat_diff.append(0)
    # height_diff = []
    # height_diff.append(0)
    dir_diff = []
    dir_diff.append(0)
    for i in range(len(data_list_lon)):
        if i == 0:
            continue
        else:
            lon_diff.append(data_list_lon[i] - data_list_lon[i - 1])
    for i in range(len(data_list_lat)):
        if i == 0:
            continue
        else:
            lat_diff.append(data_list_lat[i] - data_list_lat[i - 1])
    # for i in range(len(data_list_height)):
    #     if i == 0:
    #         continue
    #     else:
    #         height_diff.append(data_list_height[i] - data_list_height[i - 1])
    for i in range(len(data_list_dir)):
        if i == 0:
            continue
        else:
            dir_diff.append(data_list_dir[i] - data_list_dir[i - 1])
    data['lon_diff'] = lon_diff
    data['lat_diff'] = lat_diff
    #data['height_diff'] = height_diff
    data['dir_diff'] = dir_diff
    #不降维
    # data.to_excel(path + "_不降维/" + allfileq.split(".")[0]+ ".xlsx", index=False)
    dim1_data=data
    # #不计算差值降维
    # data.columns = [column for column in data]
    # column = ['lon', 'lat', 'dir', 'vin', 'speed', 'ap', 'jp', 'br', 'tag', 'lon_diff', 'lat_diff', 'dir_diff']
    # data['tra_info'] = data.iloc[:, 0:].apply(
    #     lambda x: [x.values[0], x.values[1], x.values[2], x.values[3], x.values[4], x.values[5], x.values[6],
    #                x.values[7]], axis=1)
    # data['tag_only'] = data.iloc[:, 0:].apply(lambda x: [x.values[8]], axis=1)
    # data = data.drop(column, axis=1)
    # tag_data = list(data['tag_only'])
    # for j in range(len(tag_data)):
    #     tag_data[j] = tag_data[j][0]
    # c = pd.DataFrame({'tra_info': list(data['tra_info']), 'tag': tag_data})
    # c.to_csv(path + "_不计算差值降维后/" + allfileq.split(".")[0] + ".csv", index=False)

    #计算差值降维
    dim1_data.columns = [column for column in dim1_data]
    column = ['lon', 'lat', 'dir', 'vin', 'speed', 'ap', 'jp', 'br', 'tag', 'lon_diff', 'lat_diff', 'dir_diff']
    dim1_data['tra_info'] = dim1_data.iloc[:, 0:].apply(
        lambda x: [x.values[3], x.values[4], x.values[5], x.values[6], x.values[7], x.values[9], x.values[10], x.values[11]], axis=1)
        #lambda x: [x.values[4], x.values[9], x.values[10],x.values[11]], axis=1)
    dim1_data['tag_only'] = dim1_data.iloc[:, 0:].apply(lambda x: [x.values[8]], axis=1)
    dim1_data = dim1_data.drop(column, axis=1)
    tag_data = list(dim1_data['tag_only'])
    for j in range(len(tag_data)):
        tag_data[j] = tag_data[j][0]
    c = pd.DataFrame({'tra_info': list(dim1_data['tra_info']), 'tag': tag_data})
    c.to_pickle(path + "_计算差值降维后/" + allfileq.split(".")[0] + ".xlsx")


def extract_feature(result):
    # 存入数据：
    data = result
    #print(data)
    data_list = data['时间'].tolist()
    data_list_lon = data['经度'].tolist()
    data_list_lat = data['纬度'].tolist()
    # vin
    final_vincety = []
    for j in range(len(data_list)):
        if j == 0:
            continue
        else:
            try:
                d1 = datetime.datetime.strptime(str(data_list[j - 1]), '%Y/%m/%d %H:%M:%S')
                d2 = datetime.datetime.strptime(str(data_list[j]), '%Y/%m/%d %H:%M:%S')
            except:
                d1 = datetime.datetime.strptime(str(data_list[j - 1]), '%Y-%m-%d %H:%M:%S')
                d2 = datetime.datetime.strptime(str(data_list[j]), '%Y-%m-%d %H:%M:%S')
            #print(type(d1))
            temp = (d2 - d1).seconds
            #print(temp)
            newport_ri = (data_list_lat[j - 1], data_list_lon[j - 1])
            cleveland_oh = (data_list_lat[j], data_list_lon[j])
            distance = geodesic(newport_ri, cleveland_oh).m
            final_vincety.append(distance / temp)
    final_vincety.append(0)
    col_name = data.columns.tolist()
    col_name.insert(10, 'vindistance')
    data['vindistance'] = final_vincety
    # ap
    data_list_vin = data['vindistance'].values.flatten().tolist()
    if "vindistance" in data_list_vin:
        data_list_vin.remove("vindistance")
    final_ap = []
    for j in range(len(data_list)):
        if j == 0:
            continue
        else:
            try:
                d1 = datetime.datetime.strptime(str(data_list[j - 1]), '%Y/%m/%d %H:%M:%S')
                d2 = datetime.datetime.strptime(str(data_list[j]), '%Y/%m/%d %H:%M:%S')
            except:
                d1 = datetime.datetime.strptime(str(data_list[j - 1]), '%Y-%m-%d %H:%M:%S')
                d2 = datetime.datetime.strptime(str(data_list[j]), '%Y-%m-%d %H:%M:%S')
            temp = (d2 - d1).seconds
            final_ap.append((data_list_vin[j] - data_list_vin[j - 1]) / temp)
    final_ap.append(0)
    col_name = data.columns.tolist()
    col_name.insert(11, 'Ap')
    data['Ap'] = final_ap
    # jp
    data_list_ap = data['Ap'].values.flatten().tolist()
    if "Ap" in data_list_ap:
        data_list_ap.remove("Ap")
    final_jp = []
    for j in range(len(data_list)):
        if j == 0:
            continue
        else:
            try:
                d1 = datetime.datetime.strptime(str(data_list[j - 1]), '%Y/%m/%d %H:%M:%S')
                d2 = datetime.datetime.strptime(str(data_list[j]), '%Y/%m/%d %H:%M:%S')
            except:
                d1 = datetime.datetime.strptime(str(data_list[j - 1]), '%Y-%m-%d %H:%M:%S')
                d2 = datetime.datetime.strptime(str(data_list[j]), '%Y-%m-%d %H:%M:%S')
            temp = (d2 - d1).seconds
            final_jp.append((data_list_ap[j] - data_list_ap[j - 1]) / temp)
    final_jp.append(0)

    col_name = data.columns.tolist()
    col_name.insert(12, 'Jp')
    data['Jp'] = final_jp
    # br
    final_br = []
    bearings = []
    for j in range(len(data_list_lon)):
        if j == 0:
            continue
        else:
            y = (math.sin(data_list_lon[j] - data_list_lon[j - 1])) * (math.cos(data_list_lat[j]))
            x = math.cos(data_list_lat[j - 1]) * math.sin(data_list_lat[j]) - math.sin(
                data_list_lat[j - 1]) * math.cos(data_list_lat[j]) * math.cos(
                data_list_lon[j] - data_list_lon[j - 1])
            bearings.append(math.atan2(y, x))
    for k in range(len(bearings)):
        if k == 0:
            continue
        else:
            final_br.append(bearings[k] - bearings[k - 1])
    for ss in range(2):
        final_br.append(0)
    col_name = data.columns.tolist()
    col_name.insert(13, 'BR')
    data['BR'] = final_br
    # 删去最后三个的
    k = len(data)
    for i in range(1, 4):
        data.drop([k - i], inplace=True)
    # 速度变为ms的
    data_list_lon = data['经度'].values.flatten().tolist()
    if "经度" in data_list_lon:
        data_list_lon.remove("经度")
    data_list_lat = data['纬度'].values.flatten().tolist()
    if "纬度" in data_list_lat:
        data_list_lat.remove("纬度")

    data_list_dir = data['方向'].values.flatten().tolist()
    if "方向" in data_list_dir:
        data_list_dir.remove("方向")
    # data_list_height = data['高度'].values.flatten().tolist()
    # if "高度" in data_list_height:
    #     data_list_height.remove("高度")

    data_list_vin = data['vindistance'].values.flatten().tolist()
    if "vindistance" in data_list_vin:
        data_list_vin.remove("vindistance")
    try:
        data_list_tag = data['标记'].values.flatten().tolist()
        if "标记" in data_list_tag:
            data_list_tag.remove("标记")
    except:
        data_list_tag = data['标签'].values.flatten().tolist()
        if "标签" in data_list_tag:
            data_list_tag.remove("标签")
    data_list_speed = data['速度'].values.flatten().tolist()
    if "速度" in data_list_speed:
        data_list_speed.remove("速度")
    data_list_ap = data['Ap'].values.flatten().tolist()
    if "Ap" in data_list_ap:
        data_list_ap.remove("Ap")
    data_list_jp = data['Jp'].values.flatten().tolist()
    if "Jp" in data_list_jp:
        data_list_jp.remove("Jp")
    data_list_br = data['BR'].values.flatten().tolist()
    if "BR" in data_list_br:
        data_list_br.remove("BR")
    for jjj in range(len(data_list_speed)):
        data_list_speed[jjj] = data_list_speed[jjj] / 3.6
    x = np.array(
        [data_list_speed, data_list_ap, data_list_jp, data_list_br, data_list_lon, data_list_lat, data_list_dir,
          data_list_vin])
    result = x
    data_result_speed = result[0]
    data_result_ap = result[1]
    data_result_jp = result[2]
    data_result_br = result[3]

    data_result_lon = result[4]
    data_result_lat = result[5]
    data_result_dir = result[6]
    #data_result_height = result[7]
    data_result_vin = result[7]
    dict_zip = {"lon": data_result_lon, "lat": data_result_lat, "dir": data_result_dir,
                 "vin": data_result_vin, "speed": data_result_speed,
                "ap": data_result_ap, "jp": data_result_jp, "br": data_result_br,"tag": data_list_tag}
    result_excel = pd.DataFrame(dict_zip)
    return result_excel
def main(path,method = "GCN"):
    if not os.path.exists(path +"_不降维//"):
        os.mkdir(path +"_不降维//")
    if not os.path.exists(path + "_计算差值降维后//"):
        os.mkdir(path + "_计算差值降维后//")
    # if not os.path.exists(path + "_不计算差值降维后//"):
    #     os.mkdir(path + "_不计算差值降维后//")
    if not os.path.exists(path + "点关系/"):
        os.mkdir(path + "点关系/")
    if not os.path.exists(path + "点关系/onedisandtwodis_down/"):
        os.mkdir(path + "点关系/onedisandtwodis_down/")
    if not os.path.exists(path + "点关系/onedisandtwodis_up/"):
        os.mkdir(path + "点关系/onedisandtwodis_up/")
    if not os.path.exists(path + "点关系/first/"):
        os.mkdir(path + "点关系/first/")
    if not os.path.exists(path + "点关系/second/"):
        os.mkdir(path + "点关系/second/")
    staticremove = 0
    alllen = 0
    # top_path = path + "分好的/" + num_begin + "_" + str(num_file) + "/"
    top_path=path
    allfile = os.listdir(top_path)
    for q in range(len(allfile)):
        data1 = pd.read_excel(top_path+"/" + allfile[q], header=0)
        try:
            data2 = data1.loc[:, ['时间','经度', '纬度', '速度', '方向','标记']]
        except:
            data2 = data1.loc[:, ['时间', '经度', '纬度', '速度', '方向', '标签']]
        x = data2['经度']
        y = data2['纬度']
        speed = data2['速度']
        alllen = alllen + len(speed)
        # 清理掉经纬度速度重复的
        waitdelete = []
        allpoint = []
        for j in range(len(speed)):
            point = []
            point.append([x[j], y[j], speed[j]])
            if point in allpoint:
                waitdelete.append(j)
            else:
                allpoint.append(point)
        data2 = data2.drop(waitdelete)
        data2 = data2.reset_index()
        #清理掉时间重复的
        time = data2['时间']
        timedelete = []
        all_time_point = []
        for j in range(len(time)):
            #print(time[j])
            if time[j] in all_time_point:
                timedelete.append(j)
            else:
                all_time_point.append(time[j])
        data2 = data2.drop(timedelete)
        data2 = data2.reset_index()
        del data2['index']
        staticremove = staticremove + len(waitdelete)
        data = pd.DataFrame(data2)
        result_excel=extract_feature(data)
        cal_diff_main(path, result_excel, allfile[q])
if __name__=="__main__":
    begin = time.time()
    #data path
    path = "xxx"
    method = "GCN" # "DT"
    main(path)
    end = time.time()
    print(end - begin)












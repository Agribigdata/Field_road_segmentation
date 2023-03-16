#利用目标检测的最大概率值，进行全轨迹的聚类
import csv
import matplotlib.pyplot as plt
import cudf
import os
import numpy as np
import pandas as pd
from matplotlib.path import Path
import math
import cv2
from cuml.cluster import DBSCAN as cumldbscan
from sklearn.metrics import f1_score, classification_report,accuracy_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

detection_path = 'detection_result/Swin_S_wheat_150/'   #目标检测结果文件地址
segment_data_xlsx = 'data/segment_data_150/'    #原始轨迹数据excel 文件地址

def detection_result():
    #获取目标检测的 txt 结果文件
    detection_file = []
    for root_dir,sub_dir,files in os.walk(detection_path):
        for file in files:
            if file.endswith(".txt"):
                file_name = os.path.join(root_dir, file)
                detection_file.append(file_name)
    #print('detection_file:',detection_file)
    return detection_file

def segment_data(rand_num):
    #获取原始excel形式的轨迹数据
    segment_data_list = []
    for root_dir,sub_dir,files in os.walk(segment_data_xlsx):
        for file in files:
            if file.endswith(".xlsx"):
                #构造绝对路径
                file_name = os.path.join(root_dir, file)
                segment_data_list.append(file_name)
    # 训练测试文件划分
    train_file, test_file, _, _ = train_test_split(segment_data_list, segment_data_list, test_size=0.1, random_state=rand_num)
    #print('len test_file:', len(test_file))
    return test_file

def match_file(segment_file, detection_file):
    match_list = []
    #将 excel 数据和 目标检测的结果文件进行匹配
    for seg_file in segment_file:
        flag = 0 #用于判断是否找到了匹配文件
        match_seg_tag = seg_file.split('/')[2].split('.')[0]  # 切分excel文件除路径和后缀之外的文件名
        for det_file in detection_file:
            match_det_tag = det_file.split('/')[2].split('.')[0].replace('_detection','')  # 切分目标检测文件路径和后缀之外的文件名
            if match_seg_tag == match_det_tag:
                flag = 1 #标志已经找到了匹配文件
                match_list.append([seg_file,det_file])
        if flag == 0:  #因为其中有部分农田未被模型识别到，所以存在匹配不上的情况
            match_list.append([seg_file,''])
    return match_list

def plot_fig(x, y, color,name): #借助经纬度坐标和 颜色画图

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # 设置标题
    ax1.set_title(name)
    # 设置X轴标签
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')
    # 画散点图
    # dataset.extend(datas),dataset2.extend(datas2),c=dataset3.extend(datas3)
    ax1.scatter(x, y, c=color, marker='.')
    # 设置图标
    plt.legend('x1')
    plt.show()
    plt.close()

def detect_OD(x, y, detection_file):

    fig, ax = plt.subplots()  # fig,ax = plt.subplots()等价于： fig = plt.figure()    ax = fig.add_subplot(1,1,1)
    ax.set_axis_off()  # 关闭坐标轴
    ax.scatter(x, y, 0.1)
    plt.savefig('temp.png')
    xy_pixels = ax.transData.transform(np.vstack([x, y]).T)
    # 像素坐标
    plt.close()
    xy_pixelsD = pd.DataFrame(xy_pixels, columns=['x', 'y'])
    pixel_x = xy_pixelsD['x'] #经度对应的像素位置
    pixel_y = xy_pixelsD['y'] #纬度对应的像素位置
    OD_cluster = []
    if detection_file != '':
        data = open(detection_file, encoding="utf8") #读取 目标检测文件结果
        num = 0
        for i in range(len(x)):
            OD_cluster.append(0)
        tag_label = 1
        for col in data:
            num += 1
            best_str = col
            best_str = best_str.replace('\n','')
            loclist = best_str.split('    ')
            y_min = int(loclist[2])
            y_max = int(loclist[4])
            x_min = int(loclist[1])
            x_max = int(loclist[3])
            #根据预测结果获取农田的位置
            # b'field 0.98' 65 315 105 386
            for i in range(len(pixel_x)):
                if (pixel_x[i] >= x_min and pixel_x[i] <= x_max and (480 - pixel_y[i]) >= y_min and (480 - pixel_y[i]) <= y_max):
                    OD_cluster[i] = tag_label  #
            tag_label += 1
    return OD_cluster

def trajectory_info(bir_file, bili = 1, minptsbeilv = 1):

    data = pd.read_excel(bir_file[0]) #读取数据文件
    x = list(data['经度'])
    y = list(data['纬度'])
    tag = list(data['标签'])
    OD_cluster = detect_OD(x, y, bir_file[1])
    clustering = []
    for i in range(len(x)):
        clustering.append(-1)

    prov_dataSet = [[0 for j in range(2)] for h in range(len(x))]
    total_dataSet = cudf.DataFrame()
    listx = []
    listy = []
    for j in range(len(x)):
        prov_dataSet[j][0] = x[j]
        prov_dataSet[j][1] = y[j]
        listx.append(x[j])
        listy.append(y[j])
    total_dataSet['0'] = listx
    total_dataSet['1'] = listy

    #plot_fig(x, y, clustering, bir_file[0])
    GS_cluster = cumldbscan(eps=21 / 100000, min_samples=21).fit(total_dataSet)
    nparraylabel = np.array(GS_cluster.labels_.to_pandas())
    label0_1 = np.where(nparraylabel >= 0, 1, 0)
    GS_cluster = label0_1.tolist()
    # print('len OD_cluster:',len(OD_cluster))
    # print('sum OD_cluster:', sum(OD_cluster))
    # print('len GS_cluster:',len(GS_cluster))
    # print('sum GS_cluster:',sum(GS_cluster))
    if len(OD_cluster) > 0:
        # print('len(GS_cluster):',len(GS_cluster))
        # print('sum(GS_cluster):',sum(GS_cluster))
        # print('sum(OD_cluster):', sum(OD_cluster))
        if len(GS_cluster) != sum(GS_cluster):
            # if sum(OD_cluster) == 0:
            #     OD_cluster[0] = 1
            DBI_OD = davies_bouldin_score(prov_dataSet, OD_cluster)
            DBI_DBSCAN =davies_bouldin_score(prov_dataSet,GS_cluster)
            if DBI_DBSCAN < DBI_OD:
                #print('GS_cluster')
                better_clustering = GS_cluster
            else:
                OD_cluster = np.array(OD_cluster)
                label0_1 = np.where(OD_cluster > 0, 1, 0)
                OD_cluster = label0_1.tolist()
                better_clustering = OD_cluster
        else:
            OD_cluster = np.array(OD_cluster)
            label0_1 = np.where(OD_cluster > 0, 1, 0)
            OD_cluster = label0_1.tolist()
            better_clustering = OD_cluster
    else:
        better_clustering = GS_cluster
    #仅使用 聚类:
    # better_clustering = GS_cluster

    # 仅使用 OD:
    # OD_cluster = np.array(OD_cluster)
    # label0_1 = np.where(OD_cluster > 0, 1, 0)
    # OD_cluster = label0_1.tolist()
    # better_clustering = OD_cluster

    '''
    if len(OD_cluster) > 0:
        for h in range(len(GS_cluster)):
            if OD_cluster[h] == 1:
                GS_cluster[h] = 1
    better_clustering = GS_cluster
    '''
    return better_clustering, tag

def score(pred_list, tag_list):
    target_names = ['road','field']
    print('len pred:', len(pred_list))
    result_report = classification_report(tag_list, pred_list, digits=4, target_names=target_names, output_dict=True) #
    result_acc = accuracy_score(tag_list, pred_list)
    return result_acc, result_report


if __name__ == '__main__':
    print('DBI')
    fold_num = 20  # 交叉验证次数
    start_seed = 2022
    test_road_precision = 0
    test_road_recall = 0
    test_road_f1score = 0

    test_field_precision = 0
    test_field_recall = 0
    test_field_f1score = 0

    test_accuracy = 0
    test_accuracy_single = 0

    test_macro_precision = 0
    test_macro_recall = 0
    test_macro_f1score = 0

    test_weight_precision = 0
    test_weight_recall = 0
    test_weight_f1score = 0
    lenset = 0
    for kfold in range(fold_num):
        print('kfold:',start_seed + kfold)
        segment_file = segment_data(start_seed + kfold)  #获取原始 excel 轨迹数据
        detection_file = detection_result()  #获取目标检测结果文件
        match_list = match_file(segment_file, detection_file) #将轨迹原始数据文件与目标检测文件匹配起来
        file_num = 0
        pred_list = []
        tag_list = []
        for bi_file in match_list:
            #print(file_num)
            #pred,tag = trajectory_info(bi_file, float(epsbei/2), float(minptsbei/2))
            pred,tag = trajectory_info(bi_file, 3, 1)
            # oneresult = score(pred, tag)
            # if bi_file[0] in ['pic_labled_file/segment_data_150/0610_370150988.xlsx','pic_labled_file/segment_data_150/0612_370150950.xlsx']:
            # #if oneresult < 0.7:
            #     print('oneresult:',oneresult)
            #     print('bi_file[0]:',bi_file[0])
            #     data = pd.read_excel(bi_file[0])  # 读取数据文件
            #     x = list(data['经度'])
            #     y = list(data['纬度'])
            #     plot_fig(x,y,pred,bi_file[0].split('/')[2])
            pred_list += pred
            tag_list += tag
            file_num += 1
        result_acc, result_report = score(pred_list, tag_list)
        print('acc:', result_acc)
        # 按照样本评价

        test_road_precision += float(result_report['road']['precision'])
        test_road_recall += float(result_report['road']['recall'])
        test_road_f1score += float(result_report['road']['f1-score'])

        test_field_precision += float(result_report['field']['precision'])
        test_field_recall += float(result_report['field']['recall'])
        test_field_f1score += float(result_report['field']['f1-score'])

        test_accuracy += float(result_report['accuracy'])
        print('report acc:', float(result_report['accuracy']))

        test_macro_precision += float(result_report['macro avg']['precision'])
        test_macro_recall += float(result_report['macro avg']['recall'])
        test_macro_f1score += float(result_report['macro avg']['f1-score'])

        print('road f1score:',float(result_report['road']['f1-score']))
        print('field f1score:',float(result_report['field']['f1-score']))
        print('macro f1score:',float(result_report['macro avg']['f1-score']))

        test_weight_precision += float(result_report['weighted avg']['precision'])
        test_weight_recall += float(result_report['weighted avg']['recall'])
        test_weight_f1score += float(result_report['weighted avg']['f1-score'])

        test_accuracy_single += result_acc
        lenset += 1
        print('======================')
    print('rand_state:',start_seed)
    print('fold_num:',fold_num)
    print('lenset:',lenset)
    print('test acc:', test_accuracy/lenset)

    print('++++++++++test++++++++++')
    print('               precision    recall    f1score')
    print('        road    ' + str(round(test_road_precision / lenset, 4)) + '      ' + str(
        round(test_road_recall / lenset, 4)) + '    ' + str(round(test_road_f1score / lenset, 4)))
    print('       field    ' + str(round(test_field_precision / lenset, 4)) + '      ' + str(
        round(test_field_recall / lenset, 4)) + '    ' + str(round(test_field_f1score / lenset, 4)))
    print('\n')
    print('    accuracy                          ' + str(round(test_accuracy / lenset, 4)))
    print('   macro avg    ' + str(round(test_macro_precision / lenset, 4)) + '      ' + str(
        round(test_macro_recall / lenset, 4)) + '    ' + str(round(test_macro_f1score / lenset, 4)))
    print('weighted avg    ' + str(round(test_weight_precision / lenset, 4)) + '      ' + str(
        round(test_weight_recall / lenset, 4)) + '    ' + str(round(test_weight_f1score / lenset, 4)))

import os

import numpy as np
import pandas as pd
import torch

time = 0


def findfiles(files_path):
    # 查找文件代码
    result = []
    cit_name = []
    global time
    files = os.listdir(files_path)
    for root, dirs, file in os.walk(files_path):
        cit_name.append(file)
    for s in files:
        # cit_name.append(s)
        s_path = os.path.join(files_path, s)
        if os.path.isdir(s_path):
            findfiles(s_path)
        elif os.path.isfile(s_path):
            result.append(s_path)
    return result, cit_name


def getloader(path, randomstate, flag):
    path_data_original = path + "_计算差值降维后_10fold/" + str(randomstate) + "/" + flag
    result_data_ori, cit_name = findfiles(path_data_original)
    cit_name = [i for item in cit_name for i in item]
    data_graph = []
    for i in cit_name:
        temp_graph = []
        data_x = []
        data_y = []
        data_ori = pd.read_excel(path_data_original + "/" + i)
        # data_ori = pd.read_csv(r"/home/lgy/new_GCn/dataset/data/用点的/0605_370150950.csv")
        for data_pre_x in data_ori["tra_info"]:
            data_new = data_pre_x.replace("[", "")
            data_final = data_new.replace("]", "")
            data_str = data_final.split(",")
            data_float_m = map(float, data_str)
            data = list(data_float_m)
            data_x.append(data)
        # data_ori = pd.read_excel("/home/liguangyuan/GCN_system/dataset/data/new_paddy_wheat/wheat_150_25/"+flag+"/"+ i)
        # # data_ori = pd.read_csv(r"/home/lgy/new_GCn/dataset/data/用点的/0605_370150950.csv")
        # for datasql in range(len(data_ori['speed'])):
        #     temp_test = list(data_ori.iloc[datasql, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23,25, 26]])
        #     data_x.append(temp_test)
        # 得到y
        for data_single in data_ori["tag"]:
            data_y.append(data_single)
        # print(data_x)
        x = torch.tensor(data_x, dtype=torch.float)
        y = torch.tensor(data_y, dtype=torch.float)
        data_down = pd.read_excel(path + "点关系/onedisandtwodis_down/" + i)
        # data_first = pd.read_csv(r"/home/liguangyuan/new_GCN/dataset/data/用点的/onedisandtwodis/0605_370150950.csv")
        begin_temp = list(data_down["begin_point"])
        middle_temp = list(data_down["middle_point"])
        end_temp = list(data_down["end_point"])
        dict_point = list(zip(begin_temp, middle_temp, end_temp))
        matrix = np.zeros((len(begin_temp), len(begin_temp)))
        j = 0
        for dic in dict_point:
            if j == len(begin_temp) - 1:
                continue
            elif j == len(begin_temp) - 2:
                matrix[dic[0]][dic[1]] = 0.1
                matrix[dic[1]][dic[0]] = 0.1
            else:
                # print(dic)
                matrix[dic[0]][dic[1]] = 0.1
                matrix[dic[1]][dic[0]] = 0.1
                matrix[dic[0]][dic[2]] = 0.2
                matrix[dic[2]][dic[0]] = 0.2
            j += 1
        data_up = pd.read_excel(path + "点关系/onedisandtwodis_up/" + i)
        # data_first = pd.read_csv(r"/home/liguangyuan/new_GCN/dataset/data/用点的/onedisandtwodis/0605_370150950.csv")
        begin_temp = list(data_up["begin_point"])
        middle_temp = list(data_up["middle_point"])
        end_temp = list(data_up["end_point"])
        dict_point = list(zip(begin_temp, middle_temp, end_temp))
        j = 0
        for dic in dict_point:
            if j == 0:
                continue
            elif j == 1:
                matrix[dic[0]][dic[1]] = 0.3
                matrix[dic[1]][dic[0]] = 0.3
            else:
                # print(dic)
                matrix[dic[0]][dic[1]] = 0.3
                matrix[dic[1]][dic[0]] = 0.3
                matrix[dic[0]][dic[2]] = 0.4
                matrix[dic[2]][dic[0]] = 0.4
            j += 1
        data_first = pd.read_excel(path + "点关系/first/" + i)
        # data_first = pd.read_csv(r"/home/lgy/new_GCn/dataset/data/用点的/first/0605_370150950.csv")
        begin_temp = list(data_first["begin_point"])
        end_temp = list(data_first["end_point"])
        dict_point = dict(zip(begin_temp, end_temp))
        for dic in dict_point.items():
            # print(dic)
            matrix[dic[0]][dic[1]] = 0.5
            matrix[dic[1]][dic[0]] = 0.5

        data_second = pd.read_excel(path + "点关系/second/" + i)
        # data_second = pd.read_csv(r"/home/lgy/new_GCn/dataset/data/用点的/second/0605_370150950.csv")
        begin_temp = list(data_second["begin_point"])
        end_temp = list(data_second["end_point"])
        dict_point = dict(zip(begin_temp, end_temp))
        for dic in dict_point.items():
            matrix[dic[0]][dic[1]] = 0.6
            matrix[dic[1]][dic[0]] = 0.6
        temp_graph.append(torch.tensor(matrix))
        temp_graph.append(x)
        temp_graph.append(y)
        data_graph.append(temp_graph)
    return data_graph


# getloader(16,"train")

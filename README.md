# Extraction fetch
抽取特征且计算点关系-8dim:
```
python data_pre_8dim.py 
```
抽取特征-25dim:
```
python data_pre_25dim.py
```
# GCN-8dim
Run:
```
python GCN.py
```
set_t_12_d1_graph.py为构图代码

# LSTM-8dim
Run:
```
python LSTM.py
```
# DT-8dim
Run:
```
python DecisionTree_8dim.py
```
# RF-8dim
Run:
```
python RandomForest.py
```

# DBSCAN + Rules
1、数据清理
```
     python datacleaning.py
```
2、田路分割模型
```
     python main.py
```
# DBSCAN + OD + DBI
Run:
```
python MM_paddy_GS_OD.py 
```
or
```
python MM_wheat_GS_OD.py
```
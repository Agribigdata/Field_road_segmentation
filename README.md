
# Data preparation 
Data are downloaded from: https://github.com/Agribigdata/dataset_code 
, unzipped and stored in a directory

The path to the data-stored directory needs to be set to the corresponding path parameter in py file.

# Feature extraction 
Extracting 8-dim features and calculating point relations (for GCN)
```
python data_pre_8dim_and_get_relationship.py 
```
Extraction ony 8-dim features:
```
python data_pre_8dim.py 
```
Extraction 25-dim features:
```
python data_pre_25dim.py
```
# GCN
The input is 8-dim features

Run:
```
python GCN.py
```
Notice that set_t_12_d1_graph.py is the composition code and is embedded in GCN.py

# LSTM
The input is 8-dim features
Run:
```
python LSTM.py
```
# DT
If the input is 8-dim features
Run:
```
python DecisionTree_8dim.py
```
# RF
The input is 8-dim features
Run:
```
python RandomForest.py
```

# DBSCAN + Rules
data cleaning
```
     python datacleaning.py
```
field-road segmentation
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

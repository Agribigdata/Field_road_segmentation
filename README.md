
Data downloaded from: https://github.com/Agribigdata/dataset_code 
The path to the directory where the data is stored is written in the corresponding path in each file

# fetch extraction 
Extracting 8dim features and calculating point relations (for GCN)
```
python data_pre_8dim_and_get_relationship.py 
```
Extraction feature-8dim:
```
python data_pre_8dim.py 
```
Extraction feature-25dim:
```
python data_pre_25dim.py
```
# GCN
8-dim
Run:
```
python GCN.py
```
set_t_12_d1_graph.py is the composition code and is embedded in GCN.py

# LSTM
8-dim
Run:
```
python LSTM.py
```
# DT
8-dim
Run:
```
python DecisionTree_8dim.py
```
# RF
8-dim
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

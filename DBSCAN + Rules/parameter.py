'''
create by zxq on 2020/11/1
'''

#设置算法参数

#DBSCAN聚类参数
eps=39 / 100000 #DBSCAN聚类的最小半径
min_samples = 71 #DBSCAN聚类的点数

#cluster2road参数
c2r_base = 0 #每个方向上点数目分布最小数量
c2r_ratio = 10000 #农田点方向需分布的方向数
c2r_diff = 100 #交界处的点与农田内部方向分布最多的点的浮动阈值
c2r_dir_num = 0 #对农田和道路交界处的探测点数

#FRF参数
FRF_speedcha = 20#速度浮动阈值
FRF_dirthreshold = 0#方向浮动阈值（该方向是以10度为单位）
FRF_dirnumthreshold = 0#当前轨迹前三个数目最多方向与前后轨迹段相同数目最小值

#RRR参数
RRR_speedthreshold = 20
RRR_dirthreshold = 0
RRR_dirnumthreshold = 30
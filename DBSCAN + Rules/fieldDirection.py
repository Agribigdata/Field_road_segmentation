import dbscan as db
import loaddata as lddata

# smalltianwai为定义的存储某标签类别含有点的列表
segment_field = [0 for j in range(len(db.clustering.labels_))]
for j in range(len(db.clustering.labels_)):
    segment_field[j] = []
for j in range(len(db.clustering.labels_) - 1):
    if db.clustering.labels_[j] >= 0:
        segment_field[db.clustering.labels_[j]].append(j)
segment_field_direction = []
count = 0
# 分别求取每幅轨迹中的每块农田的两个分布最多的方向
for j in range(len(segment_field)):
    if len(segment_field[j]) > 0:
        afield = []
        for m in range(len(segment_field[j])):
            if lddata.direct[segment_field[j][m]] <= 360:
                afield.append(lddata.direct[segment_field[j][m]])
        count_direction_num = [0 for b in range(361)]
        for h in range(361):
            count_direction_num[h] = 0
        for h in range(len(afield)):
            count_direction_num[afield[h]] = count_direction_num[afield[h]] + 1
        maxnum = 0
        for h in range(len(count_direction_num)):
            if count_direction_num[h] > maxnum:
                maxnum = count_direction_num[h]
                mapindex = h
        count_direction_num[mapindex] = 0
        segment_field_direction.append(mapindex)
        maxnum = 0
        for h in range(len(count_direction_num)):
            if count_direction_num[h] > maxnum:
                maxnum = count_direction_num[h]
                mapindex = h
        count_direction_num[mapindex] = 0
        segment_field_direction.append(mapindex)
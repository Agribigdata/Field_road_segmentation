'''
create by zxq on 2020/11/1
'''
import time
import warnings
import loaddata as lddata
import getsegment as getseg
import get_score as gs
import getFRF as frf
import getRRR as rrr
import cluster_to_road as ctr
import parameter as parm
import fieldDirection as fieldDir
import dbscan as db
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    time_start = time.time()

    print('cluster...\n')
    picture_index=0
    print('cluster2road...\n')

    clustering = db.clustering

    clustering = ctr.cluster2road(picture_index, parm.c2r_base, parm.c2r_ratio, parm.c2r_diff, parm.c2r_dir_num, lddata.direct, fieldDir.segment_field_direction, clustering)


    allsegment, alldir, allspeed, allerrorpoint, allslope = getseg.getfirstsegment(picture_index, clustering, lddata.clean_x,lddata.clean_y, lddata.direct, lddata.origindata_time, lddata.origindata_id,lddata.origindata_x, lddata.origindata_y,lddata.origindata_speed)

    print('FRF...\n')
    clustering = frf.getroad2field(clustering,picture_index, allsegment, alldir, allspeed, allerrorpoint, allslope,lddata.clean_x,lddata.clean_y,lddata.newrow_tag,parm.FRF_speedcha,parm.FRF_dirthreshold,parm.FRF_dirnumthreshold)


    print('RRR...\n')
    rrr.getsameroad(clustering, picture_index, allsegment, alldir, allspeed, allerrorpoint, allslope, parm.RRR_speedthreshold,parm.RRR_dirthreshold,parm.RRR_dirnumthreshold)

    # 计算分数值
    gs.getscore(picture_index,clustering, lddata.turn_newrowtag, lddata.newrow_tag)
    time_end = time.time()
    print('\ntimecost:',time_end-time_start,'s')





"""
This script generates the noiseless Gabor images from scratch, using the Screen class.
The data will be stored as lmdb in data/ directory
"""


import Screen
import numpy as np
import lmdb, os
import caffe
import copy
import time
from multiprocessing import Pool

home_dir = ''

num = 50
size = 227
sigma = 50
ctrst  = 1.0

def more_clockwise(a,b):

    assert(0<=a<=180.0), 'First angle not in range'
    assert(0<=b<=180.0), 'Second angle not in range'

    if b<=90:
        return 0<=(a-b)<90, min(abs(a-b), (180-abs(a-b))) 
    else:
        return not (0<(b-a)<=90), min(abs(a-b), (180-abs(a-b))) 


for train_feat in ['angle']:

    noise_std = 0
    jitter_type = 'fix'
    center = np.array([0.0,0.0])
    center_j = np.array([0.0,0.0])
    angle_j = 1.0  if train_feat == 'angle' else 0.0
    ctrst_j = 0.05 if train_feat == 'ctrst' else 0.0
    wave_j  = 0.1  if train_feat == 'wave' else 0.0

    angle_list = np.arange(0.0,180.0,15.0)
    wave_list =  np.array([5.0,10.0,15.0,20.0,25.0,30.0,40.0,50.0,60.0,80.0])

    if  train_feat == 'angle':
        prec_list = np.array([0.5,1.0,2.0,5.0,10.0])
    elif train_feat == 'wave':
        prec_list = np.array([0.02,0.03,0.04,0.05,0.07,0.1])
        prec_list = np.array([0.02,0.03,0.05,0.07,0.1])

    for angle in angle_list:
        for wave in wave_list:
            for p3 in prec_list:
                exec(train_feat +'_j = float(p3)')
                print angle,wave
                exec('print '+train_feat+'_j')

                filename = home_dir + 'data/pair/noise_'+str(noise_std) + '_gabor_[{0}, {1}, {2}]_[[{3}, {4}], {5}, {6}, {7}]_jit_[[{8}, {9}], {10}, {11}, {12}]_{13}_{14}'.format(num, size, sigma, center[0], center[1], angle, ctrst, wave, center_j[0], center_j[1], angle_j, ctrst_j, wave_j, jitter_type, train_feat)
                print filename
                if os.path.isdir(filename):
                    print "exists"
                    # continue
                images = np.zeros((num,2,size, size), dtype=np.uint8)
                angles = np.zeros((num,2))
                centers= np.zeros((num,4))
                ctrsts = np.zeros((num,2))
                waves = np.zeros((num,2))

                sc_original = Screen.Gabor(sigma = sigma, center = center, size = size, angle = angle, ctrst = ctrst, wave = wave)
                def get_one_datum(it):

                    np.random.seed()
                    if it % 100 == 0:
                        print  train_feat+str(num) +', in which ' + str(it) + ' done'

                    sc = copy.deepcopy(sc_original)

                    pair = np.zeros( (2,sc.size, sc.size), dtype=np.uint8 )
                    # reference
                    center_ref = sc.center + 0.0
                    angle_ref = sc.angle + 0.0
                    ctrst_ref = sc.ctrst + 0.0
                    wave_ref = sc.wave + 0.0
                    ref = sc.image + 0.0

                    a = [0,0]
                    ct = [0,0]
                    ce = [None,None]
                    wa = [None,None]
                    a[0] = angle_ref
                    ce[0] = center_ref
                    ct[0] = ctrst_ref
                    wa[0] = wave_ref

                    ref += np.random.standard_normal(ref.shape) * noise_std
                    ref[ref>255] = 255
                    ref[ref<0] = 0
                    pair[0] = ref
                    # jittered

                    lab = it%2
                    if train_feat == 'angle':
                        min_th = 0.0
                        sc.one_side_jitter(angle_o = min_th*(2*lab-1), angle_j = (angle_j-min_th)*(2*lab-1), jitter_type = jitter_type)
                    elif train_feat == 'ctrst':
                        min_th = 0.0
                        sc.one_side_jitter(ctrst_o = min_th*(2*lab-1), ctrst_j = (ctrst_j-min_th)*(2*lab-1), jitter_type = jitter_type)
                    elif train_feat == 'wave':
                        min_th = 0.0
                        sc.one_side_jitter(wave_o  = min_th*(2*lab-1), wave_j  = (wave_j -min_th)*(2*lab-1), jitter_type = jitter_type)
                    else:
                        raise ValueError('can only train angle or ctrst')

                    image = sc.add_noise(noise_std)
                    a[1]  = sc.angle
                    ce[1] = sc.center
                    ct[1] = sc.ctrst/ct[0]
                    wa[1] = sc.wave/wa[0]
                    pair[1] = image + 0.0
                    ce = np.array(ce).flatten()

                    # add label
                    if train_feat == 'angle':
                        label, diff = more_clockwise(sc.angle, angle_ref)
                        # print sc.angle, angle_ref, diff, label
                        label = np.int(label)
                        it += 1
                        # print sc.angle, angle, label
                    elif train_feat == 'ctrst':
                        diff = abs(ctrst_ref - sc.ctrst)
                        it += 1
                        label = np.int(sc.ctrst>ctrst_ref)
                        #print sc.ctrst, ctrst_ref, lab, label
                    elif train_feat == 'wave':
                        diff = abs(wave_ref - sc.wave)
                        #print sc.wave
                        it += 1
                        label = np.int(sc.wave>wave_ref)
                        #print sc.ctrst, ctrst_ref, lab, label
                    else:
                        raise ValueError('can only train angle or ctrst')
                    assert lab == label

                    return pair, label, a, ce, ct, wa


                pool = Pool(10)
                out = pool.map(get_one_datum, range(num))
                dbfile = lmdb.open(filename, map_size = images.nbytes*2, sync=True, map_async = True, writemap = True)
                db = dbfile.begin(write=True)
                for it, o in enumerate(out):
                    pair = o[0]
                    label = o[1]
                    angles[it,:] = o[2]
                    centers[it,:]= o[3]
                    ctrsts[it,:] = o[4]
                    waves[it,:] = o[5]
                    datum = caffe.io.array_to_datum(pair, label)
                    str_id = '{:0>8d}'.format(it)
                    db.put(str_id, datum.SerializeToString()) 
                    if it % 5 == 0:
                        dbfile.sync()

                db.commit()
                dbfile.close()
                angles[angles>(angle+90.0)] -= 180.0
                print angles.mean(0), angles.std(0)
                print centers.mean(0), centers.std(0)
                print ctrsts.mean(0), ctrsts.std(0)
                print waves.mean(0), waves.std(0)
                pool.close()
                #print angles,centers,ctrsts

import facemorpher
import os, lmdb, shutil
import numpy as np
from scipy.ndimage import imread, filters
from scipy.misc import imsave, imresize
from multiprocessing import Pool

size = 227
background = 125

levels = [1.0,0.5,0.3,0.2,0.1]
mid_frame_n  = len(levels) + 1
num_frames = mid_frame_n * 2 - 1
N = 12

def make_percent_list(levels):

    nlevels = len(levels)

    levels = np.array(levels)
    pl = np.ones(nlevels*2+1)*0.5
    pl[:nlevels] += levels/2
    pl[nlevels+1:] -= levels[::-1]/2
    pl = pl.tolist()
    return pl

def make_new_dir(fn):

    if os.path.isdir(fn):
        shutil.rmtree(fn)
    os.makedirs(fn) 

def load_frame(fn):

    img = imread(fn, flatten=1)
    img = img[10:-10,10:-10]
    img = imresize(img, [size, size])
    img[img<5] = background
    img = filters.median_filter(img, 3)

    return img

percent_list = make_percent_list(levels)

all_male_photos = os.listdir('male')
all_female_photos = os.listdir('female')

npair = min(len(all_male_photos), len(all_female_photos))
np.random.seed(0)
for si in range(N):

    np.random.shuffle(all_male_photos)
    np.random.shuffle(all_female_photos)

    female_photos = all_female_photos[:npair]
    male_photos = all_male_photos[:npair]

    parts = ['train','test']

    for level in levels:
        db_folder = '_'.join(map(str, ['morph',level,si]))
        #make_new_dir(db_folder) 

    for ti, tmp_folder in enumerate(parts):
        print 'generating %s' % tmp_folder
        if ti == 0:
            males   = male_photos[:2*npair/3]
            females = female_photos[:2*npair/3]
        else: 
            males   = male_photos[2*npair/3:npair]
            females = female_photos[2*npair/3:npair]
        print len(males)*len(females)
        make_new_dir(tmp_folder)

        def one_male(mf):
            for ff in females: 
                out_dir = '_'.join((mf.split('.')[0], ff.split('.')[0]))
                pair = [os.path.join('male',mf), os.path.join('female', ff)]
                out_frames = os.path.join(tmp_folder,out_dir)
                print out_frames

                facemorpher.morpher(pair, percent_list = percent_list, 
                    out_frames = out_frames,  width = 226, height = 226)

        pool = Pool(20)
        pool.map(one_male, males)
        pool.close()

       

        all_frames  = [os.path.join(f[0],f1) for f in os.walk(tmp_folder) if len(f[1]) == 0 for f1 in f[2] ]
        all_frames  = all_frames
        print all_frames

        for li, level in enumerate(levels):
            print li, level
            db_folder = '_'.join(map(str, ['morph',level,si]))

            male_frames =   [f for f in all_frames if li+1 == int(f.split('/')[-1][5:8])]
            mid_frames  =   [f for f in all_frames if mid_frame_n == int(f.split('/')[-1][5:8])]
            female_frames = [f for f in all_frames if (num_frames-li) == int(f.split('/')[-1][5:8])]

            print len(male_frames) , len(female_frames),  len(mid_frames)
            assert len(male_frames) == len(female_frames) == len(mid_frames)
            nframes = len(male_frames)
            
            db_folder = db_folder+'/'+tmp_folder
            dbfile = lmdb.open(db_folder, map_size = nframes*2*size**2*4, sync=True)
            db = dbfile.begin(write=True)
            ct = 0
            
            idxes = range(0,nframes)

            img_std = 0.0
            img_ptp = 0.0

            for idx in idxes:

                male_f  = male_frames[idx]
                mid_f   = mid_frames[idx]
                female_f= female_frames[idx]

                male_img    = load_frame(male_f)
                mid_img     = load_frame(mid_f)
                female_img  = load_frame(female_f)

                img_std += male_img.std() + 0.0
                img_std += mid_img.std() + 0.0
                img_std += female_img.std() + 0.0
                img_ptp += male_img.ptp() + 0.0
                img_ptp += mid_img.ptp() + 0.0
                img_ptp += female_img.ptp() + 0.0
                ''' 
                for label, img in enumerate([male_img, female_img]):

                    pair = np.zeros((2,size, size), dtype=np.uint8)
                    pair[0] = mid_img
                    pair[1] = img

                    datum = caffe.io.array_to_datum(pair, label)
                    str_id = '{:0>8d}'.format(ct)
                    db.put(str_id, datum.SerializeToString()) 
                    if ct % 5 == 0:
                        dbfile.sync()
                    ct += 1

                ''' 
            print db_folder, ct

            db.commit()
            dbfile.close()



"""
For a given weight snapshot, compute various tuning properties defined in
in WeightProperties.py
"""


from WeightsProperties import *
from CaffeVPLUtils import *
import fnmatch

weights_file = "A_angle_d_angle_['angle', 'wave']_*_15_[*.0_10.0_0.2]_[0.0_0.0]_20_0.0001_0.9_0.0_0_1000_6"
nfile_batch = 6
file_batch_number = 0

blobs = ['norm1','norm2','conv3', 'conv4', 'conv5']
nangles = 100
rep = 50
batch_size = 100

w_files = get_weights_files(weights_file, w_dir = 'models/')
nfiles = len(w_files)
assert nfiles % nfile_batch == 0
file_batch_size = nfiles/nfile_batch
w_files = w_files[file_batch_number*file_batch_size:(file_batch_number+1)*file_batch_size]

wf0 = "models/AlexNet/weights_1c.caffemodel"
for wi, wf in enumerate(w_files):

    tf = TuningFeatures(wf, full_cov=True)
    print wi, len(w_files)
    print 'trained'
    tf.compute_tuning(blobs, threshold = 1.0, nangles = nangles, rep = rep, batch_size = batch_size, force_replace=True)
    
    noise = tf.attrs['noise']
    angle = tf.attrs['angle']
    wave  = tf.attrs['wave']
    ctrst = tf.attrs['ctrst']

    print 'untrained'
    tf0 = TuningFeatures(wf0, ctrst = ctrst, noise = noise, angle = angle, wave = wave, full_cov=True)
    tf0.compute_tuning(blobs, threshold = 1.0, noise = noise, angle = angle, wave = wave, ctrst = ctrst, 
        nangles = nangles, rep = rep,  batch_size = batch_size, force_replace=True)


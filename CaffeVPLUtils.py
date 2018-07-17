#from CaffeVPL import *
import time, lmdb, os
import cPickle as pkl
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array
import matplotlib.pyplot as plt
from collections import OrderedDict
import caffe
from Screen import *
from scipy.ndimage.filters import gaussian_filter1d
import fnmatch
import pandas as pd

home_dir = '/mnt/ssd/tmp/kevinli/Code/vpl/'
if not os.path.isdir(home_dir):
    home_dir = ''

def get_tuning_curves(net, angles, blobs, rep=1, noise=20, pool = False, sigma = 50, wave = 20, ctrst = 0.2):

    nangles = len(angles)
    centers = dict(map(lambda ln: (ln, (net.blobs[ln].data.shape[3]-1)/2), \
        filter(lambda bn: bn[0:2] not in ['fc', 'vp'], blobs)))

    nfilters = map(lambda b: net.blobs[b].data.shape[1], blobs)
    resps = map(lambda n: np.zeros((nangles, n)), nfilters)
    resps = OrderedDict(zip(blobs, resps))
    resps2 = map(lambda n: np.zeros((nangles, n, n)), nfilters)
    resps2 = OrderedDict(zip(blobs, resps2))
    sc = Gabor(size = 227, sigma = sigma, wave = wave, center=np.array([0.0,0.0]), angle = 0,ctrst = 1.0)
    batch = sc.batch_change(nangles, angle_j = angles)
    t_start = time.time()
    for i in range(rep):
        if i % 10 == 0: 
            print i, time.time() - t_start
        batch_noise = (batch - 125) * ctrst  + np.random.randn(*batch.shape)*noise
        out = net.forward_all(data=batch_noise, blobs = blobs);
        # neg_out = net.forward_all(data=-batch_noise, blobs = blobs);

        for l in blobs:
            # check if it is a convolution layer
            if  net.blobs[l].data.ndim==4:
                # r = np.maximum(out[l][:,:,centers[l], centers[l]], neg_out[l][:,:,centers[l], centers[l]])
                if pool:
                    r = out[l].max((2,3)).astype(np.float64)
                else:
                    r = out[l][:,:,centers[l], centers[l]].astype(np.float64)
            else:
                # r = np.maximum(out[l], neg_out[l])
                r = out[l]
            resps[l] += r
            resps2[l] += r[:,np.newaxis,:] * r[:,:,np.newaxis]
    for l in blobs:
        resps[l] /= rep
        resps2[l] /= rep
        resps2[l] = resps2[l] - resps[l][:,np.newaxis,:] * resps[l][:,:,np.newaxis]
    return resps, resps2

def smooth_gradient(r, axis = 0, sigma = 3): 
    '''
    compute the gradient filtered by Gaussian of width sigma (3) along axis (0)
    '''
    return gaussian_filter1d(r, sigma = sigma, axis = axis, order = 1, mode='wrap')

def smoothen(r, axis = 0, sigma = 3): 
    '''
    compute the gradient filtered by Gaussian of width sigma (3) along axis (0)
    '''
    return gaussian_filter1d(r, sigma = sigma, axis = axis, mode='wrap')

def logiters(s, e, num):
    iters = np.logspace(np.log10(s), np.log10(e), num, dtype = int)
    iters = np.hstack([0, iters])
    for i in range(num-1):
        while iters[i+1] <= iters[:i+1].max():
           iters[i+1]+=1 
    iters[-1] = e
    return iters

def learn_layer(net_spec, names):
    map(lambda l: plastic_layer(l), filter(lambda x:x.name.split('_')[0] in names, net_spec.layer))
    map(lambda l: freeze_layer(l), filter(lambda x:x.name.split('_')[0] not in names, net_spec.layer))

    
def freeze_layer(l):
    if l.type in ['Convolution', 'InnerProduct']:
        l.param[0].lr_mult = 0
        l.param[0].decay_mult = 0
        l.param[1].lr_mult = 0
        l.param[1].decay_mult = 0
        # print l.name + ' is frozen'

def plastic_layer(l, r = 1.0):
    if l.type in ['Convolution', 'InnerProduct']:
        l.param[0].lr_mult = 1 * r
        if l.name in ['vpl', 'vpl_p']:
            l.param[0].decay_mult = 1 * r
            l.param[1].lr_mult = 0 * r
        else:
            l.param[0].decay_mult = 0 * r
        l.param[1].lr_mult = 2 * r
        l.param[1].decay_mult = 0
        # print l.name + ' is plastic'


def default_solver():
    s = caffe_pb2.SolverParameter()

    s.test_initialization = True
    
    s.test_interval = 10000000

    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = 1
    
    s.max_iter = 100000     # # of times to update the net (training iterations)
    
    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = 'SGD'

    # Set the initial learning rate for SGD.
    s.base_lr = 1e-3

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    s.lr_policy = 'step'
    s.gamma = 0.5
    s.stepsize = 200

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # eighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 1e-4

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 10

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- ten times during training.
    s.snapshot = 100000000
    s.random_seed = -1
    s.snapshot_prefix = str('models/snapshots/')

    # Train on the GPU.  Using the CPU to train large networks is very slow.
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    
# Write the solver to a temporary file and return its filename.
    return s


def test_net(net, data_size, nfold = 1):
    # test a loaded(net with prespecified data directory and a known size of database
    block_size = net.blobs['data'].data.shape[0]
    niter = data_size/nfold/block_size
    
    acc = np.zeros((nfold, niter))
    loss = np.zeros((nfold, niter))
    n_acc = np.zeros((nfold, niter))
    dp = np.zeros(nfold)
    for i in range(nfold):
        acc[i], loss[i] , dp[i], n_acc[i] = test_net_nbatch(net, niter)
    

    acc = acc.mean(1)
    loss = loss.mean(1)
    n_acc = n_acc.mean(1)

    return acc, loss, dp, n_acc


def test_net_nbatch(net, niter):
    # test a loaded test_net with prespecified data directory for a number of batches
    acc = np.zeros(niter)
    n_acc = np.zeros(niter)
    loss= np.zeros(niter)
    fc = np.zeros((0,2))
    labels = np.array([])
    for i in range(niter):
        net.forward()
        # acc[i]= net.blobs['acc'].data.copy()
        loss[i]= net.blobs['loss'].data.copy()
        l = net.blobs['label'].data.copy()
        acc[i] = get_noisy_acc(net.blobs['prob'].data, l)
        labels = np.hstack((labels,l))
        fc_iter = net.blobs['fc8'].data.copy()
        if fc_iter.shape[1] == 1:
            fc_iter = np.hstack((1-fc_iter, fc_iter))
        fc = np.vstack((fc, fc_iter))
    dp, n_acc = get_dprime(fc, labels)
    return acc, loss, dp, n_acc

'''
def test_net_dis(net, data_size, nfold = 1):
    # test a loaded(net with prespecified data directory and a known size of database
    block_size = net.blobs['data'].data.shape[0]
    niter = data_size/nfold/block_size
    
    acc = np.zeros((nfold, niter))
    loss = np.zeros((nfold, niter))
    dp = np.zeros(nfold)
    for i in range(nfold):
        acc[i], loss[i] , dp = test_net_nbatch(net, niter)

    acc = acc.mean(1)
    loss = loss.mean(1)

    return acc, loss, dp


def test_net_dis_nbatch(net, niter):
    # test a loaded test_net with prespecified data directory for a number of batches
    acc = np.zeros(niter)
    loss= np.zeros(niter)
    dp  = np.zeros(niter)
    labels = net.blobs['label'].data.copy()
    labels = labels.reshape(labels.shape[0])
    fc = np.zeros((0,2))
    for i in range(niter):
        net.forward()
        acc[i]= net.blobs['acc'].data.copy()
        loss[i]= net.blobs['loss'].data.copy()
        fc_iter = net.blobs['fc8'].data.copy()
        fc = np.vstack((fc, fc_iter))
    return acc, loss, dp
'''

def get_weights_change(net, wi, layers, iters):
    # compute weights change in norm with respect to iteration 0
    # from a loaded net, weights initial for the set of training, set of layers and iters
    caffe.set_mode_cpu()
    w_0_file = make_weights_file(wi, iters[0])
    net.copy_from(w_0_file)
    w_0 = OrderedDict((ln, np.hstack((net.params[ln][0].data.flatten(), \
                                    net.params[ln][1].data.flatten())))
                       for ln in layers)
    dw = OrderedDict((ln, np.zeros(len(iters))) for ln in layers)
    for si, it in enumerate(iters):
        weights_file = make_weights_file(wi, it)
        net.copy_from(weights_file)
        for k, v in dw.iteritems():
            w = np.hstack((net.params[k][0].data.flatten(), net.params[k][1].data.flatten()))
            v[si] = np.linalg.norm(w-w_0[k])
    caffe.set_mode_gpu()
    return dw






def test_net_2(net, images, labels, batch_size, nfold = 1):

    assert images.shape[0] == labels.shape[0], 'number of images does not match number of labels'
    data_size = labels.shape[0]
    niter = data_size/nfold/batch_size
    assert niter > 0
    block_size = data_size/nfold

    images -= images.mean((2,3),keepdims = True)
    
    acc = np.zeros(nfold)
    n_acc = np.zeros(nfold)
    loss = np.zeros((nfold, niter))
    dp = np.zeros(nfold)
    n_dp = np.zeros(nfold)
    count = 0
    for i in range(nfold):
        fc = np.zeros((block_size,2))
        niter_count = 0
        for j in range(niter):
            image_batch = images[count:count+batch_size]
            label_batch = labels[count:count+batch_size]

            net.set_input_arrays(image_batch, label_batch)
            net.forward()

            # acc[i,j] = net.blobs['acc'].data.copy()
            # acc[i,j] = get_noisy_acc(net.blobs['prob'].data, label_batch)
            loss[i,j] = net.blobs['loss'].data.copy()
            fc[niter_count:niter_count+batch_size,:] = net.blobs['fc8'].data.copy()

            niter_count += batch_size
            count += batch_size

        acc[i], dp[i], n_acc[i], n_dp[i] = get_dprime(fc, labels[i*block_size:(i+1)*block_size])
    # acc = acc.mean(1)
    # n_acc = n_acc.mean(1)
    loss = loss.mean(1)

    return acc,loss, dp, n_acc, n_dp

def get_dprime(fc, labels):

    n = len(labels)
    pos = fc[range(n),np.int8(labels)]
    dm = np.abs(pos.mean())
    ss = pos.std() + 1e-15
    dp = dm/ss

    fc1 = fc[:,1]
    acc = np.exp(2*fc1)/(1+np.exp(2*fc1))
    acc = np.c_[acc,1-acc]
    acc1 = acc[range(n),np.int8(labels)]
    acc2 = acc[range(n),1-np.int8(labels)]
    acc = np.maximum(acc1.mean(), acc2.mean())

    fc_ub = fc[:,1]
    fc_ub -= fc_ub.mean()

    dp_ub = np.c_[-fc_ub,fc_ub]
    pos = dp_ub[range(n),np.int8(labels)]
    dm = np.abs(pos.mean())
    ss = pos.std() + 1e-15
    dp_ub = dm/ss

    acc_ub = np.exp(2*fc_ub)/(1+np.exp(2*fc_ub))
    acc_ub = np.c_[acc_ub,1-acc_ub]
    acc1 = acc_ub[range(n),np.int8(labels)]
    acc2 = acc_ub[range(n),1-np.int8(labels)]
    n_acc = np.maximum(acc1.mean(), acc2.mean())


    return acc, dp, n_acc, dp_ub

def get_noisy_acc(p, label):

    assert len(p) == len(label)
    if p.shape[1] == 1:
        p = np.hstack((1 - p, p))
    n = len(p)
    acc1 = p[range(n),np.int8(label)]
    acc2 = p[range(n),1-np.int8(label)]
    acc = np.maximum(acc1.mean(), acc2.mean())
    return acc
    
# not in use
def get_results_2(train_noise, ln, train_center, train_angle, train_ctrst, train_wave,
                               train_center_j, train_angle_j, train_ctrst_j, train_wave_j, 
                               train_jt, test_noise, fn, test_jt, test_jv, subdir = '', suffix=''):
    assert train_center.shape == (2,)
    assert train_center_j.shape == (2,)
    train_data = make_data_2(train_noise, train_center, train_angle, train_ctrst, train_wave,
                                          train_center_j, train_angle_j, train_ctrst_j, train_wave_j,
                                          train_jt, subdir = subdir, suffix = suffix)
    train_data += '_train'      # training data always has this suffix
    train_data_file = train_data.split('/')[-1]
    filename = 'results/'+subdir+'TRAIN_' + '_'.join((ln, train_data_file))
    filename += '_TEST_' + '_'.join((str(test_noise),fn, test_jt, str(test_jv)))
    assert os.path.isfile(filename), 'RESULTS FILE DOES NOT EXIST: '+ filename
    data = pkl.load(open(filename))
    ds = ['acc', 'loss', 'dp']
    acc, loss, dp = (data[d] if d in data else None for d in ds)
    snap = data['snap'] if 'snap' in data else 1
    return acc, loss, dp, snap

# not in use
def get_results(tnoise, ln, tfn, tjt, noise, fn, jt, extra = False):
    filename = 'results/TRAIN_'+'_'.join((str(tnoise), ln, tfn, tjt))+'_TEST_'+'_'.join((noise, fn, jt))
    if extra:
        filename = 'TRAIN_'+'_'.join((str(tnoise), ln, tfn, tjt, 'extra'))+'_TEST_'+'_'.join((noise, fn, jt, 'extra'))
    assert os.path.isfile(filename), 'RESULTS FILE DOES NOT EXIST: '+ filename

    data = pkl.load(open(filename))
    ds = ['acc', 'loss', 'dp']
    acc, loss, dp = (data[d] if d in data else None for d in ds)
    snap = data['snap'] if 'snap' in data else 1
    return acc, loss, dp, snap

def data_size(path):

    # number of samples in a LMDBa
    env = lmdb.open(path)
    size = int(env.stat()['entries'])
    return size



def make_data_2(noise=0, size = 227, sigma = 100, 
                center=np.array([0.0,0.0]), angle=0.0, ctrst=0.5, wave = 10.0,
                center_j=np.array([0.0,0.0]), angle_j=0.0, ctrst_j=0.0, wave_j = 0.0, 
                jitter_type = 'Gaussian', num = 100, label_feat='', phase = '', subdir= ''):

    center = np.array(center)
    center_j = np.array(center_j)

    datafile = home_dir + 'data/' +subdir+ 'noise_'+str(noise) + '_gabor_[{0}, {1}, {2}]_[[{3}, {4}], {5}, {6}, {7}]_jit_[[{8}, {9}], {10}, {11}, {12}]_{13}'.format(num, size, sigma, center[0], center[1], angle, ctrst, wave, center_j[0], center_j[1], angle_j, ctrst_j, wave_j, '_'.join((jitter_type, label_feat)))

    if phase != '': datafile += '_'+phase

    assert os.path.isdir(datafile), 'DATA DIR DOES NOT EXIST: '+datafile
    return str(datafile)

def data_file_params(fn):

    # parse data file to parameters, centers, etc.
    if '/' in fn: fn = fn.split('/')[-1]
    ps = fn.split('_')
    assert(len(ps) >=9), 'file name is out-dated: ' + fn

    pd = OrderedDict()
    pd['noise'] = int(ps[1])
    pd['stimulus_type'] = ps[2]
    num_size = map(int, ps[3][1:-1].split(', '))
    pd['num'] = num_size[0]
    pd['size'] = num_size[1]
    pd['sigma'] = num_size[2]
    
    ref_param = map(float,ps[4][1:-1].translate(None,'[]').split(', '))
    pd['center'] = np.array(ref_param[0:2])
    pd['angle']  = ref_param[2]
    pd['ctrst']  = ref_param[3]
    pd['wave']  = ref_param[4]


    jit_param = map(float,ps[6][1:-1].translate(None,'[]').split(', '))
    pd['center_j'] = np.array(jit_param[0:2])
    pd['angle_j']  = jit_param[2]
    pd['ctrst_j']  = jit_param[3]
    pd['wave_j']   = jit_param[4]

    pd['jitter_type'] = ps[7]
    pd['label_feat'] = ps[8]

    pd['phase'] = '' if len(ps) == 9 else ps[9]
    return pd


# not in use
def make_data(noise = None, fn = None, fv = None, jn = None, jv = None, jitter_type='Gaussian'):

    noise = 10 if noise is None else noise

    test_center, test_angle, test_ctrst = '-60.0', '15.0', '0.6'
    angle_j, center_j, ctrst_j = '0.0','0.0','0.0'

    if fn == 'angle':
        test_angle = str(fv)
    elif fn == 'center':
        test_center = str(fv)
    elif fn == 'ctrst':
        test_ctrst = str(fv)

    if jn == 'center':
        center_j = str(jv)
    elif jn == 'angle':
        angle_j = str(jv)
    elif jn == 'ctrst':
        ctrst_j = str(jv) 

    datafile =  'data/noise_'+str(noise)+'_gabor_[500, 227]_[['+test_center+', '+test_center+'], '+test_angle + ', '+test_ctrst+']_jit_[['+center_j+', '+center_j+'], '+angle_j+', '+ctrst_j+']_'+jitter_type

    assert os.path.isdir(datafile), 'DATA DIR DOES NOT EXIST' 
    return str(datafile)
        
def make_weights_file(train_desc, it, prefix = ''):

    # compose weight file without snapshots_dir prefix

    weights_file = prefix + '_'.join( (train_desc, 'iter', str(it) ) ) + '.caffemodel'
    return str(weights_file)

    



def get_net_spec(net_file, net_name = None):
    
    with open(net_file, 'r') as temp_def:
        temp_spec = caffe_pb2.NetParameter()
        text_format.Merge(str(temp_def.read()), temp_spec)

    net_spec = temp_spec.__deepcopy__()
    if net_name is not None:
        net_spec.name = net_name
    return net_spec


def scale_lr(net, net_spec, r):

    layers = filter(    lambda l:l.type in ('Convolution', 'InnerProduct'),
                           net_spec.layer)
    for l in layers:
        stds = (net.params[l.name][0].data.std())
        stds = 0.02
        if stds == 0:
            stds = 0.02
        l.param[0].lr_mult *= (stds*r)
        print l.name +' ' + str(l.param[0].lr_mult)


def get_weights(net_file, weights):
    
    caffe.set_mode_cpu()
    net = caffe.Net(net_file, weights, caffe.TEST)
    w = net.params.copy()
    caffe.set_mode_gpu()
    return w

def read_lmdb(db_file, n = 1000):

    assert os.path.isdir(db_file), "dbfile does not exist"

    lmdb_env = lmdb.open(db_file)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()
    size = int(lmdb_env.stat()['entries'])
    size = min(size, n)
    batch_image = [None]*size
    batch_label = [None]*size
    count = 0
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        batch_label[count] = datum.label
        batch_image[count] = datum_to_array(datum)
        count += 1
        if count == n:
            break
    
    batch_image = np.array(batch_image, dtype=np.float32)
    batch_label = np.array(batch_label, dtype=np.float32)
    lmdb_env.close()
    return batch_image, batch_label


def retrieve(dtstr=None):
    if dtstr is None:
        return [], OrderedDict({}), []

    results_file = []
    weights_files = OrderedDict({})
    FI_file      = []

    for root, dirs, files in os.walk('results'):
        for fn in files:
            if fn in dtstr:
                results_file = os.path.join(root, fn)
    for root, dirs, files in os.walk('models'):
        for fn in files:
            if dtstr in fn:
                it = fn.split('.')[-2]
                it = it.split('_')[-1]
                it = int(it)
                weights_files[it] = os.path.join(root, fn)

    for root, dirs, files in os.walk(os.path.join('results','fisher')):
        for fn in files:
            if dtstr in fn:
                FI_file = os.path.join(root, fn)
    weights_files = OrderedDict(sorted(weights_files.items(), key=lambda x: x[0]))
    return results_file, weights_files, FI_file

def parse_w_filename(fn, var_param_name=None):
    
    file_attr = dict()
    attrs = fn.split('_')
    attrs = [s.translate(None, '[]') for s in attrs]

    if 'morph' not in fn:

        # merge d_angle
        if 'd' in attrs:
            d_idx = attrs.index('d')
            attrs[d_idx] += '_'+attrs[d_idx+1]
            del attrs[d_idx+1]
            
        file_attr['var_param_name'] = attrs[2]

        file_attr['network'] = attrs[5]
        file_attr['train_feat'] = attrs[1]
        file_attr['noise'] = int(attrs[5])
        file_attr['angle'] = float(attrs[6])
        file_attr['wave'] = float(attrs[7])
        file_attr['ctrst'] = float(attrs[8])

        file_attr['d_angle'] = float(attrs[9])
        file_attr['d_wave'] = float(attrs[10])
        
        file_attr['batch_size'] = int(attrs[11])
        file_attr['base_lr'] = float(attrs[12])
        file_attr['momentum'] = float(attrs[13])
        file_attr['weight_decay'] = float(attrs[14])
        file_attr['niter']  = int(attrs[16])
        file_attr['version']  = int(attrs[17])
        file_attr['dtstr']  = str(attrs[18])
        if var_param_name is not None:
            file_attr[var_param_name]  = float(attrs[19])

    else:
        "A_morph_1_127_layer_[0.0,5.0,1.0]_20_0.2_20_0.0001_0_1000_6_03-29-10-52-45"
        file_attr['var_param_name'] = attrs[4]
        file_attr['level'] = int(attrs[2])
        file_attr['dset'] = int(attrs[3])
        file_attr['train_feat'] = 'face'
        file_attr['noise'] = int(attrs[6])
        file_attr['ctrst'] = float(attrs[7])
        
        file_attr['batch_size'] = int(attrs[8])
        file_attr['base_lr'] = float(attrs[9])
        
        #file_attr['momentum'] = float(attrs[10])
        #file_attr['weight_decay'] = float(attrs[11])
        
        file_attr['niter']  = int(attrs[11])
        file_attr['version']  = int(attrs[12])

    return file_attr

def get_weights_files(s, sort_by = None, w_dir = 'models'):
    
    '''
    to get weights files or tuning files, input string should have similar to fig file
    get_weights_files(s, sort_by = None, w_dir = 'models')
    '''
    
    if w_dir[-1] == '/':
        w_dir = w_dir[:-1]
    
    s = s.replace('[','?')
    s = s.replace(']','?')
    s = s.replace("'","?")
    s = s.replace(",","?")
    
    if '.caffemodel' in s:
        s = s[0:-11]
    
    s = '*' + s + '*'
    files = ['/'.join((f[0], f1)) for f in os.walk(w_dir) for f1 in f[2]]

    files = fnmatch.filter(files, s)
    
    return files

def load_weights_df(s, w_dir, var_param_name):
    
    files = get_weights_files(s, w_dir = w_dir)
    results = dict()
    
    for this_file in files:
        results.setdefault('w_file', [])
        results['w_file'].append(this_file)
        
        filename = this_file.split('/')[-1][:-11]
        attrs = parse_w_filename(filename, var_param_name = var_param_name)

        for attr in attrs:
            results.setdefault(attr, [])
            results[attr].append(attrs[attr])
            
    return pd.DataFrame(results)


#==========================
''' plotting functions '''
#=========================

def cool_spines(ax):
    ax.tick_params(reset = False, which = u'both', direction='out')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

def ax_xticks(ax, ticks, label, **kwargs):
    ax.set_xticks(ticks)
    ax.set_xticklabels(label, **kwargs)
    
def ax_yticks(ax, ticks, label, **kwargs):
    ax.set_yticks(ticks)
    ax.set_yticklabels(label, **kwargs)

def show_weights(net, ln, max_plots = 10):
    net.forward()
    weights = net.params[ln][0].data.copy()
    nweights = len(weights)
    nplots = min(nweights,max_plots) if max_plots > 0 else nweights
    
    plt.figure(figsize = (10,2*nplots))

    count = 1
    for pi in range(nplots):
        for ci in range(1):
            w = weights[pi,ci]
            plt.subplot(nplots,1,count)
            filt_min, filt_max = w.min(), w.max()
            plt.title("min: {0:.1f}, max: {1:.1f}".format(w.min(), w.max()))
            plt.imshow(w, vmin=filt_min, vmax=filt_max)
            plt.axis('off')
            count += 1

def show_2_weights(net, net2, ln, max_plots = 10):

    weights = net.params[ln][0].data.copy()
    weights2 = net.params[ln][0].data.copy()
    nweights = len(weights)
    nplots = min(nweights,max_plots) if max_plots > 0 else nweights
    
    plt.figure(figsize = (10,2*nplots))

    count = 1
    for pi in range(nplots):
        for ci in range(1):
            w = weights[pi,ci]
            plt.subplot(nplots,1,count)
            filt_min, filt_max = w.min(), w.max()
            plt.title("min: {0:.1f}, max: {1:.1f}".format(w.min(), w.max()))
            plt.imshow(w, vmin=filt_min, vmax=filt_max)
            plt.axis('off')
            count += 1

def plot_image(im):
    filt_min, filt_max = im.min(), im.max()
    plt.title("min: {0:.1f}, max: {1:.1f}".format(im.min(), im.max()))
    plt.imshow(im, vmin=filt_min, vmax=filt_max)
    plt.axis('off')

def plot_image_weights_resps(net, ln, im_idx=0, max_plots = 10):
    net.forward()
    weights = net.params[ln][0].data.copy()
    nweights = len(weights)
    nplots = min(nweights,max_plots) if max_plots > 0 else nweights
    plt.figure(figsize = (12,2*nplots))
    count = 1
    
    im = net.blobs['data'].data[im_idx,0]
    
    for pi in range(nplots):
        
        plt.subplot(nplots,5,count)
        plot_image(im)
        count += 1
        
        for ci in range(3):
            w = weights[pi,ci]
            plt.subplot(nplots,5,count)
            plot_image(w)
            count += 1
            
        resp = net.blobs[net.top_names[ln][0]].data[im_idx, pi]
        plt.subplot(nplots,5,count)
        plot_image(resp)
        count += 1

def plot_image_conv_resps(net, ln, im_idx=0, max_plots = 9):
    net.forward()
    resps = net.blobs[net.top_names[ln][0]].data[im_idx]
    nresps = len(resps)
    nresps = min(nresps,max_plots) if max_plots > 0 else nresps
    plt.figure(figsize = (12,0.5*nresps))
    count = 1
    
    im = net.blobs['data'].data[im_idx,0]
    plt.subplot(np.ceil((nresps+1)/5.0),5,count)
    plot_image(im)
    count += 1
    
    for ri in range(nresps):
        resp = resps[ri]    
        plt.subplot(np.ceil((nresps+1)/5.0),5,count)
        plot_image(resp)
        count += 1


def plot_image_fc_resps(net, ln):
    net.forward()
    resps = net.blobs[net.top_names[ln][0]].data[im_idx]
    nresps = len(resps)
    nresps = min(nresps,max_plots) if max_plots > 0 else nresps
    plt.figure(figsize = (12,0.5*nresps))
    count = 1
    
    im = net.blobs['data'].data[im_idx,0]
    plt.subplot(np.ceil((nresps+1)/5.0),5,count)
    plot_image(im)
    count += 1
    
    for ri in range(nresps):
        resp = resps[ri]    
        plt.subplot(np.ceil((nresps+1)/5.0),5,count)
        plot_image(resp)
        count += 1


def plot_batch(batch, fv=None):
    n = batch.shape[0]
    n = min(n, 15)
    fig, axes = plt.subplots(1,n, figsize = (12,3))
    map(lambda i: axes[i].imshow(batch[i,0], vmin = 0, vmax = 255), range(n))
    if fv is not None:
        assert batch.shape[0] == fv.shape[0], 'number of images and feature values do not match.'
        map(lambda i: axes[i].set_title('{0:.1f}'.format(fv[i])), range(n))
    map(lambda i: axes[i].set_axis_off(), range(n))



def plot_v1_difference(w1, w2, top=10):

    if top==0: top = w1.shape[0]
    
    assert np.all(w1.shape == w2.shape), 'different weights'
    diff = np.linalg.norm(w1-w2, axis=(2,3))
    w1norm = np.linalg.norm(w1, axis=(2,3))
    diff = diff.reshape(-1)
    w1norm = w1norm.reshape(-1)
    ndiff = diff/w1norm
    if top >= 0:
        idx = [i[0] for i in sorted(enumerate(ndiff), key=lambda x:x[1])]
        idx = idx[-top:]
        idx.reverse()
    else:
        idx = range(min(-top,w1.shape[0]))
    top = abs(top)

    nrow = np.int(np.ceil(top/10.0))

    idx2 = np.unravel_index(idx, w1.shape[0:2])
    w1, w2 = w1[idx2], w2[idx2]
    for r in range(nrow):
        fig, axes = plt.subplots(3,10,figsize=(15,3))
        for i in range(10):
            if i+r*10 >= len(w1): 
                fig.delaxes(axes[0,i])
                fig.delaxes(axes[1,i])
                fig.delaxes(axes[2,i])
            else:
                w1i, w2i = w1[i+r*10], w2[i+r*10]
                lim = np.array([w1i, w2i])
                lim_low = lim.min()
                lim_hi  = lim.max()
                axes[0,i].imshow(w1i,vmin = lim_low, vmax = lim_hi, interpolation = 'none')
                axes[0,i].set_axis_off()
                axes[0,i].set_title('d/D={0:3.2g}'.format(ndiff[idx[i+r*10]]))
                axes[1,i].imshow(w2i,vmin = lim_low, vmax =  lim_hi)
                axes[1,i].set_axis_off()
                axes[2,i].imshow(w2i-w1i)
                axes[2,i].set_axis_off()

import sys
import matplotlib 
if sys.platform == 'linux2':
   matplotlib.use('Agg')
import matplotlib.pyplot as plt
import caffe
import os, h5py
import numpy as np
import cPickle as pkl
from IncrementalAlexNet import build_net
from CaffeVPLUtils import get_tuning_curves, smooth_gradient, smoothen
from collections import OrderedDict
from scipy.signal import argrelextrema
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import pearsonr
from copy import deepcopy

caffe.set_mode_gpu()

weights_dir = 'models/real/'

class TuningFeatures:

    nangles = None
    angles = None
    res    = None

    file_dir = None
    filename = None
    attrs = None
    blobs  = None

    raw_tc = None
    t      = None
    t_test = None

    sense_measures = None
    full_cov = None

    def __init__(self, fn, noise = None, angle = None, wave = None, ctrst = None, full_cov = False):


        self.tuning_dir  = 'data/tuning'
        self.full_cov = full_cov

        if not os.path.isdir(self.tuning_dir):
            os.makedirs(self.tuning_dir)
        
        assert type(fn) is str
        
        if '/' not in fn:
            self.file_dir = weights_dir + fn
            self.filename = fn
        else:
            self.file_dir = fn
            self.filename = fn.split('/')[-1]
        assert os.path.isfile(self.file_dir)

        self.attrs = self.parse_filename(self.filename, 
            noise = noise, angle = angle, wave = wave, ctrst = ctrst)
        ''' 
        if self.filename != 'weights_1c.caffemodel':
            f = open('results/transfer/'+self.attrs['dtstr'], 'r')
            self.res = pkl.load(f)
            f.close()
        else:
            self.res = 'untrained'
        '''


    @staticmethod
    def w_norm(w, order, start_axis):
        ndim = w.ndim
        w = np.array(w)
        sum_dim = tuple(range(start_axis, ndim))
        return np.sum(np.abs(w)**order, sum_dim)**(1.0/order)

    def __sub__(self, other):

        return TuningDifferences(self, other)


    @staticmethod
    def parse_filename(fn, noise = None, angle = None, wave = None, ctrst = None):

        file_attr = dict()
        fn = fn.replace('.caffemodel', '')
        fn = fn.replace('.tuning', '')


        if fn == 'weights_1c':

            assert noise is not None
            assert angle is not None
            assert wave is not None
            assert ctrst is not None

            file_attr['network'] = 'AlexNet'
            file_attr['train_feat'] = 'untrained'

            file_attr['noise'] = int(noise)
            file_attr['angle'] = angle
            file_attr['wave'] =  wave
            file_attr['ctrst'] = ctrst

            file_attr['niter'] = 0
            file_attr['dtstr'] = ''
            file_attr['train_feat_val'] = 'untrained'
                
        else: 

            attrs = fn.split('_')
            attrs = [s.translate(None, '[]') for s in attrs]

            # merge d_angle
            if 'd' in attrs:
                d_idx = attrs.index('d')
                attrs[d_idx] += "_"+attrs[d_idx+1]
                del attrs[d_idx+1]
            
            iter_idx = attrs.index('iter')

            file_attr['network'] = attrs[0]
            file_attr['train_feat'] = attrs[1]
            file_attr['var_param_name'] = attrs[2]


            file_attr['noise'] = int(attrs[5])
            file_attr['angle'] = float(attrs[6])
            file_attr['wave'] = float(attrs[7])
            file_attr['ctrst'] = float(attrs[8])

            file_attr['d_angle'] = float(attrs[9])
            file_attr['d_wave'] = float(attrs[10])

            file_attr['niter'] = int(attrs[iter_idx+1])
            file_attr['dtstr'] = attrs[iter_idx-2]
            file_attr[ file_attr['var_param_name'] ] = float(attrs[iter_idx-1])

        return file_attr

    def load_tuning(self):
        '''
        load tuning data from HDF5
        '''
        
        if 'weights_1c' in self.filename:
            
            
            angle = self.attrs['angle']
            wave  = self.attrs['wave']
            ctrst = self.attrs['ctrst']
            noise = self.attrs['noise']
            
            tuning_file_dir = self.tuning_dir + '_'.join(map(str, ['weights_1c', 
                                                              noise, 
                                                              '['+str(angle), 
                                                              wave, 
                                                              str(ctrst)+']'] 
                                                       )
                                                   ) + '.tuning'
        else:
            tuning_file_dir = self.tuning_dir + self.filename.replace('.caffemodel', '.tuning')
        
        # print tuning_file_dir
        assert os.path.isfile(tuning_file_dir), 'need to compute tuning first'

        dset = h5py.File(tuning_file_dir, 'r')
        self.raw_tc = dset
        self.angles = dset.attrs['angles']
        self.blobs =  dset.attrs['blobs']
        self.nangles = len(self.angles)

    def compute_tuning(self, blobs, threshold = 0.0, batch_size = 100, nangles = 200, sigma = 50, 
                        angle = None, wave = None, ctrst = None, noise = None, 
                        rep = 100, ridge = 0.01, force_replace = False):
        if 'weights_1c' in self.filename:
            tuning_file_dir = self.tuning_dir + '_'.join(map(str, ['weights_1c', noise, '['+str(angle), wave, str(ctrst)+']'] )) + '.tuning'
            if os.path.isfile(tuning_file_dir):
                return 
        else: 
            tuning_file_dir = self.tuning_dir + self.filename.replace('.caffemodel', '.tuning')
        
        # confirm recomputation
        if os.path.isfile(tuning_file_dir):
            if not force_replace:
                print 'skipping '+tuning_file_dir
                return
            else: 
                os.remove(tuning_file_dir)

        angle = self.attrs['angle'] if angle is None else angle
        wave  = self.attrs['wave'] if wave is None else wave
        ctrst = self.attrs['ctrst'] if ctrst is None else ctrst
        noise = self.attrs['noise'] if noise is None else noise

        self.angles = np.linspace(-90,90,nangles)+angle
        self.nangles = nangles
        self.blobs  = blobs
        
        net_file, _ = build_net(batch_size = batch_size, classifier_name = None, top = blobs[-1])

        net = caffe.Net(net_file, caffe.TEST)
        os.remove(net_file)

        f = h5py.File(tuning_file_dir)

        # get all the weights


        # get tuning curve

        net.copy_from(self.file_dir)
        print 'computing tuning, nangles = %d, rep = %d...' % (nangles, rep)
        mean, cov = get_tuning_curves(net, self.angles, blobs, rep=rep, \
                    noise=noise, sigma = sigma, wave = wave, ctrst = ctrst)

        for b in blobs:

            layer = [k for k in net.params.keys() if k[-1] in b][0]

            neuron_idx = np.where(mean[b].mean(0) > threshold)[0]
            mean[b] = mean[b][:,neuron_idx]
            cov[b]  = cov[b][np.ix_(range(nangles), neuron_idx, neuron_idx)]
            cov[b] += np.eye(len(neuron_idx))[None,:,:] * ridge

            grp = f.create_group(b)
            grp.create_dataset("mean", data = mean[b])
            grp.create_dataset("var",  data = np.einsum('ijj->ij',cov[b]) )
            if self.full_cov:
                grp.create_dataset("cov",  data = cov[b])
            grp.create_dataset("neurons",data = neuron_idx)
            grp.create_dataset("w",    data = net.params[layer][0].data[neuron_idx].copy())

        del net

        f.attrs["angles"]  = self.angles
        f.attrs['nangles'] = nangles
        f.attrs['ctrst']   = ctrst
        f.attrs['noise']   = noise
        f.attrs['rep']     = rep
        f.attrs['blobs']   = blobs

        self.raw_tc = f
    
    def compute_features(self, sense_measures, smoothing = 1):

        # define a few sensitivity measures
        
        self.t = OrderedDict()
        self.sense_measures = sense_measures
        
        for b in self.blobs:
            nneurons = len(self.raw_tc[b+"/neurons"])
            r_mean = smoothen(self.raw_tc[b+"/mean"] ,sigma = smoothing)
            r_max  = r_mean.max(0)[None,:]
            #r_mean /= r_max
            r_var  = smoothen(self.raw_tc[b+"/var"]  ,sigma = smoothing)
            r_grad = smooth_gradient(self.raw_tc[b+"/mean"])
            if self.full_cov:
                r_cov  = self.raw_tc[b+"/cov"]
                t = dict((k, v(r_mean, r_var, r_grad, r_cov)) for k, v in sense_measures.items())
            else:
                t = dict((k, v(r_mean, r_var, r_grad)) for k, v in sense_measures.items())

            t['nneurons'] = nneurons
            t['neurons']  = np.array(self.raw_tc[b+'/neurons'])
            
            t['w1n']  = self.w_norm(np.array(self.raw_tc[b+'/w']), 1, 1)
            t['w2n']  = self.w_norm(np.array(self.raw_tc[b+'/w']), 2, 1)
            t['ori'] = self.angles[np.argmax(r_mean, axis=0)]
            self.t[b] = t

        return self.t

    def measure_test(self, test_angle, wl):

        t_test = dict()
        test_idx = np.argmax(self.angles>test_angle)
        for b in self.blobs:
            t_test[b] = dict((k, v.take( 
                            range(test_idx-wl, test_idx+wl+1, 1), axis = 0, mode = 'wrap').mean(0)) 
                       for k, v in self.t[b].items() if k in self.sense_measures.keys() and k != 'cov')
            t_test[b]['local_ori'] = self.local_peak(test_angle, self.t[b]['mean'], smoothing = 2)

        self.t_test = t_test

    def local_peak(self, test_angle, r_mean, smoothing = 1):

        angles = self.angles
        test_idx = np.argmax(angles>test_angle)
        nneurons= r_mean.shape[1]
        ori = np.zeros(nneurons)
        for i in range(nneurons):
            max_idx = argrelextrema(gaussian_filter1d(r_mean[:,i], smoothing, axis = 0, mode = 'wrap'), np.greater, axis = 0)[0]
            if len(max_idx) == 0:
                max_idx = [0]
            ori_idx = max_idx[np.argmin(np.abs(max_idx-test_idx))]
            ori[i] = angles[ori_idx]
        return ori

    def quick_test(self, blobs, threshold = 0.0, batch_size = 100, sigma = 50, 
                   angles = None, waves = None, ctrsts = None, noises = None, 
                   rep = 100): 
        
        angle = self.attrs['angle'] if angle is None else angle
        wave  = self.attrs['wave'] if wave is None else wave
        ctrst = self.attrs['ctrst'] if ctrst is None else ctrst
        noise = self.attrs['noise'] if noise is None else noise

        self.angles = np.linspace(-90,90,nangles)+angle
        self.nangles = nangles
        self.blobs  = blobs
        
        net_file, _ = build_net(batch_size = batch_size, classifier_name = None, top = blobs[-1])
        net = caffe.Net(net_file, caffe.TEST)
        mean, cov = get_tuning_curves(net, self.angles, blobs, rep=rep, \
                    noise=noise, sigma = sigma, wave = wave, ctrst = ctrst)
        return mean, cov

class TuningDifferences:
    
    def __init__(self, A, B):

        assert np.all(A.angles == B.angles)
        assert all(A.blobs  == B.blobs)
        assert A.t.keys() == B.t.keys()
        assert A.t_test.keys() == B.t_test.keys()
        assert A.sense_measures.keys() == B.sense_measures.keys()

        self.A_common = OrderedDict()
        self.B_common = OrderedDict()

        self.dt = OrderedDict()
        self.dt_test = OrderedDict()

        self.A = A; 
        self.B = B

        for b in A.t.keys():
            
            self.dt[b] = dict()

            # find common neurons in the two files
            neurons_1 = A.raw_tc[b+'/neurons']
            neurons_2 = B.raw_tc[b+'/neurons']

            common_1  = np.in1d(neurons_1, neurons_2)
            common_2  = np.in1d(neurons_2, neurons_1)

            self.A_common[b] = common_1
            self.B_common[b] = common_2

            for m in A.sense_measures.keys():
                t_dim = A.t[b][m].ndim 
                # single cell tuning information
                if t_dim == 2:
                    self.dt[b][m] = A.t[b][m][:,common_1] - B.t[b][m][:, common_2]
                # layer tuning information
                elif t_dim == 1:
                    self.dt[b][m] = A.t[b][m] - B.t[b][m]

            self.dt_test[b] = dict()
            for m in A.t_test[b].keys():      
                
                val = A.t_test[b][m]
                if isinstance(val, np.ndarray):
                    self.dt_test[b][m] = A.t_test[b][m][common_1] - B.t_test[b][m][common_2]
                else:
                    self.dt_test[b][m] = A.t_test[b][m] - B.t_test[b][m]
                    
                
            w1 = np.array(A.raw_tc[b+'/w'][common_1,:,:,:])
            w2 = np.array(B.raw_tc[b+'/w'][common_2,:,:,:])
            self.dt[b]['mean_abs_d'] = A.w_norm(w1 - w2, 1, 1)/np.prod(w1.shape[1:])
            self.dt[b]['sqrt_mean_sq_d'] = A.w_norm(w1 - w2, 2, 1)/np.sqrt(np.prod(w1.shape[1:]))
            self.dt[b]['rel_1norm'] = A.w_norm(w1 - w2, 1, 1)/A.w_norm(w2, 1, 1)
            self.dt[b]['rel_2norm'] = A.w_norm(w1 - w2, 2, 1)/A.w_norm(w2, 2, 1)
            self.dt[b]['d1n'] = A.t[b]['w1n'][common_1] - B.t[b]['w1n'][common_2]
            self.dt[b]['d2n'] = A.t[b]['w2n'][common_1] - B.t[b]['w2n'][common_2]



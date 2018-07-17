from sys import platform
import matplotlib
if platform != 'win32':
    matplotlib.use('Agg')
import caffe, numpy as np, matplotlib.pyplot as plt
import os
from caffe import layers as l, params as p
from caffe.proto import caffe_pb2
from google.protobuf import text_format
from tempfile import NamedTemporaryFile
from CaffeVPLUtils import *
import numpy as np
from fractions import gcd
import Screen
import time

plt.rcParams['figure.figsize'] = (10,10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

tmpdir = '/tmp'
caffe.set_mode_gpu()
class Learner:
    
    def __init__(self, pretrained_net, pretrained_weights = None, verbose=True):

        self.net_file = pretrained_net
        self.weights_file = pretrained_weights

        path = 'models/snapshots/'
        self.snapshot_dir = path
        self.solver_def = None
        self.solver = None
        self.net = None
        self.train_param = None
        self.test_param = None
        self.set_inspect(verbose=verbose)
        
    def load_weights(self, weights):
        path = self.snapshot_dir + weights
        assert os.path.isfile(path), 'WEIGHTS NOT FOUND: ' + path
        self.net.copy_from(path)
        self.weights_file = path
    
    @staticmethod
    def setup_noise_layer(net_spec, tp):
        noise_layer = filter(lambda l: l.name == 'input_noise', net_spec.layer)
        ctrst_layer = filter(lambda l: l.name == 'ctrst', net_spec.layer)
        if 'noise_param' not in tp:
            if len(noise_layer) == 0:
                return
            else: 
                tp['noise_param'] = dict(type='gaussian', std=0, ctrst=1.0)
        else:
            assert 'std' in tp['noise_param']
            assert 'ctrst' in tp['noise_param']

            assert len(noise_layer) == 1, 'no noise layer in prototxt file'
            noise_layer = noise_layer[0]
            for k, v in tp['noise_param'].items():
                if k == 'ctrst': continue
                noise_layer.dummy_data_param.data_filler[0].__setattr__(k,v)
            noise_layer.dummy_data_param.shape[0].dim[0] = tp['batch_size']

            assert len(ctrst_layer) == 1, 'no ctrst layer in prototxt file'
            ctrst_layer = ctrst_layer[0]
            ctrst_layer.scale_param.filler.value = tp['noise_param']['ctrst']

    def set_training(self, train_param):

        ''' 
        required train_param: 
            layer_names
            batch_size
            data
        '''

        ''' SET NET PARAMETERS  '''
        self.solver_def = default_solver()
        self.train_param = train_param

        # comput data size
        self.train_param['data_size'] = data_size(train_param['data'])


        layer_names = train_param['layer_names']
        
        net_spec = get_net_spec(self.net_file)

        net_spec.state.stage.append('train')
        net_spec.layer[0].data_param.batch_size = train_param['batch_size']
        net_spec.layer[0].data_param.source = train_param['data']

        self.setup_noise_layer(net_spec, train_param)

        if 'rand_skip' in train_param:
            net_spec.layer[0].data_param.rand_skip = train_param['rand_skip']
        if 'test_param' in train_param:
            test_param = train_param['test_param']
            net_spec.layer[1].data_param.batch_size = test_param['batch_size']
            net_spec.layer[1].data_param.source = test_param['data']
            test_param['data_size'] = test_param['data_size'] if 'data_size' in test_param \
               else  data_size(test_param['data']) 

            if 'test_interval' not in test_param:
                train_param['test_param']['test_interval'] = self.solver_def.test_interval

        learn_layer(net_spec, layer_names)
        weights_file = train_param['in_weights'] if 'in_weights' in train_param else \
            self.weights_file

        # scale learning rate according to weight std
        #scale_lr(self.net, net_spec, 1.0)    

        # write net definition
        net_file = NamedTemporaryFile(dir = tmpdir, prefix = 'net_', delete = False)
        net_file.write(text_format.MessageToString(net_spec))
        net_file.close()

        solver_file = NamedTemporaryFile(dir = tmpdir, prefix = 'sol_', delete = False)


        ''' SET SOLVER PARAMETERS '''

        for k, v in train_param.items():
            if hasattr(self.solver_def, k):
                self.solver_def.__setattr__(k, v)
        self.solver_def.train_net = net_file.name
        self.solver_def.snapshot_prefix = self.snapshot_dir + train_param['description']

        print 'TEST FILE NAME ============= ' + net_file.name
        if 'test_param' in train_param:

            # adding test_state to the test net so it load data from LMDB
            test_state = caffe_pb2.NetState()
            test_state.stage.append('test')
            self.solver_def.test_state.extend([test_state])
            if 'test_net' in test_param:
                self.solver_def.test_net.append(test_param['test_net'])
            else:
                self.solver_def.test_net.append(net_file.name)
            if 'data_size' in test_param:
             # Test on 100 batches each time we test.
                self.solver_def.test_iter.append(\
                    test_param['data_size']/test_param['batch_size'])

            if 'test_interval' in test_param:
                self.solver_def.test_interval = test_param['test_interval']
            else:
                test_param['test_interval'] = self.solver_def.test_interval
        solver_file.write(str(self.solver_def))
        solver_file.close()
        
        self.solver = None
        self.net = None
        self.solver = caffe.get_solver(solver_file.name)
            
        self.solver.net.copy_from(weights_file)

        if 'test_param' in train_param:
            self.solver.test_nets[0].copy_from(weights_file)
        os.remove(solver_file.name)
        os.remove(net_file.name)
        print 'Learning set with weights: ' + weights_file

    def set_testing(self, test_param):
        
        self.solver = None
        self.test_param = test_param

        # compute data size
        self.test_param['data_size'] =  test_param['data_size']\
            if 'data_size' in test_param else data_size(test_param['data'])

        net_spec = get_net_spec(self.net_file)
        net_spec.state.stage.append('test')

        net_spec.layer[1].data_param.batch_size = test_param['batch_size']
        net_spec.layer[1].data_param.source = test_param['data']

        self.setup_noise_layer(net_spec, test_param)
        
        # freeze all layers to prevent backwards
        learn_layer(net_spec, '')
        

        net_file = NamedTemporaryFile(dir = tmpdir, prefix = 'net_', delete = False)

        net_file.write(str(net_spec))
        net_file.close()

        if 'weights' in test_param:
            weights_file = test_param['weights'] 
        else:
            test_param['weights'] = self.weights_file
            weights_file = self.weights_file

        self.net = None
        self.net = caffe.Net(net_file.name, weights_file, caffe.TEST)

        os.remove(net_file.name)

        print 'NETWORK LOADED FOR TESTING WITH WEIGHTS: ' + weights_file
       
    def set_testing_2(self, test_param, verbose = True):

        # test_from memory data
        # should be faster than loading network every time

        self.solver = None
        self.test_param = test_param

        net_spec = get_net_spec(self.net_file)
        net_spec.state.stage.append('test_2')

        net_spec.layer[2].memory_data_param.batch_size = test_param['batch_size']
        
        self.setup_noise_layer(net_spec, test_param)
        
        # freeze all layers to prevent backwards
        learn_layer(net_spec, '')

        net_file = NamedTemporaryFile(dir = tmpdir, prefix = 'net_', delete = False)

        net_file.write(str(net_spec))
        net_file.close()

        self.net = None
        weights_file = test_param['weights'] if 'weights' in test_param else self.weights_file
        assert os.path.isfile(weights_file)
        self.net = caffe.Net(net_file.name, weights_file, caffe.TEST)
        os.remove(net_file.name)

        if verbose:
            print 'NETWORK LOADED FOR TESTING FROM "MEMORY DATA" WITH WEIGHTS: ' + self.weights_file

    def set_inspect(self, weights_file = None, verbose = True):

        weights_file = self.weights_file if weights_file is None else weights_file

        net_spec = get_net_spec(self.net_file) 
        net_spec.state.stage.append('inspect')

        net_file = NamedTemporaryFile(dir = tmpdir, prefix = 'net_', delete = False)
        net_file.write(str(net_spec))
        net_file.close()
        self.net = None
        self.net = caffe.Net(net_file.name.replace('\\','/'), weights_file, caffe.TEST)

        os.remove(net_file.name)
        if verbose:
            print 'NETWORK LOADED FOR INSPECTION WITH WEIGHTS: ' + weights_file


    def run_training(self, niter=50, load_weights = True, save_weights = False, \
                        save_final_weights = False, diff_layers = [], dw_funs = []):
        test_acc = None
        test_dp  = None
        test_loss  = None
        test_n_acc  = None
        snap_file = None
        esp = self.train_param['description']
        do_test = 'test_param' in self.train_param
        init_test = self.solver_def.test_initialization
        iters = self.train_param['iters']
        num_snaps = len(iters)
        if len(diff_layers) != 0:
            assert len(dw_funs) != 0
        if save_weights: save_final_weights = False
        if do_test:
        
            test_param = self.train_param['test_param'] 
            test_acc = np.zeros(num_snaps)
            test_n_acc = np.zeros(num_snaps)
            test_dp = np.zeros(num_snaps)
            test_loss = np.zeros(num_snaps)

            initial_acc, initial_loss, initial_dp, initial_n_acc = test_net(self.solver.test_nets[0], test_param['data_size'])
            initial_iter = self.solver.iter
            print '======= TEST: ' + esp + ',\n initialization: ' + str(self.solver.iter) + \
                ': {0:.3f}, {1:.3f}, {2:.3f}, {3:.3f}'.format(initial_acc[0], initial_loss[0], initial_dp[0], initial_n_acc[0])
            test_acc[0] = initial_acc
            test_n_acc[0] = initial_n_acc
            test_loss[0] = initial_loss
            test_dp[0] = initial_dp

        # weights change
        w_0 = OrderedDict((ln, np.hstack((self.solver.net.params[ln][0].data.flatten(), \
                                        self.solver.net.params[ln][1].data.flatten())))
                           for ln in diff_layers)
        dw = OrderedDict(
                            (ln, OrderedDict((fn, np.zeros(num_snaps)) for fn in dw_funs.keys()))
                        for ln in diff_layers)
        if save_weights:

            # 0 iteration weights and testing
            snap_file = self.solver_def.snapshot_prefix + '_iter_' + str(iters[0]) + '.caffemodel'
            snap_file = str(snap_file)
            print '======== SAVING WEIGHTS TO: ' + snap_file
            self.solver.net.save((snap_file))

        for ii, it in enumerate(iters[1:]):

            self.solver.step(it-iters[ii])

            # computingweights change
            for k, v in dw.iteritems():
                w = np.hstack((self.solver.net.params[k][0].data.flatten(), self.solver.net.params[k][1].data.flatten()))
                for fn, f in dw_funs.iteritems():
                    v[fn][ii+1] = f(w,w_0[k]) 

            snap_file = self.solver_def.snapshot_prefix + '_iter_' +\
                str(it) + '.caffemodel'
            snap_file = str(snap_file)
            
            # saving weights
            if save_weights:
                print '======== SAVING WEIGHTS TO: ' + snap_file
                self.solver.net.save((snap_file))
            # test network
            if do_test:

                acc, loss, dp, n_acc = test_net(self.solver.test_nets[0], test_param['data_size'])
                if acc == np.nan or n_acc == np.nan:
                    raise "nan!"
                print '======== TEST: ' + esp + ', \n\t iter: ' + str(it) + \
                ': {0:.3f}, {1:.3f}, {2:.3f}, {3:.3f}'.format(acc[0], loss[0], dp[0], n_acc[0])
                test_acc[ii+1] = acc
                test_n_acc[ii+1] = n_acc
                test_loss[ii+1] = loss
                test_dp[ii+1] = dp

        if save_final_weights:
            print '======== SAVING WEIGHTS TO: ' + snap_file
            self.solver.net.save(snap_file)
                    
        if load_weights: 
            self.set_inspect(snap_file)
            self.weights_file = snap_file

        if do_test:        
            return {'acc': test_acc, 'loss': test_loss, 'dp': test_dp, 'iters': iters, 'n_acc': test_n_acc}, snap_file, dw
        else: 
            empty = np.array([])
            return {'acc': empty, 'loss':empty, 'dp':empty, 'iters':empty, 'n_acc':empty}, snap_file, dw

    def run_testing(self, nfold = 1):
        tp = self.test_param 
        print 'TESTING SNAPSHOT: ' + tp['weights'] + ' ON DATA: ' + tp['test_desc']
        return test_net(self.net, self.test_param['data_size'], nfold)
        
    def run_testing_2(self, test_param, nfold = 1):

        tp = test_param
        # generate reference frame
        test_data = tp['data']
        batch_size = tp['batch_size']
        images, labels = read_lmdb(test_data, tp['data_size'])
        lmdb_data_size = len(labels)
        data_size = tp['data_size']

        if data_size > lmdb_data_size:
            rep = int(np.ceil(data_size*1.0/lmdb_data_size ))
            images = np.tile(images, [rep,1,1,1])
            labels = np.tile(labels, [rep])
        else:
            images = images[0:data_size]
            labels = labels[0:data_size]

        if 'weights' not in tp:
            tp['weights'] = self.weights_file
        if 'nfold' not in tp:
            tp['nfold'] = 1
        # print '   {0:20s}: {1:s}\n   {2:20s}: {3:s} '.format('TESTING SNAPSHOT', tp['weights'],'ON DATA', tp['test_desc'])
        acc, loss, dp, n_acc, n_dp = test_net_2(self.net, images, labels, batch_size, tp['nfold'])
        return acc, loss, dp, n_acc, n_dp

class Discriminator_pair(Learner):
    pass

class Identifier(Learner):
    pass

"""
This script runs simulations over all the 12 trained orientations 
for a particular set of other parameters, including noise, spatial
frequency (wavelength) contrast. The trained network
is then tested on transfer stimuli. 

The results of test and weight changes will be stored in results/ 
and a summary figure will be saved in figs/
"""

from CaffeVPL import *
import cPickle as pkl
from CaffeVPLUtils import *
from time import strftime, gmtime, time
from shutil import copyfile
from collections import OrderedDict

home_dir = ''

train_feat = 'angle'        # don't change!
var_param_name = 'd_'+train_feat
transfer_feats = ['angle', 'wave']
ref_feat = 'angle1'         # can be any feature
network = 'AlexNet'
version = 'conv5'

# ================================
# can choose between feature or noise
if var_param_name == 'd_angle':
    params = [10.0,5.0,2.0,1.0,0.5]              # FIXME
elif var_param_name == 'd_wave':
    params = [0.2,0.15,0.1]
    #params = [0.1,0.01]                     # FIXME
else:
    raise('d_angle not valid')

stim_size = 227
sigma = 50
center = 0.0

noise   = 15
angle1  = 0.0
wave1   = 40.0
ctrst1  = 0.5

#========================
# features to iterate over
# SEt up different features for the reference
if ref_feat == 'angle1':
    # ref_list = np.arange(0.0,180.0,45.0)
    ref_list = np.arange(0.0,180.0,15.0)    #FIXME
elif ref_feat == 'wave1':
    #ref_list = [10.0,20.0,30.0,40.0,50.0]
    ref_list = [10.0,20,30, 40.0]    #FIXME
elif ref_feat == 'ctrst1':
    ref_list = [0.1,0.2,0.3,0.4,0.5]
    #ref_list = [5.0,10.0]    #FIXME
else:
    raise('ref_feat not valid, can implement and generate data')

#========================
# Training parameters
momentum = 0.9
base_lr = 1e-4
weight_decay = 0.000
niter = 1000            #FIXME
num_iters = 20          #FIXME
keep_weights = 20 # keep every three iters
assert num_iters % keep_weights == 0
iters = logiters(1, niter, num_iters) if niter >= 10\
    else np.linspace(0, niter , niter+1, dtype=np.int)
num_iters = len(iters)

num = 50
test_num = 200          #FIXME
nfold = 10              #FIXME
batch_size = 20
test_batch_size = 20
solver_type = 'SGD'
assert test_num%(nfold*test_batch_size) == 0, 'adjust batch_size or nfold'

temp_def = str('models/'+network+'/pair_pipeline_'+version+'_sym.prototxt')
model_weights = str('models/'+network+'/weights_1c.caffemodel')
layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'vpl']

# ref_list = [1] # PLOT
for ref in ref_list:

    # ''' # PLOT
    exec(ref_feat + ' = %.10f' % ref)

    angle_j = 0.0
    wave_j = 0.0
    ctrst_j = 0.0

    # =====================
    # set train parameters
    # =====================
    train_noise = noise
    train_jt = str('fix')
    train_angle = angle1;      train_center = np.array([center,0.0]);   train_ctrst = ctrst1; train_wave = wave1
    train_angle_j = angle_j;   train_center_j = np.array([0.0,0.0]);    train_ctrst_j = 0.0; train_wave_j = wave_j

    # =====================
    # test parameters
    # =====================

    test_noise = train_noise
    test_jt = train_jt
    test_angle = angle1;      test_center = np.array([center,0.0]);      test_ctrst = ctrst1;   test_wave = wave1
    test_angle_j = angle_j;     test_center_j = np.array([0.0,0.0]);    test_ctrst_j = ctrst_j; test_wave_j = wave_j

    # =================
    # Weights change measure
    # ================= 

    dw_funs = OrderedDict([('mean_abs_d'        ,lambda w, w0: np.abs(w-w0).mean()),\
                          ('sqrt_mean_sq_d'     ,lambda w, w0: np.sqrt(np.mean((w-w0)**2))),\
                          ('rel_1norm'          ,lambda w, w0: np.sum(np.abs((w-w0)))/np.maximum(1, np.sum(np.abs(w0)))),\
                          ('rel_2norm'          ,lambda w, w0: np.linalg.norm(w-w0)/np.maximum(1, np.linalg.norm(w0)))\
                          ])

    init_niter = 0
    init_iters = np.array([0,init_niter])

    accs = [None]*len(params)
    n_accs = [None]*len(params)
    losses = [None]*len(params)
    dps = [None]*len(params)
    n_dps = [None]*len(params)
    dws = [None]*len(params)
    test_start = time()

    noise_param = dict(type='gaussian', std=train_noise, ctrst=train_ctrst)

    train_param = {
        'batch_size'    :   batch_size,
        'rand_skip'     :   batch_size-1,
        'iter_size'     :   1,
        'momentum'      :   momentum,
        'base_lr'       :   base_lr,
        'weight_decay'  :   weight_decay,
        'iters'         :   iters,
        'display'       :   0,
        'gamma'         :   1.0,
        'stepsize'     :    200,
        'type'          :   solver_type,
        'noise_param'   :   noise_param
    }

    ## test while training
    #test_angle_j = 45.0
    #test_data = make_data_2(0, stim_size, sigma,\
    #            test_center, test_angle, test_ctrst, test_wave,\
    #            test_center_j, test_angle_j, test_ctrst_j, test_wave_j,\
    #            test_jt, num = test_num, subdir = 'pair/', label_feat = train_feat)
    #test_param = {
    #        'data'      :   test_data,
    #        'data_size':    test_num,
    #        'batch_size':   test_batch_size,
    #}
    #train_param['test_param'] = test_param
   
    test_param = {
            'data_size':    test_num,
            'batch_size':   test_batch_size,
            'noise_param':  noise_param,
            'nfold':        nfold,
    }

    params_to_print = '[%.1f,%.1f,%.1f]' % (params[0], params[-1], params[1]-params[0])
    dtstr = strftime('%m-%d-%H-%M-%S', gmtime())
    filename = '_'.join((network[0],
        train_feat, var_param_name, str(transfer_feats), params_to_print,
        str(noise),
        '['+str(angle1),
        str(wave1),
        str(ctrst1)+']',
        '['+str(angle_j),
        str(wave_j)+']',
        str(batch_size), 
        str(base_lr), 
        str(momentum),
        str(weight_decay),
        str(init_niter), 
        str(niter), 
        str(len(layer_names)),
        dtstr
        ))
    # initialisation

    for pi,p in enumerate(params):
        
        if var_param_name == 'd_angle':
            train_angle_j = p
            test_angle_j  = p
        elif var_param_name == 'd_wave':
            train_wave_j = p
            test_wave_j  = p
        else:
            raise NameError('param not exist')

        #========================
        # feature to test transfer

        test_datas = []
        train_data = make_data_2(0, stim_size, sigma,\
                    train_center, train_angle, 1.0, train_wave, 
                    train_center_j, train_angle_j, train_ctrst_j, train_wave_j,
                    train_jt, num = num, subdir = 'pair/', label_feat = train_feat, phase = '')
        train_data_file = train_data.split('/')[-1]
        
        test_data = make_data_2(0, stim_size, sigma,\
                    test_center, test_angle, 1.0, test_wave,
                    test_center_j, test_angle_j, test_ctrst_j, test_wave_j,
                    test_jt, num = num, subdir = 'pair/', label_feat = train_feat)
        test_phase_names = transfer_feats[:]
        test_phase_names.insert(0, 'train')
        test_datas.append([test_data])

        if 'angle' in transfer_feats:
            
            test_angle_2, test_angle_3 = test_angle + 45.0, test_angle + 90.0
            if test_angle_2 >= 180:
                test_angle_2 -= 180
            if test_angle_3 >= 180:
                test_angle_3 -= 180

            test_data_2 = make_data_2(0, stim_size, sigma,
                        test_center, test_angle_2, 1.0, test_wave,
                        test_center_j, test_angle_j, test_ctrst_j, test_wave_j,
                        test_jt, num = num, subdir = 'pair/', label_feat = train_feat)

            test_data_3 = make_data_2(0, stim_size, sigma,
                        test_center, test_angle_3, 1.0, test_wave,
                        test_center_j, test_angle_j, test_ctrst_j, test_wave_j,
                        test_jt, num = num, subdir = 'pair/', label_feat = train_feat)
            test_datas.append([test_data_2, test_data_3])

        if 'wave' in transfer_feats:

            test_wave_2, test_wave_3 = test_wave*0.5, test_wave*2

            test_data_2 = make_data_2(0, stim_size, sigma,
                        test_center, test_angle, 1.0, test_wave_2,
                        test_center_j, test_angle_j, test_ctrst_j, test_wave_j,
                        test_jt, num = num, subdir = 'pair/', label_feat = train_feat)

            test_data_3 = make_data_2(0, stim_size, sigma,
                        test_center, test_angle, 1.0, test_wave_3,
                        test_center_j, test_angle_j, test_ctrst_j, test_wave_j,
                        test_jt, num = num, subdir = 'pair/', label_feat = train_feat)
            test_datas.append([test_data_2, test_data_3])

        test_phases = OrderedDict(zip(test_phase_names, test_datas))

        learner = Discriminator_pair(temp_def, model_weights)
        train_start = time()
        #==================
        #==================
        #   beginning of training
        #==================
        #==================

        train_param['data'] = train_data
        train_param['noise_param']['std'] = train_noise
        train_param['noise_param']['ctrst'] = train_ctrst
        train_desc = '_'.join((filename, str(p)))
        train_param['description'] = train_desc
        if init_niter != 0:
            train_layers = layer_names[:-1]
        else:
            train_layers = layer_names
        diff_layers = train_layers[:]

        train_param['layer_names'] = train_layers
        learner.set_training(train_param)
        print "========== PROGRESS: %s in %s, %s in %s" % (str(ref), str(ref_list), p, str(params))
        results, train_weights, dws[pi] = learner.run_training(niter = niter, load_weights = False, 
            save_weights = True, diff_layers = diff_layers, dw_funs = dw_funs)

        #==================
        #==================
        #   beginning of testing
        #==================
        #==================

        ## test_2, with memory data
        test_start = time()

        learner.set_testing_2(test_param)

        ncalls = 0
        accs[pi]={}
        n_accs[pi]={}
        losses[pi]={}
        dps[pi]={}
        n_dps[pi]={}
        
        for ln, test_data_group in test_phases.iteritems():

            ntest_data = len(test_data_group) 
            accs[pi][ln]   = np.zeros((ntest_data, num_iters, nfold))
            n_accs[pi][ln] = np.zeros((ntest_data, num_iters, nfold))
            losses[pi][ln] = np.zeros((ntest_data, num_iters, nfold))
            dps[pi][ln]    = np.zeros((ntest_data, num_iters, nfold))
            n_dps[pi][ln]  = np.zeros((ntest_data, num_iters, nfold))
            tpd = data_file_params(train_data_file)
            t_center, t_angle, t_wave = tpd['center'], tpd['angle'], tpd['wave']

            for ii, it in enumerate(iters):

                test_param['weights'] = make_weights_file(train_desc, it)
                learner.load_weights(test_param['weights'])

                for di, test_data in enumerate(test_data_group):

                    # reset parameters 
                    print "========== PROGRESS: %s in %s, %s in %s" % (str(ref), str(ref_list), p, str(params))
                    print "========== FILENAME: " + filename
                    print "========== TESTING MODE: param: {0}, test data: {1}, iter: {2} ==========".format(p, ln, it)
                    # set test fature value
                    print "   {5:20s}: noise: {0}, center: {1}, angle: {2}, ctrst: {3}, wave: {4}".format(\
                            test_noise, str(t_center.tolist()), t_angle, test_ctrst, t_wave, 'TRAIN FEATURE VALUES') 
                    # get data
                    test_data_file = test_data.split('/')[-1]
                    test_param['data'] = test_data
                    test_param['noise_param']['std'] = test_noise
                    test_param['noise_param']['ctrst'] = test_ctrst
                    test_param['test_desc'] = test_data_file

                    accs[pi][ln][di, ii], losses[pi][ln][di, ii], \
                            dps[pi][ln][di, ii], n_accs[pi][ln][di, ii], n_dps[pi][ln][di, ii]= learner.run_testing_2(test_param)
                    print "   {5:20s}: acc: {0:.3f}, loss: {1:.3f}, dp: {2:.3f}, n_acc: {3:.3f}, n_dp: {4:.3f}\n".format(
                            accs[pi][ln][di, ii].mean(), 
                            losses[pi][ln][di, ii].mean(), 
                            dps[pi][ln][di, ii].mean(), 
                            n_accs[pi][ln][di, ii].mean(),
                            n_dps[pi][ln][di, ii].mean(),'RESULT')

                    ncalls += 1
                if ncalls >=100:
                    test_param['weights'] = model_weights
                    learner.set_testing_2(test_param)
                    ncalls = 0

        # remove weights file
        if keep_weights > 0:
            for it in iters:
                if it not in iters[keep_weights::keep_weights]:
                    test_param['weights'] = make_weights_file(train_desc, it, prefix = learner.snapshot_dir)
                    os.remove(test_param['weights'])
        else:
            for it in iters:
                test_param['weights'] = make_weights_file(train_desc, it, prefix = learner.snapshot_dir)
                os.remove(test_param['weights'])
        del test_param['weights']
        del test_param['data']
        del test_param['test_desc']
                    
    with open('results/transfer/'+filename+'.results', 'w') as f:
        pkl.dump({'accs': accs, 'losses': losses, 'dps': dps, \
        'n_accs': n_accs, 'n_dps': n_dps,\
        'dws': dws, 'test_phases':test_phases,\
        'iters': iters, 'var_param_name': var_param_name,
        'params': params, 'dw_funs': dw_funs.keys(),
        'filename': filename}, f)
   
    email_file = '/nfs/home/kevinli/Code/mail/send_completion.py'
    if os.path.isfile(email_file):
        execfile(email_file, dict(filename=filename))

    # ''' # PLOT
    # p================
    # start of plotting
    #= ===============
  

    try:
        f = open('results/transfer/'+filename + '.results', 'r')
    except:
        f = open('results/transfer/03-23-11-54-41', 'r')
    rf = pkl.load(f)
    f.close()
    accs = rf['accs']
    n_accs = rf['n_accs']
    losses = rf['losses']
    filename= rf['filename']
    dps = rf['dps']
    n_dps = rf['n_dps']
    dws = rf['dws']
    iters = rf['iters']
    test_phases = rf['test_phases']
    var_param_name = rf['var_param_name']

    params = rf['params']
    nparams = len(params)

    dw_funs = rf['dw_funs']
    ndw_funs = len(dw_funs)

    layer_names = dws[0].keys()
    nlayers = len(layer_names)
    nplot_m = 2   # number of plots per measure at each iteration
    
    fig, axes = plt.subplots(nplot_m*ndw_funs+4*(len(test_phases)-1),nparams+1,figsize=(5*(nparams+1),4*(nplot_m*ndw_funs + 4*(len(test_phases)-1))))

    ti_cm = plt.get_cmap('Reds')
    dw_cm = plt.get_cmap('Greens')
    dwr_cm = plt.get_cmap('Blues')
    dp_dwr_cm = plt.get_cmap('Purples')

    dw_colors = [dw_cm( (1.*i + 1.)/(nlayers)) for i in range(nlayers-1)]
    dwr_colors = [dwr_cm( (1.*i + 1.)/(nlayers)) for i in range(nlayers-1)]
    ti_colors = [ti_cm( (1.*i + 1.)/(nparams)) for i in range(nparams)]
    cp_dwr_colors = [dp_dwr_cm( (1.*i + 1.)/(nparams)) for i in range(nparams)]

    def share_ylim(axes, min = None, max = None):
        ylims = np.array([a.get_ylim() for a in axes])
        ylims = [ylims[:,0].min(), ylims[:,1].max()]
        if min is not None:
            ylims[0] = min
        if max is not None:
            ylims[1] = max
        for a in axes:
            a.set_ylim(ylims[0], ylims[1])
    def set_data_ylim(ax, data, s = 0.05, min = None):
        data = np.array(data)
        ptp = data.ptp()
        if min is None:
            min = data.min()
            min -= ptp*s
        max = data.max()
        max += ptp*s
        ax.set_ylim(min, max)

    for fi, fn in enumerate(dw_funs):

        # weights change
        ylim = max(map(lambda dw: max(map(lambda d: d[fn].max(), dw.values()[:-1])), dws))*1.1
        for pi, p in enumerate(params):
            ax = axes[fi*nplot_m, pi] 
            ax.set_color_cycle(dw_colors)
            dw = dws[pi]
            dw_v = np.vstack(map(lambda w: w[fn], dw.values()[:-1])).T
            ax.plot(iters, dw_v, linewidth = 2)
            if pi == 0:
                ax.set_ylabel(fn + ' change from 0th iter')
            else:
                ax.set_yticklabels([])
            if fi == 0:
                ax.set_title(var_param_name+': '+str(p))
            ax.set_xlim([1,iters[-1]])
            ax.grid(True)
            ax.set_xscale('log')
            if np.any(dw_v>0):
                ax.set_yscale('log')
            set_data_ylim(ax, dw_v)
        share_ylim(axes[fi*nplot_m, :nparams])

        fig.delaxes(axes[fi*nplot_m,-1])
        
        # weights change proportions
        for pi, param in enumerate(params):

            ax = axes[fi*nplot_m+1,pi]
            dw = dws[pi]
            dw_niter = map(lambda dw_l: dw_l[fn][-1], dw.values())
            if pi != 0:
                ax.set_yticklabels([])
            bar_height = dw_niter[:-1]/np.sum(dw_niter[:-1])
            ax.bar(range(nlayers-1), bar_height, color = dw_colors, align='center')
            set_data_ylim(ax, dw_niter[:-1])
            ax.set_ylim([0,1.0])
            ax.set_xticks(range(nlayers-1))
            ax.grid(True, axis = 'y')
            if fi == ndw_funs-1:
                ax.set_xticklabels(layer_names[:-1], rotation='vertical')

            if pi > 0:

                prev_dw = dws[0]
                prev_dw_niter = map(lambda dw_l: dw_l[fn][-1], prev_dw.values())
                ax = axes[fi*nplot_m+1,-1]
                cross_param_dw_r = np.array(dw_niter)/np.array(prev_dw_niter)
                ax.plot(range(nlayers),cross_param_dw_r, 
                    label='%.2f:%.2f' % (params[pi],params[pi-1]),
                    color = cp_dwr_colors[pi], lw = 2)
                ax.set_xticks(range(nlayers))
                ax.set_yscale('log')
                ax.grid(True)
                if fi == ndw_funs-1:
                    ax.set_xticklabels(layer_names[:-1], rotation='vertical')
        share_ylim(axes[fi*nplot_m+1,:-1])
        ax.legend(bbox_to_anchor=(1,2))

    # time course of accuracy
    for pi, p in enumerate(params):
        for li, ln in enumerate(test_phases):
            if ln == 'train': continue
            ax = axes[nplot_m*ndw_funs + 4*(li-1) ,pi] 
            acc = n_accs[pi][ln]
            acc = np.r_[n_accs[pi]['train'], acc]

            for di in range(len(acc)):
                acc = np.maximum(acc, 1-acc)
                acc_mean = acc[di].mean(1)
                acc_std = acc[di].std(1)
                ls = '--' if di > 0 else '-'
                mean_line = ax.plot(iters, acc_mean, ls, linewidth=2)[0]
                ax.fill_between(iters, acc_mean+acc_std, acc_mean-acc_std, 
                    linewidth = 0,
                    facecolor = mean_line.get_color(), alpha = 0.3)

            ax.set_ylim([0.5,1.05])
            ax.set_xscale('log')
            ax.set_xticklabels([])
            ax.set_xlim([1,iters[-1]])
            ax.grid(True)
            if pi == 0:
                ax.set_ylabel('accuracy: %s' % (ln))
            else:
                ax.set_yticklabels([])

    # final accuracy for all tests and params
    xind = np.arange(nparams)
    bar_colors = 'bgr'
    for li, ln in enumerate(test_phases):
        if ln == 'train': continue
        ax = axes[nplot_m*ndw_funs + 4*(li-1), -1] 
        acc = np.array([np.r_[n_accs[pi]['train'][:,-1],n_accs[pi][ln][:,-1]] for pi in range(nparams)])
        acc = np.maximum(acc, 1-acc)
        barlist = []
        for di in range(acc.shape[1]):

            #final mean accuracy
            offsets = np.linspace(-0.3,0.3,acc.shape[1])
            width   = 0.6/((acc.shape[1])-1)
            acc_mean = acc[:,di].mean(1)
            acc_std = acc[:,di].std(1)
            
            # hatch stype
            hs = '//' if di > 0 else None
            bar = ax.bar(xind + offsets[di], acc_mean, width, yerr = acc_std, align='center', 
                    color = bar_colors[di], hatch = hs)
            barlist.append(bar[0])
            ax.set_xlim([-0.5,nparams-0.5])
            ax.set_xticks(xind)
            ax.set_xticklabels(map(str, params))
            ax.set_ylabel('acc')
            ax.set_ylim([0.5,1.05])
            ax.grid(True)
        #ax.legend(barlist, test_phases.keys(), bbox_to_anchor=(0.5,0),loc = 'upper center', ncol = len(test_phases))

    # time course of biased accuracy
    for pi, p in enumerate(params):
        for li, ln in enumerate(test_phases):
            if ln == 'train': continue
            ax = axes[nplot_m*ndw_funs + 4*(li-1) + 1,pi] 
            acc = accs[pi][ln]
            acc = np.r_[accs[pi]['train'], acc]

            for di in range(len(acc)):
                acc = np.maximum(acc, 1-acc)
                acc_mean = acc[di].mean(1)
                acc_std = acc[di].std(1)
                ls = '--' if di > 0 else '-'
                mean_line = ax.plot(iters, acc_mean, ls, linewidth=2)[0]
                ax.fill_between(iters, acc_mean+acc_std, acc_mean-acc_std, 
                    linewidth = 0,
                    facecolor = mean_line.get_color(), alpha = 0.3)

            ax.set_ylim([0.5,1.05])
            ax.set_xscale('log')
            ax.set_xticklabels([])
            ax.set_xlim([1,iters[-1]])
            ax.grid(True)
            if pi == 0:
                ax.set_ylabel('accuracy: %s' % (ln))
            else:
                ax.set_yticklabels([])

    # final biased accuracy for all tests and params
    xind = np.arange(nparams)
    bar_colors = 'bgr'
    for li, ln in enumerate(test_phases):
        if ln == 'train': continue
        ax = axes[nplot_m*ndw_funs + 4*(li-1) + 1, -1] 
        acc = np.array([np.r_[accs[pi]['train'][:,-1],accs[pi][ln][:,-1]] for pi in range(nparams)])
        acc = np.maximum(acc, 1-acc)
        barlist = []
        for di in range(acc.shape[1]):

            #final mean accuracy
            offsets = np.linspace(-0.3,0.3,acc.shape[1])
            width   = 0.6/((acc.shape[1])-1)
            acc_mean = acc[:,di].mean(1)
            acc_std = acc[:,di].std(1)
            
            # hatch stype
            hs = '//' if di > 0 else None
            bar = ax.bar(xind + offsets[di], acc_mean, width, yerr = acc_std, align='center', 
                    color = bar_colors[di], hatch = hs)
            barlist.append(bar[0])
            ax.set_xlim([-0.5,nparams-0.5])
            ax.set_xticks(xind)
            ax.set_xticklabels(map(str, params))
            ax.set_ylabel('acc')
            ax.set_ylim([0.5,1.05])
            ax.grid(True)
        #ax.legend(barlist, test_phases.keys(), bbox_to_anchor=(0.5,0),loc = 'upper center', ncol = len(test_phases))

    # time course of unbiased d-prime
    for pi, p in enumerate(params):
        for li, ln in enumerate(test_phases):
            if ln == 'train': continue
            ax = axes[nplot_m*ndw_funs + 4*(li-1)+2 ,pi] 
            dp = n_dps[pi][ln]
            dp = np.r_[n_dps[pi]['train'], dp]

            for di in range(len(dp)):
                dp_mean = dp[di].mean(1)
                dp_std = dp[di].std(1)
                ls = '--' if di > 0 else '-'
                mean_line = ax.plot(iters, dp_mean, ls, linewidth=2)[0]
                ax.fill_between(iters, dp_mean+dp_std, dp_mean-dp_std, 
                    linewidth = 0,
                    facecolor = mean_line.get_color(), alpha = 0.3)

            ax.set_xscale('log')
            if li != len(test_phases):
                ax.set_xticklabels([])
            else:
                ax.set_xtitle('iterations')
            ax.set_xlim([1,iters[-1]])
            ax.grid(True)
            if pi == 0:
                ax.set_ylabel('d-prime: %s' % (ln))
            else:
                ax.set_yticklabels([])
        if li == 0:
            set_data_ylim(ax, np.max(dp_mean + dp_std), min=0.0)

    # final unbiased d-prime for all tests and params
    xind = np.arange(nparams)
    bar_colors = 'bgr'
    for li, ln in enumerate(test_phases):
        if ln == 'train': continue
        ax = axes[nplot_m*ndw_funs + 4*(li-1)+2, -1] 
        dp = np.array([np.r_[n_dps[pi]['train'][:,-1],n_dps[pi][ln][:,-1]] for pi in range(nparams)])
        barlist = []
        for di in range(dp.shape[1]):

            #final mean d-prime
            offsets = np.linspace(-0.3,0.3,dp.shape[1])
            width   = 0.6/((dp.shape[1])-1)
            dp_mean = dp[:,di].mean(1)
            dp_std = dp[:,di].std(1)
            
            # hatch stype
            hs = '//' if di > 0 else None
            bar = ax.bar(xind + offsets[di], dp_mean, width, yerr = dp_std, align='center', 
                    color = bar_colors[di], hatch = hs)
            barlist.append(bar[0])
        ax.set_xlim([-0.5,nparams-0.5])
        ax.set_xticks(xind)
        ax.set_xticklabels(map(str, params))
        ax.set_ylabel('dp')
        set_data_ylim(ax, np.max(dp_mean + dp_std), min=0.0)
        ax.grid(True)
        #ax.legend(barlist, test_phases.keys(), bbox_to_anchor=(0.5,0),loc = 'upper center', ncol = len(test_phases))
        share_ylim(axes[nplot_m*ndw_funs+4*(li-1)+2], min = 0.0)


    # time course of biased d-prime
    for pi, p in enumerate(params):
        for li, ln in enumerate(test_phases):
            if ln == 'train': continue
            ax = axes[nplot_m*ndw_funs + 4*(li-1)+3 ,pi] 
            dp = dps[pi][ln]
            dp = np.r_[dps[pi]['train'], dp]

            for di in range(len(dp)):
                dp_mean = dp[di].mean(1)
                dp_std = dp[di].std(1)
                ls = '--' if di > 0 else '-'
                mean_line = ax.plot(iters, dp_mean, ls, linewidth=2)[0]
                ax.fill_between(iters, dp_mean+dp_std, dp_mean-dp_std, 
                    linewidth = 0,
                    facecolor = mean_line.get_color(), alpha = 0.3)

            ax.set_xscale('log')
            if li != len(test_phases):
                ax.set_xticklabels([])
            else:
                ax.set_xtitle('iterations')
            ax.set_xlim([1,iters[-1]])
            ax.grid(True)
            if pi == 0:
                ax.set_ylabel('d-prime: %s' % (ln))
            else:
                ax.set_yticklabels([])
        if li == 0:
            set_data_ylim(ax, np.max(dp_mean + dp_std), min=0.0)

    # final biased d-prime for all tests and params
    xind = np.arange(nparams)
    bar_colors = 'bgr'
    for li, ln in enumerate(test_phases):
        if ln == 'train': continue
        ax = axes[nplot_m*ndw_funs + 4*(li-1)+3, -1] 
        dp = np.array([np.r_[dps[pi]['train'][:,-1],dps[pi][ln][:,-1]] for pi in range(nparams)])
        barlist = []
        for di in range(dp.shape[1]):

            #final mean d-prime
            offsets = np.linspace(-0.3,0.3,dp.shape[1])
            width   = 0.6/((dp.shape[1])-1)
            dp_mean = dp[:,di].mean(1)
            dp_std = dp[:,di].std(1)
            
            # hatch stype
            hs = '//' if di > 0 else None
            bar = ax.bar(xind + offsets[di], dp_mean, width, yerr = dp_std, align='center', 
                    color = bar_colors[di], hatch = hs)
            barlist.append(bar[0])
        ax.set_xlim([-0.5,nparams-0.5])
        ax.set_xticks(xind)
        ax.set_xticklabels(map(str, params))
        ax.set_ylabel('dp')
        set_data_ylim(ax, np.max(dp_mean + dp_std), min=0.0)
        ax.grid(True)
        #ax.legend(barlist, test_phases.keys(), bbox_to_anchor=(0.5,0),loc = 'upper center', ncol = len(test_phases))
        share_ylim(axes[nplot_m*ndw_funs+4*(li-1)+3], min = 0.0)

    plt.tight_layout()
    plt.show()
    figfile = 'figs/transfer/'+filename+'.pdf'
    fig.savefig(figfile)

    print filename
   

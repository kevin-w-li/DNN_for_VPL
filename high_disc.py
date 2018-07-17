from CaffeVPL import *
import cPickle as pkl
from CaffeVPLUtils import *
from time import strftime, gmtime, time
from shutil import copyfile
from collections import OrderedDict

# In this file, the outermost iteration is over parameters that is not the precision
home_dir = ''
var_param_name = 'layer'
dset = 1
level = 5

#========================
# Training parameters
network = 'AlexNet'
version = 'full'
noise = 5
ctrst = 0.5

test_num = 625          #FIXME
nfold = 25              #FIXME
niter = 5000            #FIXME
num_iters = 30          #FIXME
keep_weights = -1 # keep every three iters
assert num_iters % keep_weights == 0
iters = logiters(1, niter, num_iters) if niter >= 10\
    else np.linspace(0, niter , niter+1, dtype=np.int)
num_iters = len(iters)
batch_size = 20
test_batch_size = 25
base_lr = 1e-4
momentum = 0.9
solver_type = 'SGD'
assert test_num%(nfold*test_batch_size) == 0, 'adjust batch_size or nfold'

if not os.path.isdir(home_dir):
    home_dir = ''

temp_def = str('models/'+network+'/pair_pipeline_'+version+'_sym.prototxt')
model_weights = str('models/'+network+'/weights_1c.caffemodel')
if version == 'conv5':
    layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'vpl']
elif version == 'full':
    layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'vpl']
train_layers = layer_names
diff_layers = train_layers[:]

if var_param_name == 'ctrst':
    params = [0.1,0.2,0.5,1.0]
elif var_param_name == 'noise':
    params = [5,10,20]                  #FIXME
elif var_param_name == 'layer':
    params = np.arange(len(layer_names))
elif var_param_name == 'level':
    params = [1,2,3,4,5]
else:
    raise('var_param_name not valid')

ref_list = [0,1,2,3,4,5,6,7,8,9,10,11]

for ref in ref_list:
    dset = ref
    dataset  = 'morph_'+str(level)+'_' + str(dset)

    # ''' # PLOT
    # =====================
    # set train parameters
    # =====================
    train_noise = noise
    train_ctrst = ctrst

    test_noise = noise
    test_ctrst = ctrst

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
    dp_tis = [None]*len(params)
    acc_tis = [None]*len(params)
    dws = [None]*len(params)
    test_start = time()

    noise_param = dict(type='gaussian', std=train_noise, ctrst=train_ctrst)

    train_param = {
        'batch_size'    :   batch_size,
        'rand_skip'     :   batch_size-1,
        'iter_size'     :   1,
        'momentum'      :   momentum,
        'base_lr'       :   base_lr,
        'weight_decay'  :   0.0,
        'iters'         :   iters,
        'display'       :   0,
        'gamma'         :   1.0,
        'stepsize'     :    200,
        'type'          :   solver_type,
        'noise_param'   :   noise_param
    }

    # test while training

    # test_data = home_dir + 'data/high/' + dataset + '/pair_test2'
    # test_param = {
    #      'data'      :   test_data,
    #      'data_size':    test_num,
    #      'batch_size':   batch_size,
    #      'nfold':        nfold,
    #      'noise_param'   :   noise_param
    # }
    # train_param['test_param'] = test_param

    test_param = {
            'data_size':    test_num,
            'batch_size':   test_batch_size,
            'noise_param':  noise_param,
            'nfold':        nfold,
            'noise_param'   :   noise_param
    }

    params_to_print = '[%.1f,%.1f,%.1f]' % (params[0], params[-1], params[1]-params[0])
    dtstr = strftime('%m-%d-%H-%M-%S', gmtime())
    filename = '_'.join((network[0],
        dataset,var_param_name,params_to_print,
        str(noise),
        str(ctrst),
        str(batch_size), 
        str(base_lr), 
        str(init_niter), 
        str(niter), 
        str(len(layer_names)),
        dtstr
        ))

    train_data = home_dir + 'data/high/' + dataset + '/train'
    test_data1 = home_dir + 'data/high/' + dataset + '/train'
    test_data2 = home_dir + 'data/high/' + dataset + '/test'
    test_phases = OrderedDict([('val', test_data1), ('test', test_data2)])
    train_phase = test_phases.items()[0][0]
    for fn in test_phases.values():
        assert os.path.isdir(fn), fn

    for pi,p in enumerate(params):
        
        if var_param_name == 'ctrst':
            train_ctrst = p
            test_ctrst = p
        elif var_param_name == 'noise':
            train_noise = p
            test_noise = p
        elif var_param_name == 'layer':
            train_layers = layer_names[p:]
        elif var_param_name == 'level':
            dataset = 'morph_'+str(p)+'_' + str(dset)
            train_data = home_dir + 'data/high/' + dataset + '/train'
            test_data1 = home_dir + 'data/high/' + dataset + '/train'
            test_data2 = home_dir + 'data/high/' + dataset + '/test'
            test_phases = OrderedDict([('val', test_data1), ('test', test_data2)])
            train_phase = test_phases.items()[0][0]
            for fn in test_phases.values():
                assert os.path.isdir(fn), fn
        else:
            raise NameError('var_param_name not exist')

        

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

        train_param['layer_names'] = train_layers
        learner.set_training(train_param)
        print "========== PROGRESS: %s in %s, %s in %s" % (p, str(params), ref, str(ref_list))
        results, train_weights, dws[pi] = learner.run_training(niter = niter, load_weights = False, 
            save_weights = True, diff_layers = diff_layers, dw_funs = dw_funs)

        #==================
        #==================
        #   beginning of testing
        #==================
        #==================

        test_start = time()

        learner.set_testing_2(test_param)

        ncalls = 0
        accs[pi]={}
        n_accs[pi]={}
        losses[pi]={}
        dps[pi]={}
        dp_tis[pi]={}
        acc_tis[pi]={}
        
        for ln, test_data in test_phases.iteritems():

            accs[pi][ln] = np.zeros((num_iters,nfold))
            n_accs[pi][ln] = np.zeros((num_iters,nfold))
            losses[pi][ln] = np.zeros((num_iters,nfold))
            dps[pi][ln] = np.zeros((num_iters,nfold))
            for ii, it in enumerate(iters):
                test_param['weights'] = make_weights_file(train_desc, it)
                learner.load_weights(test_param['weights'])

                # reset parameters 
                print "========== PROGRESS: %s in %s, %s in %s" % (p, str(params), ref, str(ref_list))
                print "========== FILENAME: " + filename
                print "========== TESTING MODE: param: {0}, test data: {1}, iter: {2} ==========".format(p, ln, it)
                # set test fature value
                print "   {3:20s}: noise: {0} ctrst: {1} dset: {2}".format(\
                                test_noise, str(test_ctrst), str(dset), 'TRAIN FEATURE VALUES') 
                # get data
                test_data_file = test_data.split('/')[-1]
                test_param['data'] = test_data
                test_param['noise_param']['std'] = test_noise
                test_param['noise_param']['ctrst'] = test_ctrst
                
                test_param['test_desc'] = test_data_file
                accs[pi][ln][ii], losses[pi][ln][ii], \
                        dps[pi][ln][ii], n_accs[pi][ln][ii], _= learner.run_testing_2(test_param)
                print "   {4:20s}: acc: {0:.3f}, loss: {1:.3f}, dp: {2:.3f}, n_acc: {3:.3f}\n".format(
                        accs[pi][ln][ii].mean(), losses[pi][ln][ii].mean(), dps[pi][ln][ii].mean(), n_accs[pi][ln][ii].mean(), 'RESULT')


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
        'n_accs': n_accs, 'acc_tis': acc_tis,'dp_tis': dp_tis, \
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
        f = open('results/transfer/'+filename+'.results', 'r')
    except:
        f = open('results/transfer/03-28-01-50-32', 'r')
    rf = pkl.load(f)
    f.close()
    n_accs = rf['n_accs']
    accs = rf['accs']
    losses = rf['losses']
    filename= rf['filename']
    dps = rf['dps']
    acc_tis = rf['acc_tis']
    dp_tis = rf['dp_tis']
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
    fig, axes = plt.subplots(2+nplot_m*ndw_funs,nparams+1,figsize=(5*(nparams+1),4*(2+nplot_m*ndw_funs)))


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
        ax = axes[nplot_m*ndw_funs,pi] 
        for li, ln in enumerate(test_phases):
            acc = accs[pi][ln].copy()
            acc = np.maximum(acc, 1-acc)
            acc_mean = acc.mean(1)
            acc_std = acc.std(1)
            ls = '--' if li > 0 else '-'
            mean_line = ax.plot(iters, acc_mean, ls, linewidth=2)[0]
            ax.fill_between(iters, acc_mean+acc_std, acc_mean-acc_std, 
                linewidth = 0,
                facecolor = mean_line.get_color(), alpha = 0.3)
        # ax.legend(test_phases.keys(), loc = 'upper left')
        ax.set_ylim([0.5,1.05])
        ax.set_xscale('log')
        ax.set_xticklabels([])
        ax.set_xlim([1,iters[-1]])
        ax.grid(True)
        if pi == 0:
            ax.set_ylabel('accuracy')
        else:
            ax.set_yticklabels([])


    # final accuracy for all tests and params
    ax = axes[nplot_m*ndw_funs,-1]
    barlist = []
    offsets = np.linspace(-0.2,0.2,len(test_phases))
    width   = 0.4/(len(test_phases)-1)
    xind    = np.arange(nparams)
    bar_colors = 'bgr'
    for li, ln in enumerate(test_phases):

        #final mean accuracy
        acc = np.array([accs[pi][ln][-1].copy() for pi in range(nparams)])
        acc = np.maximum(acc, 1-acc)
        acc_mean = acc.mean(1)
        acc_std = acc.std(1)
        
        # hatch stype
        hs = '//' if li > 0 else None
        bar = ax.bar(xind + offsets[li], acc_mean, width, yerr = acc_std, align='center', 
                color = bar_colors[li], hatch = hs)
        barlist.append(bar[0])
        ax.set_xlim([-0.5,nparams-0.5])
        ax.set_xticks(xind)
        ax.set_xticklabels(map(str, params))
        ax.set_ylabel('acc')
        ax.set_ylim([0.5,1.05])
        ax.grid(True)
    ax.legend(barlist, test_phases.keys(), bbox_to_anchor=(0.5,0),loc = 'upper center', ncol = len(test_phases))

    for pi, p in enumerate(params):

        ax = axes[nplot_m*ndw_funs+1,pi]
        for li, ln in enumerate(test_phases):
            dp = dps[pi][ln]
            dp_mean = dp.mean(1)
            dp_std = dp.std(1)
            ls = '--' if li > 0 else '-'
            mean_line = ax.plot(iters, dp_mean, ls, linewidth=2)[0]
            ax.fill_between(iters, dp_mean+dp_std, dp_mean-dp_std, 
                linewidth = 0,
                facecolor = mean_line.get_color(), alpha = 0.3)
        
        ax.set_xscale('log')
        ax.set_xlim([1,iters[-1]])
        ax.set_xlabel('iteration')
        ax.grid(True)
        if pi == 0:
            ax.set_ylabel('d-prime')
        else:
            ax.set_yticklabels([])
        # assume that the first test case is trained
        if li == 0.0:
            set_data_ylim(ax, np.max(dp_mean + dp_std), min=0.0)

    # final dprime for all tests and params
    ax = axes[nplot_m*ndw_funs+1,-1]
    barlist = []
    for li, ln in enumerate(test_phases):

        #final mean dprime
        dp_mean = np.array([dps[pi][ln][-1].mean() for pi in range(nparams)])
        dp_std  = np.array([dps[pi][ln][-1].std()  for pi in range(nparams)])
        
        hs = '//' if li > 0 else None
        bar = ax.bar(xind + offsets[li], dp_mean, width, yerr = dp_std, align='center',
                color = bar_colors[li], hatch = hs)
        barlist.append(bar[0])
        ax.set_xticks(xind)
        ax.set_xlim([-0.5,nparams-0.5])
        ax.set_xticks(range(nparams))
        ax.set_xticklabels(map(str, params))
        set_data_ylim(ax, np.max(dp_mean + dp_std), min=0.0)
        ax.set_ylabel('d-prime')
        ax.grid(True)
    share_ylim(axes[nplot_m*ndw_funs+1], min = 0.0)

    plt.tight_layout()
    plt.show()
    figfile = 'figs/transfer/'+filename+'.pdf'
    fig.savefig(figfile)

    print filename

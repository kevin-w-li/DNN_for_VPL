# DNN_for_VPL

This repo has the core code for the following paper
> Deep Neural Networks for Modeling Visual Perceptual Learning  
> Li K. Wenliang, Aaron R. Seitz  
> Journal of Neuroscience 4 July 2018, 38 (27) 6028-6044; DOI: 10.1523/JNEUROSCI.1620-17.2018

It contains the core files needed to run the simulation described in the paper and the Jupyter Notebooks that plots figures (all but Figure 1). Let me know if anything is missing.

Unfortunately Caffe was the best DNN toolbox during this project, but it was still very difficult to use. Hence, the pipeline of reproducing the results may seem convoluted.

As a result, it is strongly encouraged to use better DNN toolbox (TensorFlow or PyTorch) to continue this line of research!

The code was written in Python 2.7 with the following dependencies:

1. Caffe 1.0 (https://github.com/BVLC/caffe/tree/1.0) built on Ubuntu 14.04 with GPU support.
1. CUDA 7.5 (https://developer.nvidia.com/cuda-75-downloads-archive)
1. cuDNN 5.1 for CUDA 7.5 (https://developer.nvidia.com/rdp/cudnn-archive)
1. facemorepher (https://github.com/alyssaq/face_morpher)
1. lmdb 0.86 
1. numpy 1.11.0
1. matplotlib 1.5.1

To run the simulations:

##  Experiment 1: orientation discrimination of Gabor patches

1. Generate data (Gabors) using `generate_pair_angle.py`.
1. Train the network by running `pair_transfer.py`. Need to set the following parameters:
  * `noise`
  * `angle1`
  * `wave1`
  * `ctrst1` (contrast)

  The script will produce a figure in the figs/ directory and a result file in the results/ directory. The results are used by the `plot_summary.ipynb` notebook to produce behaviour and layer-wise results.
1. Run `compute_tuning.py` to get the tuning properties of the units (neurons) in the network. The results will be saved in results/tuning/. These are used in by the `tuning_main.ipynb` and `tuning_supp.ipynb` to generate figures of tuning attributes.

## Experiment 2: Face gender discrimination
1. You would have to obtain the PhotoFace dataset from the authors of this paper: https://ieeexplore.ieee.org/document/5981840/
1. Choose male and female images from the dataset and put them into data/high/male and data/high/female
1. Inside data/high, run `data/high/morph.py` which will generate 12 sets of train-test data in data/high.
1. Run `high_desc.py` to train the network. Need to set the `dset` parameter to loop from 1 to 12
1. The results will be used to generate figures in the `plot_summary.ipynb` notebook.


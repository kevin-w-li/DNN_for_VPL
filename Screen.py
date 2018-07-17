import numpy as np
from scipy.stats import truncnorm
import copy


class Stimulus:

    def __init__(self, size, center, angle, ctrst, wave, pxl_range):
        # make the side length odd
        if size % 2 == 0:
            size = size + 1;

        self.size = size
        self.center = center
        self.angle = angle
        self.ctrst = ctrst
        self.wave = wave
        self.pxl_range = pxl_range

        self.stim_params = None
        self.image = None
        self.bgd = None


        return self

    def one_side_jitter(self, center_o = np.array([0.0,0.0]), angle_o = 0.0, ctrst_o = 0.0, wave_o = 0.0,
            center_j = np.array([0.0,0.0]), angle_j = 0.0, ctrst_j = 0.0, wave_j = 0.0, jitter_type='Gaussian'):
    
        if jitter_type == 'Gaussian':
            rands = abs(np.random.standard_normal(5))
            if angle_j != 0:
                rands[2] = np.sign(angle_j) * (truncnorm.rvs(a = 0, b=45.0/abs(angle_j), loc = 0, scale = abs(angle_j)))
            else:
                rands[2] = 0
        elif jitter_type == 'uniform':
            rands = np.random.rand(5)
            rands[2] *= angle_j
        elif jitter_type == 'fix':
            rands = np.ones(5)
            rands[2] *= angle_j
        center_j = rands[0:2] * center_j + center_o;
        angle_j = rands[2] + angle_o;
        ctrst_j = rands[3]*ctrst_j + ctrst_o;
        wave_j  = rands[4]*wave_j  + wave_o;

        if ctrst_j > 1:
            ctrst_j = 1.0
        elif ctrst_j < -1:
            ctrst_j = -1

        wave_j = np.maximum(-1.0, wave_j)

        if any( center_j != 0.0) and angle_j != 0 and ctrst_j != 0:
            self.change(center_j, angle_j, ctrst_j)
        else:
            if any( center_j != 0.0):
                self.translate(center_j)
            if angle_j != 0.0:
                self.rotate(angle_j)
            if ctrst_j != 0.0:
                self.contrast(ctrst_j)
            if wave_j  != 0.0:
                self.stretch(wave_j)
        return self

    def jitter(self, center_j = np.array([0.0,0.0]), angle_j = 0.0, ctrst_j = 0.0, wave_j = 0.0, jitter_type='Gaussian'):
    
        if jitter_type == 'Gaussian':
            rands = np.random.standard_normal(5)
            if angle_j != 0:
                rands[2] = truncnorm.rvs(a=-45.0/angle_j, b=45.0/angle_j, loc = 0, scale = angle_j)
            else:
                rands[2] = 0
        elif jitter_type == 'uniform':
            rands = np.random.rand(4)*2-1
            rands[2] *= angle_j
        elif jitter_type == 'fix':
            rands = 2*(np.random.rand(4)>0.5)-1
            rands[2] *= angle_j
        center_j = rands[0:2] * center_j;
        angle_j = rands[2];
        ctrst_j = rands[3]*ctrst_j;
        wave_j  = rands[4]*wave_j;

        if ctrst_j > 1:
            ctrst_j = 1.0
        elif ctrst_j < -1:
            ctrst_j = -1

        wave_j = np.maximum(0.0, wave_j)

        if any( center_j != 0.0) and angle_j != 0 and ctrst_j != 0 and wave_j != 0:
            self.change(center_j, angle_j, ctrst_j)
        else:
            if any( center_j != 0.0):
                self.translate(center_j)
            if angle_j != 0.0:
                self.rotate(angle_j)
            if ctrst_j != 0.0:
                self.contrast(ctrst_j)
            if wave_j  != 0.0:
                self.stretch(wave_j)
        return self

    def initialise(self):
        raise NotImplementedError('Need to implement initialise!')

    def change(self, **kwargs):
        raise NotImplementedError('Need to implement rotate!')

    def rotate(self, angle_j):
        raise NotImplementedError('Need to implement rotate!')

    def contrast(self,ctrst_j):
        raise NotImplementedError('Need to implement contrast!')

    def translate(self,center_j):
        raise NotImplementedError('Need to implement translate!')
    
    def rescale(self):
        raise NotImplementedError('Need to implement rescale!')

    def stretch(self):
        raise NotImplementedError('Need to implement wave!')

    def toRGB(self, ch_idx = 0):

        if self.image.ndim == 3:
            return self.image
        
        repdim = [1,1,1];
        repdim[ch_idx] = 3;
        self.image = np.tile(self.image, repdim)
        return self.image

    def constrain_pxl(self, image=None):

        if image is None:
            image = self.image

        pxl_range = self.pxl_range
        out_min = pxl_range[0]
        out_max = pxl_range[1]
        image[image>out_max] = out_max
        image[image<out_min] = out_min
        return image

    def add_noise(self, s):
        self.image += np.random.standard_normal(self.image.shape)*s
        self.constrain_pxl()
        return self.image

    
    def batch_jitter(self, n, center_j = np.array([0.0,0.0]), angle_j = 0.0, ctrst_j = 0.0, wave_j = 0.0, jitter_type = 'normal'):

        self_copy = copy.deepcopy(self)

        def get_image():
            self = copy.deepcopy(self_copy)
            self.jitter( center_j = center_j, angle_j = angle_j, ctrst_j = ctrst_j, wave_j = wave_j, jitter_type=jitter_type)
            return self

        batch = map(lambda x:get_image(), range(n))
        batch_image = map(lambda b: b.image[np.new_axis,:], batch)
        batch_center = map(lambda b: b.center, batch) 
        batch_angle = map(lambda b: b.angle, batch) 
        batch_ctrst = map(lambda b: b.ctrst, batch) 
        batch_image = np.array(batch_image)
        return batch_image, batch_center, batch_angle, batch_ctrst


    def batch_change(self, n, center_j = np.array([[0.0,0.0]]), angle_j = np.array([0.0]), ctrst_j = np.array([0.0]), wave_j = np.array([0.0])):
        ncenter = center_j.shape[0]
        nangle = angle_j.shape[0]
        nctrst = ctrst_j.shape[0]
        nwave = wave_j.shape[0]
        assert (ncenter == 1 or ncenter == n), 'center_j does not match'
        assert (nangle == 1 or nangle == n), 'angle_j does not match'
        assert (nctrst == 1 or nctrst == n), 'ctrst_j does not match'
        assert (nwave == 1 or nwave == n), 'ctrst_j does not match'
        center_j = np.tile(center_j,[n,1]) if ncenter == 1 else center_j
        angle_j = np.tile(angle_j,[n]) if nangle == 1 else angle_j
        ctrst_j = np.tile(ctrst_j,[n]) if nctrst == 1 else ctrst_j
        wave_j = np.tile(wave_j,[n]) if nwave == 1 else wave_j

        self_copy = copy.deepcopy(self)

        def get_image(c_j, a_j, t_j, w_j):
            self = copy.deepcopy(self_copy)
            self.change( center_j = c_j, angle_j = a_j, ctrst_j = t_j, wave_j = w_j)
            return self.image[np.newaxis,:]
        batch = map(lambda i: get_image( center_j[i], angle_j[i], ctrst_j[i], wave_j[i]), range(n)) 
        batch = np.array(batch)
        return batch

    def gradient(self, feature, d = 0.0, eps = 1e-6):
        
        if feature[-2:] != '_j':
            feature += '_j'

        self_copy = copy.deepcopy(self)
        self_copy.change(**{feature: d})
        image_1 = self_copy.change(**{feature: eps/2.0})
        image_2 = self_copy.change(**{feature: -eps})

        grad = (image_1 - image_2)/eps

        return grad, image_1, image_2

class Gabor(Stimulus):

    def __init__(self, size = 227, sigma = 10, center = np.array([0.0,0.0]), angle=0.0, ctrst = 1.0, wave=10.0, pxl_range = (0.0,255.0)):

        Stimulus.__init__(self, size, np.array([0.0,0.0]), 0.0, ctrst, wave, pxl_range)
        
        center = np.array(center)
        mid = np.array([(self.size-1)/2.0, (self.size-1)/2.0])
        x, y = np.meshgrid(\
                np.array(range(size))-mid[0],\
                np.array(range(size-1,-1,-1))-mid[1], \
                indexing = 'xy')
        
        
        self.stim_params = {'x':x, 'y':y, 'bgd' : 0.0, 'sigma': sigma}
        self.image = self.change(center_j = center, angle_j = angle) 

    def change(self, center_j = np.array([0.0,0.0]), angle_j= 0.0, ctrst_j = 0.0, wave_j = 0.0):
        self.wave *= 1.0+wave_j

        sigma = self.stim_params['sigma']

        freq = 2*3.1416/self.wave;

        center_j = np.array(center_j)
        x = self.stim_params['x']
        y = self.stim_params['y']

        if angle_j!= 0.0:
            #rotation
            
            RotRad = angle_j * np.pi / 180
            self.angle = (self.angle + angle_j) % 180.0
            rot_mat= np.array([[np.cos(RotRad), -np.sin(RotRad)],
                              [np.sin(RotRad), np.cos(RotRad)]])
            gstack = np.einsum('ji, mni -> jmn', rot_mat, np.dstack([x, y]))
            x = gstack[0]
            y = gstack[1]
        if any(center_j != 0.0 ):
            # tranlation
            angle_rad = self.angle*np.pi/180.0
            rot_mat= np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                              [np.sin(angle_rad), np.cos(angle_rad)]])
            trans_ref = np.dot(rot_mat, center_j)
            x = x - trans_ref[0]
            y = y - trans_ref[1]
            self.center = self.center + center_j
         
        # gabor prototype
        image = 1/(2*np.pi*sigma*sigma) *\
                np.exp(-0.5 * (x**2.0 + y**2.0)/sigma/sigma) * \
                np.cos(freq * x)
        self.image, self.bgd = self.rescale(image)
        
        self.stim_params['x'] = x
        self.stim_params['y'] = y
        #contrast 
        self.ctrst = self.ctrst * (1+ ctrst_j)
        if self.ctrst > 1: self.ctrst = 1.0
        if self.ctrst < 0: self.ctrst = 0.0
        self.image = self.bgd + (self.image - self.bgd) * self.ctrst        
        self.image = self.constrain_pxl(self.image)
        return self.image


    def rotate(self, angle_j):    
        return self.change(angle_j = angle_j ) 

    def contrast(self, ctrst_j):
        return self.change(ctrst_j = ctrst_j)

    def translate(self, center_j):
        return self.change(center_j= center_j)

    def stretch(self, wave_j):
        return self.change(wave_j = wave_j)

    def rescale(self,image):
        in_bgd = self.stim_params['bgd']
        pxl_range = self.pxl_range
        in_min = image.min()
        in_max = image.max()
        out_min = pxl_range[0]
        out_max = pxl_range[1]
        
        r = (out_max - out_min) / (in_max - in_min)
        bgd = (in_bgd - in_min) * r
        image = image * r + bgd
        image = self.constrain_pxl(image)
        return image, bgd


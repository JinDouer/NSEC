import numpy as np
from PIL import Image

from utils import *

def gen_samples(generator, bbox, n, overlap_range=None, scale_range=None):

    if overlap_range is None and scale_range is None:
        return generator(bbox, n)

    else:
        samples = None
        remain = n
        factor = 2
        #while remain > 0:
        while remain > 0 and factor < 16:
            samples_ = generator(bbox, remain*factor)

            idx = np.ones(len(samples_), dtype=bool)
            if overlap_range is not None:
                r = overlap_ratio(samples_, bbox)
                idx *= (r >= overlap_range[0]) * (r <= overlap_range[1])
            if scale_range is not None:
                s = np.prod(samples_[:,2:], axis=1) / np.prod(bbox[2:])
                idx *= (s >= scale_range[0]) * (s <= scale_range[1])

            samples_ = samples_[idx,:]
            samples_ = samples_[:min(remain, len(samples_))]
            if samples is None:
                samples = samples_
            else:
                samples = np.concatenate([samples, samples_])
            remain = n - len(samples)
            factor = factor*2

        return samples


class SampleGenerator():
    def __init__(self, type, img_size, trans_f=1, scale_f=1, aspect_f=None, valid=False):
        self.type = type
        self.img_size = np.array(img_size) # (w, h)
        self.trans_f = trans_f
        self.scale_f = scale_f
        self.aspect_f = aspect_f
        self.valid = valid

    def __call__(self, bb, n):
        #
        # bb: target bbox (min_x,min_y,w,h)
        bb = np.array(bb, dtype='float32')

        # (center_x, center_y, w, h)
        sample = np.array([bb[0]+bb[2]/2, bb[1]+bb[3]/2, bb[2], bb[3]], dtype='float32')
        samples = np.tile(sample[None,:],(n,1))

        # vary aspect ratio
        if self.aspect_f is not None:
            ratio = np.random.rand(n,1)*2-1
            samples[:,2:] *= self.aspect_f ** np.concatenate([ratio, -ratio],axis=1)

        # sample generation
        if self.type=='gaussian':
            samples[:,:2] += self.trans_f * np.mean(bb[2:]) * np.clip(0.5*np.random.randn(n,2),-1,1)
            samples[:,2:] *= self.scale_f ** np.clip(0.5*np.random.randn(n,1),-1,1)

        elif self.type=='uniform':
            samples[:,:2] += self.trans_f * np.mean(bb[2:]) * (np.random.rand(n,2)*2-1)
            samples[:,2:] *= self.scale_f ** (np.random.rand(n,1)*2-1)
        elif self.type=='part':#exact part no bg
            '''
            samples[:,0] += self.trans_f * bb[2] * (np.random.rand(n)*2-1)
            samples[:,1] += self.trans_f * bb[3] * (np.random.rand(n)*2-1)
            samples[:,2:] *= self.scale_f ** (np.random.rand(n,1)*2-1)
            samples[:,2] = np.clip(samples[:,2],0,bb[2])
            samples[:,3] = np.clip(samples[:,3],0,bb[3])
            '''

            x = bb[0]+bb[2]*np.random.rand(n)
            y = bb[1]+bb[3]*np.random.rand(n)
            w = (bb[2]+bb[0]-x)*np.random.rand(n)
            h = (bb[3]+bb[1]-y)*np.random.rand(n)
            samples[:,0] = x+w/2
            samples[:,1] = y+h/2
            samples[:,2] = w
            samples[:,3] = h
        elif self.type == 'edge':
            m = n / 4   ## smaples per edge
            samples[0:m,0] = bb[0]+bb[2]*np.random.rand(m)
            samples[0:m,1] = bb[1]
            #samples[0:m,2] = 2*np.minimum(samples[0:m,0]-bb[0],bb[2]+bb[0]-samples[0:m,0])*np.random.rand(m)
            samples[0:m,2] = 2*bb[2]*np.random.rand(m)
            samples[0:m,3] = 2*bb[3]*np.random.rand(m)

            samples[m:2*m,0] = bb[0]+bb[2]
            samples[m:2*m,1] = bb[1]+bb[3]*np.random.rand(m)
            samples[m:2*m,2] = 2*bb[2]*np.random.rand(m)
            samples[m:2*m,3] = 2*bb[3]*np.random.rand(m)
            #samples[m:2*m,3] = 2*np.minimum(samples[m:2*m,1]-bb[1],bb[3]+bb[1]-samples[m:2*m,1])*np.random.rand(m)

            samples[2*m:3*m,0] = bb[0]+bb[2]*np.random.rand(m)
            samples[2*m:3*m,1] = bb[1]+bb[3]
            #samples[2*m:3*m,2] = 2*np.minimum(samples[2*m:3*m,0]-bb[0],bb[2]+bb[0]-samples[2*m:3*m,0])*np.random.rand(m)
            samples[2*m:3*m,3] = 2*bb[2]*np.random.rand(m)
            samples[2*m:3*m,3] = 2*bb[3]*np.random.rand(m)

            samples[3*m:4*m,0] = bb[0]
            samples[3*m:4*m,1] = bb[1]+bb[3]*np.random.rand(m)
            samples[3*m:4*m,2] = 2*bb[2]*np.random.rand(m)
            samples[3*m:4*m,2] = 2*bb[3]*np.random.rand(m)
            #samples[3*m:4*m,3] = 2*np.minimum(samples[3*m:4*m,1]-bb[1],bb[3]+bb[1]-samples[3*m:4*m,1])*np.random.rand(m)
        elif self.type=='hole':##over include
            samples[:,:2] += self.trans_f * np.mean(bb[2:]) * (np.random.rand(n,2)*2-1)
            samples[:,2:] *= self.scale_f ** np.random.rand(n,1)
        elif self.type=='round':
            m = int(2*np.sqrt(n))
            xy = np.dstack(np.meshgrid(np.linspace(-1.6,1.6,m),np.linspace(-1.6,1.6,m))).reshape(-1,2)
            ab = []
            for i,p in enumerate(xy):
                if p[0] > -1.0 and p[0] < 1.0 and p[1] > -1.0 and p[1] < 1.0:
                    pass
                else:
                    ab.append(xy[i])
            ab = np.array(ab)
            ab = np.random.permutation(ab)[:n]
            #ipdb.set_trace()
            samples[:,:2] += np.mean(bb[2:]) * ab
            samples[:,2:] *= self.scale_f ** (np.random.rand(n,1)*2-1)
        elif self.type=='whole': ##by limit the overlap low, get pure bg
            m = int(2*np.sqrt(n))
            xy = np.dstack(np.meshgrid(np.linspace(0,1,m),np.linspace(0,1,m))).reshape(-1,2)
            xy = np.random.permutation(xy)[:n]
            samples[:,:2] = bb[2:]/2 + xy * (self.img_size-bb[2:]/2-1)
            #samples[:,:2] = bb[2:]/2 + np.random.rand(n,2) * (self.img_size-bb[2:]/2-1)
            samples[:,2:] *= self.scale_f ** (np.random.rand(n,1)*2-1)
        elif self.type == 'test':
            samples[:,:2] += self.trans_f * np.mean(bb[2:]) * np.clip(0.5*np.random.randn(n,2),-1,1)
            samples[:,2:] *= self.scale_f ** np.clip(0.5*np.random.randn(n,1),-1,1)

            samples2 = samples
            samples3 = samples
            samples2[:,2] *= 0.5
            samples2[:,3] *= 2
            samples3[:,2] *= 2
            samples3[:,3] *= 0.5

            samples = np.concatenate((samples,samples2,samples3),axis=0)

        # adjust bbox range
        samples[:,2:] = np.clip(samples[:,2:], 10, self.img_size-10)
        if self.valid:
            samples[:,:2] = np.clip(samples[:,:2], samples[:,2:]/2, self.img_size-samples[:,2:]/2-1)
        else:
            samples[:,:2] = np.clip(samples[:,:2], 0, self.img_size)

        # (min_x, min_y, w, h)
        samples[:,:2] -= samples[:,2:]/2

        return samples

    def set_trans_f(self, trans_f):
        self.trans_f = trans_f

    def get_trans_f(self):
        return self.trans_f
    def set_scale_f(self, scale_f):
        self.scale_f = scale_f

    def get_scale_f(self):
        return self.scale_f

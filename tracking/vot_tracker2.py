import vot
import numpy as np
import os
import sys
import time
import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt
import collections
import torch
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
from scipy.spatial import distance
sys.path.insert(0,'/home/prisimage/tracker/py-MDNetST/modules')
sys.path.insert(0,'/home/prisimage/tracker/py-MDNetST/tracking')
from sample_generator import *
from data_prov import *
from model import *
from bbreg import *
from options import *
from gen_config import *
np.random.seed(123)
torch.manual_seed(456)
torch.cuda.manual_seed(789)

def dist_change(samples,target_bbox):
    dist = samples[:,:2] - target_bbox[:2]
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            dist[i][j] = dist[i][j]*dist[i][j]
    dist = np.sum(dist,axis=1)
    dist = np.sqrt(dist)
    N = 4.0*(target_bbox[2] + target_bbox[3])
    window = 0.5+0.5*np.cos(2*np.pi*dist/N)
    window = np.tile(window,(2,1)).transpose()
    return window
def transform_box(target_bbox,trans=1):
    x,y,w,h = target_bbox
    cx = x + w/2
    cy = y + h/2
    if trans == 1:
        return np.array([cx-h/2,cy-w/2,h,w])
    elif trans == 2:
        w = np.sqrt(w*h)
        return np.array([cx-w/2,cy-w/2,w,w])
    elif trans == 3:
        w = 1.5*np.sqrt(w*h)
        return np.array([cx-w/2,cy-w/2,w,w])
def stackList(featList):
    nframes = len(featList)
    for start in range(nframes):
        if start == 0:
            pos_data = featList[start].clone()
        else:
            pos_data = torch.cat((pos_data,featList[start].clone()),0)
    return pos_data
def forward_samples(model, image, samples, out_layer='conv3'):
    model.eval()
    extractor = RegionExtractor(image, samples, opts['img_size'], opts['padding'], opts['batch_test'])
    for i, regions in enumerate(extractor):
        regions = Variable(regions)
        if opts['use_gpu']:
            regions = regions.cuda()
        feat = model(regions, out_layer=out_layer)
        if i==0:
            feats = feat.data.clone()
        else:
            feats = torch.cat((feats,feat.data.clone()),0)
    return feats


def set_optimizer(model, lr_base, lr_mult=opts['lr_mult'], momentum=opts['momentum'], w_decay=opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.iteritems():
        lr = lr_base
        for l, m in lr_mult.iteritems():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr':lr})
    optimizer = optim.SGD(param_list, lr = lr, momentum=momentum, weight_decay=w_decay)
    return optimizer


def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
    model.train()

    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while(len(pos_idx) < batch_pos*maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while(len(neg_idx) < batch_neg_cand*maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for iter in range(maxiter):

        # select pos idx
        pos_next = pos_pointer+batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer+batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = Variable(pos_feats.index_select(0, pos_cur_idx))
        batch_neg_feats = Variable(neg_feats.index_select(0, neg_cur_idx))

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0,batch_neg_cand,batch_test):
                end = min(start+batch_test,batch_neg_cand)
                score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start==0:
                    neg_cand_score = score.data[:,1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.data[:,1].clone()),0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats.index_select(0, Variable(top_idx))
            model.train()

        # forward
        pos_score = model(batch_pos_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)

        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
        optimizer.step()

        #print "Iter %d, Loss %.4f" % (iter, loss.data[0])


class CONVTracker(object):
    def __init__(self,image,region):
        #region [x1,y1,x2,y2...x4,y4]

        x1,y1,x2,y2,x3,y3,x4,y4 = region.points[0].x,region.points[0].y,region.points[1].x,region.points[1].y,region.points[2].x,region.points[2].y,region.points[3].x,region.points[3].y
        cx = (x1+x2+x3+x4) / 4
        cy = (y1+y2+y3+y4) / 4



        xmin = min(x1,x2,x3,x4)
        xmax = max(x1,x2,x3,x4)
        ymin = min(y1,y2,y3,y4)
        ymax = max(y1,y2,y3,y4)
        A1 = distance.euclidean((x1,y1),(x2,y2))*distance.euclidean((x2,y2),(x3,y3))
        A2 = (xmax - xmin)*(ymax - ymin)
        s = np.sqrt(A1 / A2)
        if s < 0.95:
            p = 0.5
        else:
            p = 0
        w = (s*p+1.*(1-p)) * (xmax - xmin) + 1
        h = (s*p+1.*(1-p)) * (ymax - ymin) + 1
        init_bbox = [cx - w / 2., cy - h / 2., w, h]
        
        '''
        init_bbox = [region.x,region.y,region.width,region.height]
        '''
        #init_bbox = region
        self.target_bbox =np.array(init_bbox)
        self.result = []
        self.result_bb = []
        self.result.append(self.target_bbox)
        self.result_bb.append(self.target_bbox)
        self.model = MDNet(opts['model_path'])
        if opts['use_gpu']:
            self.model = self.model.cuda()
        self.model.set_learnable_params(opts['ft_layers'])

        # Init criterion and optimizer
        self.criterion = BinaryLoss()
        self.init_optimizer = set_optimizer(self.model, opts['lr_init'])
        self.update_optimizer = set_optimizer(self.model, opts['lr_update'])
        # Train bbox regressor
        bbreg_examples = gen_samples(SampleGenerator('uniform', image.size, 0.3, 1.5, 1.1),
                                     self.target_bbox, opts['n_bbreg'], opts['overlap_bbreg'], opts['scale_bbreg'])
        bbreg_feats = forward_samples(self.model, image, bbreg_examples)
        self.bbreg = BBRegressor(image.size)
        self.bbreg.train(bbreg_feats, bbreg_examples, self.target_bbox)


        ##mix
        pos_examples = gen_samples(SampleGenerator('gaussian', image.size, 0.1, 1.2),
                                   self.target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])
        if len(pos_examples) == 0:
            pos_examples = np.tile(self.target_bbox[None,:],(opts['n_pos_init'],1))
        neg_examples = np.concatenate([
                        gen_samples(SampleGenerator('uniform', image.size, 1, 2, 1.1),
                                    self.target_bbox, opts['n_neg_init']//2, opts['overlap_neg_init']),
                        gen_samples(SampleGenerator('whole', image.size, 0, 1.2, 1.1),
                                    self.target_bbox, opts['n_neg_init']//2, opts['overlap_neg_init'])])
        neg_examples = np.random.permutation(neg_examples)

        # Extract pos/neg features
        pos_feats = forward_samples(self.model, image, pos_examples)
        neg_feats = forward_samples(self.model, image, neg_examples)
        self.feat_dim = pos_feats.size(-1)
        train(self.model, self.criterion, self.init_optimizer, pos_feats, neg_feats, opts['maxiter_init'])
        # Init sample generators
        opts['trans_f'] = 1.0
        self.sample_generator = SampleGenerator('uniform', image.size, opts['trans_f'], opts['scale_f'], 1.1,valid=True)



        self.pos_generator = SampleGenerator('gaussian', image.size, 0.1, 1.2)
        self.neg_generator = SampleGenerator('uniform', image.size, 1.5, 1.2)
        self.ds_generator = SampleGenerator('round', image.size, 1.5, 1.2)
        # Init pos/neg features for update


        self.pos_feats_all = [pos_feats[:opts['n_pos_update']]]
        self.neg_feats_all = [neg_feats[:opts['n_neg_update']]]
    def track(self,image,i):

        # Estimate target bbox
        opts['n_samples'] = 512


        samples0 = gen_samples(self.sample_generator, self.target_bbox, opts['n_samples'])
        samples1 = gen_samples(self.sample_generator, transform_box(self.target_bbox,1), opts['n_samples'])
        samples2 = gen_samples(self.sample_generator, transform_box(self.target_bbox,2), opts['n_samples'])
        samples = np.concatenate((samples0,samples1,samples2),axis=0)

        #samples0 = gen_samples(self.sample_generator, self.target_bbox, opts['n_samples'])
        #samples1 = gen_samples(self.sample_generator, transform_box(self.target_bbox,1), opts['n_samples'])
        #samples2 = gen_samples(self.sample_generator, transform_box(self.target_bbox,2), opts['n_samples'])
        #samples = np.concatenate((samples0,samples1,samples2),axis=0)

        sample_scores = forward_samples(self.model, image, samples, out_layer='fc6')
        '''
        window_penalty = 0.2
        window = dist_change(samples,self.target_bbox)
        window = torch.from_numpy(window).float()
        sample_scores = sample_scores.cpu()
        sample_scores = (1-window_penalty) * sample_scores + window_penalty * window * sample_scores
        '''
        top_scores, top_idx = sample_scores[:,1].topk(5)
        top_idx = top_idx.cpu().numpy()
        target_score = top_scores.mean()
        self.target_bbox = samples[top_idx].mean(axis=0)

        success = target_score > opts['success_thr']

        # Expand search area at failure

        if success:
            self.sample_generator.set_trans_f(opts['trans_f'])
        else:
            self.sample_generator.set_trans_f(opts['trans_f_expand'])

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            bbreg_feats = forward_samples(self.model, image, bbreg_samples)
            bbreg_samples = self.bbreg.predict(bbreg_feats, bbreg_samples)
            self.bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = self.target_bbox

        # Copy previous result at failure

        if not success:
            self.target_bbox = self.result[-1]
            self.bbreg_bbox = self.result_bb[-1]

        # Save result
        self.result.append(self.target_bbox)
        self.result_bb.append(self.bbreg_bbox)

        # Data collect
        if success:

            # Draw pos/neg samples
            pos_examples = gen_samples(self.pos_generator, self.target_bbox,
                                       opts['n_pos_update'],
                                       opts['overlap_pos_update'])
            if len(pos_examples) == 0:
                pos_examples = np.tile(self.target_bbox[None,:],(opts['n_pos_init'],1))
            neg_examples = gen_samples(self.neg_generator, self.target_bbox,
                                       opts['n_neg_update'],
                                       opts['overlap_neg_update'])

            # Extract pos/neg features
            pos_feats = forward_samples(self.model, image, pos_examples)
            neg_feats = forward_samples(self.model, image, neg_examples)
            self.pos_feats_all.append(pos_feats)
            self.neg_feats_all.append(neg_feats)
            if len(self.pos_feats_all) > opts['n_frames_long']:
                del self.pos_feats_all[0]
            if len(self.neg_feats_all) > opts['n_frames_short']:
                del self.neg_feats_all[0]
            print('====================================')
            print('Distractor suppression!')
            print('====================================')
            ds_samples = gen_samples(self.ds_generator, self.target_bbox, opts['n_samples'])
            ds_sample_scores = forward_samples(self.model, image, ds_samples, out_layer='fc6')
            ds_idx = ds_sample_scores[:,1].gt(0.0).nonzero().cpu().numpy()
            if len(ds_idx) > 0:
                print('Distractor suppression!')
                #ipdb.set_trace()
                for ds_i,ds_id in enumerate(ds_idx):
                    if ds_i == 0:
                        ds_neg_examples = gen_samples(self.pos_generator, ds_samples[ds_id[0]],opts['n_pos_update'],opts['overlap_pos_update'])
                    else:
                        ds_neg_examples = np.concatenate((ds_neg_examples,gen_samples(self.pos_generator, ds_samples[ds_id[0]],opts['n_pos_update'],opts['overlap_pos_update'])),axis=0)
                ds_neg_feats = forward_samples(self.model, image, ds_neg_examples)
                self.neg_feats_all.append(ds_neg_feats)
                nframes = min(opts['n_frames_short'],len(self.pos_feats_all))
                pos_data = torch.stack(self.pos_feats_all[-nframes:],0).view(-1,self.feat_dim)
                neg_data = stackList(self.neg_feats_all).view(-1,self.feat_dim)
                train(self.model, self.criterion, self.update_optimizer, pos_data, neg_data, opts['maxiter_update'])
        # Short term update
        '''
        if not success:


            nframes = min(opts['n_frames_short'],len(self.pos_feats_all))
            pos_data = stackList(self.pos_feats_all[-nframes:])
            neg_data = stackList(self.neg_feats_all)
            train(self.model, self.criterion, self.update_optimizer, pos_data, neg_data, opts['maxiter_update'])
        '''
        # Long term update
        if i % opts['long_interval'] == 0:

            pos_data = stackList(self.pos_feats_all)
            neg_data = stackList(self.neg_feats_all)
            train(self.model, self.criterion, self.update_optimizer, pos_data, neg_data, opts['maxiter_update'])
        return vot.Rectangle(self.result_bb[-1][0],self.result_bb[-1][1],self.result_bb[-1][2],self.result_bb[-1][3]),target_score
        #return np.array([self.result_bb[-1][0],self.result_bb[-1][1],self.result_bb[-1][2],self.result_bb[-1][3]])
handle = vot.VOT("polygon")

#handle = vot.VOT("rectangle")
selection = handle.region()

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

image = Image.open(imagefile).convert('RGB')
tracker = CONVTracker(image,selection)
i = 0
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    i = i + 1

    image = Image.open(imagefile).convert('RGB')
    region,confidence = tracker.track(image,i)
    handle.report(region,confidence)

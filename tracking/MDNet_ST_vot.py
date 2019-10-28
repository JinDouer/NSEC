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

        print "Iter %d, Loss %.4f" % (iter, loss.data[0])


handle = vot.VOT("rectangle")
selection = handle.region()
init_bbox = [selection.x-1,selection.y-1,selection.width,selection.height]
target_bbox = np.array(init_bbox)
last_target = target_bbox
last_bbreg = target_bbox
model = MDNet(opts['model_path'])
if opts['use_gpu']:
    model = model.cuda()
model.set_learnable_params(opts['ft_layers'])

# Init criterion and optimizer 
criterion = BinaryLoss()
init_optimizer = set_optimizer(model, opts['lr_init'])
update_optimizer = set_optimizer(model, opts['lr_update'])
imagefile = handle.frame()


if not imagefile:
    sys.exit(0)
image = Image.open(imagefile).convert('RGB')


# Train bbox regressor
bbreg_examples = gen_samples(SampleGenerator('uniform', image.size, 0.3, 1.5, 1.1),
                             target_bbox, opts['n_bbreg'], opts['overlap_bbreg'], opts['scale_bbreg'])
bbreg_feats = forward_samples(model, image, bbreg_examples)
bbreg = BBRegressor(image.size)
bbreg.train(bbreg_feats, bbreg_examples, target_bbox)


##class1: pure bg
pos_examples1 = gen_samples(SampleGenerator('gaussian', image.size, 0.1, 1.2),
                           target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])
neg_examples1 = gen_samples(SampleGenerator('uniform', image.size, 1,2,1.1),target_bbox, 1000, [0,0.1])
pos_feats1 = forward_samples(model, image, pos_examples1)
neg_feats1 = forward_samples(model, image, neg_examples1)
##class2:over include
pos_examples2 = gen_samples(SampleGenerator('gaussian', image.size, 0.1, 1.2),
                       target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])
neg_examples2 = gen_samples(SampleGenerator('hole', image.size, 0.3,2.0),target_bbox, 1000, [0,0.5])
pos_feats2 = forward_samples(model, image, pos_examples2)
neg_feats2 = forward_samples(model, image, neg_examples2)
##class3:part+bg
pos_examples3 = gen_samples(SampleGenerator('gaussian', image.size, 0.1, 1.2),
                       target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])
neg_examples3 = gen_samples(SampleGenerator('edge', image.size, 1,2.0,1.1),target_bbox, 1000, [0.2,0.6])
pos_feats3 = forward_samples(model, image, pos_examples3)
neg_feats3 = forward_samples(model, image, neg_examples3)
##class4:pure part
pos_examples4 = gen_samples(SampleGenerator('gaussian', image.size, 0.1, 1.2),target_bbox,
                       opts['n_pos_init'], opts['overlap_pos_init'])
neg_examples4 = gen_samples(SampleGenerator('part', image.size, 1,2.0,1.1),target_bbox, 1000, [0.2,0.6])
pos_feats4 = forward_samples(model, image, pos_examples4)
neg_feats4 = forward_samples(model, image, neg_examples4)
    
    
# Initial training
train(model, criterion, init_optimizer, pos_feats1, neg_feats1, opts['maxiter_init'])
train(model, criterion, init_optimizer, pos_feats2, neg_feats2, opts['maxiter_init'])
train(model, criterion, init_optimizer, pos_feats3, neg_feats3, opts['maxiter_init'])
train(model, criterion, init_optimizer, pos_feats4, neg_feats4, opts['maxiter_init'])
    
##mix
pos_examples = gen_samples(SampleGenerator('gaussian', image.size, 0.1, 1.2),
                           target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

neg_examples = np.concatenate([
                gen_samples(SampleGenerator('uniform', image.size, 1, 2, 1.1), 
                            target_bbox, opts['n_neg_init']//2, opts['overlap_neg_init']),
                gen_samples(SampleGenerator('whole', image.size, 0, 1.2, 1.1),
                            target_bbox, opts['n_neg_init']//2, opts['overlap_neg_init'])])
neg_examples = np.random.permutation(neg_examples)

# Extract pos/neg features
pos_feats = forward_samples(model, image, pos_examples)
neg_feats = forward_samples(model, image, neg_examples)
feat_dim = pos_feats.size(-1)
train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'])


# Init sample generators
sample_generator = SampleGenerator('gaussian', image.size, opts['trans_f'], opts['scale_f'], valid=True)
pos_generator = SampleGenerator('gaussian', image.size, 0.1, 1.2)
neg_generator = SampleGenerator('uniform', image.size, 1.5, 1.2)
# Init pos/neg features for update
pos_feats_all = [pos_feats[:opts['n_pos_update']]]
neg_feats_all = [neg_feats[:opts['n_neg_update']]]
i=0
while True:
    i=i+1
    '''
    with open('/home/prisimage/debug.txt','wb') as f:
        f.write(str(i))
    '''
    imagefile = handle.frame()
    
    if not imagefile:
        break
    image = Image.open(imagefile).convert('RGB')
    # Estimate target bbox
    samples = gen_samples(sample_generator, target_bbox, opts['n_samples'])
    sample_scores = forward_samples(model, image, samples, out_layer='fc6')
    top_scores, top_idx = sample_scores[:,1].topk(5)
    top_idx = top_idx.cpu().numpy()
    target_score = top_scores.mean()
    target_bbox = samples[top_idx].mean(axis=0)
    
    success = target_score > opts['success_thr']
        
    # Expand search area at failure
    if success:
        sample_generator.set_trans_f(opts['trans_f'])
    else:
        sample_generator.set_trans_f(opts['trans_f_expand'])

    # Bbox regression
    if success:
        bbreg_samples = samples[top_idx]
        bbreg_feats = forward_samples(model, image, bbreg_samples)
        bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
        bbreg_bbox = bbreg_samples.mean(axis=0)
    else:
        bbreg_bbox = target_bbox
        
    # Copy previous result at failure
    if not success:
        target_bbox = last_target
        bbreg_bbox = last_bbreg
        
    # Save result
    last_target = target_bbox
    last_bbreg = bbreg_bbox
    
    # Data collect
    if success:
                
        # Draw pos/neg samples
        pos_examples = gen_samples(pos_generator, target_bbox, 
                                   opts['n_pos_update'],
                                   opts['overlap_pos_update'])
        neg_examples = gen_samples(neg_generator, target_bbox, 
                                   opts['n_neg_update'],
                                   opts['overlap_neg_update'])

        # Extract pos/neg features
        pos_feats = forward_samples(model, image, pos_examples)
        neg_feats = forward_samples(model, image, neg_examples)
        pos_feats_all.append(pos_feats)
        neg_feats_all.append(neg_feats)
        if len(pos_feats_all) > opts['n_frames_long']:
            del pos_feats_all[0]
        if len(neg_feats_all) > opts['n_frames_short']:
            del neg_feats_all[0]
          
        
            
    # Short term update
    if not success:
         
            
        nframes = min(opts['n_frames_short'],len(pos_feats_all))
        pos_data = stackList(pos_feats_all[-nframes:])
        neg_data = stackList(neg_feats_all)
        train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])
        
    # Long term update
    elif i % opts['long_interval'] == 0:
        pos_data = stackList(pos_feats_all)
        neg_data = stackList(neg_feats_all)
        train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])
    
    region = vot.Rectangle(target_bbox[0]+1,target_bbox[1]+1,target_bbox[2],target_bbox[3])
    
    handle.report(region)


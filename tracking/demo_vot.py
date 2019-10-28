import glob
import os
import pandas as pd
import argparse
import numpy as np
import torch
import cv2
import time
import sys
import ipdb
sys.path.append(os.getcwd())
from modules.utils import overlap_ratio
from fire import Fire
from tqdm import tqdm
from PIL import Image
from python_REC import CONVTracker

def main(video_dir):
    # load videos
    filenames = sorted(glob.glob(os.path.join('./dataset/OTB',video_dir, "img/*.jpg")),
           key=lambda x: int(os.path.basename(x).split('.')[0]))
    frames = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in filenames]
    gt_bboxes = pd.read_csv(os.path.join('./dataset/OTB',video_dir, "groundtruth_rect.txt"), sep='\t|,| ',
            header=None, names=['xmin', 'ymin', 'width', 'height'],
            engine='python')

    overlap = np.zeros(len(frames))
    overlap[0] = 1
    title = video_dir.split('/')[-1]
    # starting tracking

    img_list = []

    for idx, frame in enumerate(frames):
        if idx == 0:
            bbox = gt_bboxes.iloc[0].values
            #ipdb.set_trace()
            image = Image.fromarray(frame)
            img_list.append(frame)
            tracker = CONVTracker(image,bbox,img_list)
            bbox = np.array([bbox[0]-1, bbox[1]-1,
                    bbox[0]+bbox[2]-1, bbox[1]+bbox[3]-1])
        else:
            image = Image.fromarray(frame)
            bbox = tracker.track(image,idx)
        # bbox xmin ymin xmax ymax
        #ipdb.set_trace()
        frame = cv2.rectangle(frame,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])),
                              (0, 255, 0),
                              2)

        gt_bbox = gt_bboxes.iloc[idx].values
        #ipdb.set_trace()
        overlap[idx] = overlap_ratio(gt_bbox, bbox)[0]
        gt_bbox = (gt_bbox[0], gt_bbox[1],
                   gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3])
        frame = cv2.rectangle(frame,
                              (int(gt_bbox[0]-1), int(gt_bbox[1]-1)), # 0-index
                              (int(gt_bbox[2]-1), int(gt_bbox[3]-1)),
                              (255, 0, 0),
                              1)

        print('Frame {:d}/{:d}, Overlap {:.3f}'
            .format(idx, len(frame), overlap[idx]))
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.putText(frame, str(idx)+':'+str(bbox[0]), (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
        cv2.imshow(title, frame)
        cv2.waitKey(30)
    print('meanIOU: {:.3f}'.format(overlap.mean()))
if __name__ == "__main__":
    Fire(main)

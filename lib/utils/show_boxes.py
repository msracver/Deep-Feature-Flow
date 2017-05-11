# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Haocheng Zhang, Xizhou Zhu
# --------------------------------------------------------

import matplotlib.pyplot as plt
import cv2
import random

def show_boxes(im, dets, classes, scale = 1.0):
    plt.cla()
    plt.axis("off")
    plt.imshow(im)
    for cls_idx, cls_name in enumerate(classes):
        cls_dets = dets[cls_idx]
        for det in cls_dets:
            bbox = det[:4] * scale
            color = (random.random(), random.random(), random.random())
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor=color, linewidth=2.5)
            plt.gca().add_patch(rect)

            if cls_dets.shape[1] == 5:
                score = det[-1]
                plt.gca().text(bbox[0], bbox[1],
                               '{:s} {:.3f}'.format(cls_name, score),
                               bbox=dict(facecolor=color, alpha=0.5), fontsize=9, color='white')
    plt.show()
    return im


def draw_boxes(im, dets, classes, scale = 1.0):
    color_white = (255, 255, 255)
    for cls_idx, cls_name in enumerate(classes):
        cls_dets = dets[cls_idx]
        for det in cls_dets:
            bbox = det[:4] * scale
            bbox = map(int, bbox)
            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=3)

            if cls_dets.shape[1] == 5:
                score = det[-1]
                cv2.putText(im, '%s %.3f' % (cls_name, score), (bbox[0], bbox[1]+10),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, thickness=2)
    return im

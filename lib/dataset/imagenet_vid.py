# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xizhou Zhu
# --------------------------------------------------------

"""
ImageNet VID database
This class loads ground truth notations from standard ImageNet VID XML data formats
and transform them into IMDB format. Selective search is used for proposals, see roidb
function. Results are written as the ImageNet VID format. Evaluation is based on mAP
criterion.
"""

import cPickle
import cv2
import os
import numpy as np

from imdb import IMDB
from imagenet_vid_eval import vid_eval
from ds_utils import unique_boxes, filter_small_boxes


class ImageNetVID(IMDB):
    def __init__(self, image_set, root_path, dataset_path, result_path=None):
        """
        fill basic information to initialize imdb
        """
        det_vid = image_set.split('_')[0]
        super(ImageNetVID, self).__init__('ImageNetVID', image_set, root_path, dataset_path, result_path)  # set self.name

        self.det_vid = det_vid
        self.root_path = root_path
        self.data_path = dataset_path

        self.classes = ['__background__',  # always index 0
                        'airplane', 'antelope', 'bear', 'bicycle',
                        'bird', 'bus', 'car', 'cattle',
                        'dog', 'domestic_cat', 'elephant', 'fox',
                        'giant_panda', 'hamster', 'horse', 'lion',
                        'lizard', 'monkey', 'motorcycle', 'rabbit',
                        'red_panda', 'sheep', 'snake', 'squirrel',
                        'tiger', 'train', 'turtle', 'watercraft',
                        'whale', 'zebra']
        self.classes_map = ['__background__',  # always index 0
                        'n02691156', 'n02419796', 'n02131653', 'n02834778',
                        'n01503061', 'n02924116', 'n02958343', 'n02402425',
                        'n02084071', 'n02121808', 'n02503517', 'n02118333',
                        'n02510455', 'n02342885', 'n02374451', 'n02129165',
                        'n01674464', 'n02484322', 'n03790512', 'n02324045',
                        'n02509815', 'n02411705', 'n01726692', 'n02355227',
                        'n02129604', 'n04468005', 'n01662784', 'n04530566',
                        'n02062744', 'n02391049']

        self.num_classes = len(self.classes)
        self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        print 'num_images', self.num_images

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        image_set_index_file = os.path.join(self.data_path, 'ImageSets', self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            lines = [x.strip().split(' ') for x in f.readlines()]
        if len(lines[0]) == 2:
            self.image_set_index = [x[0] for x in lines]
            self.frame_id = [int(x[1]) for x in lines]
        else:
            self.image_set_index = ['%s/%06d' % (x[0], int(x[2])) for x in lines]
            self.pattern = [x[0]+'/%06d' for x in lines]
            self.frame_id = [int(x[1]) for x in lines]
            self.frame_seg_id = [int(x[2]) for x in lines]
            self.frame_seg_len = [int(x[3]) for x in lines]
        # return image_set_index, frame_id

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        if self.det_vid == 'DET':
            image_file = os.path.join(self.data_path, 'Data', 'DET', index + '.JPEG')
        else:
            image_file = os.path.join(self.data_path, 'Data', 'VID', index + '.JPEG')

        # assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self.load_vid_annotation(index) for index in range(0, len(self.image_set_index))]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def load_vid_annotation(self, iindex):
        """
        for a given index, load image and bounding boxes info from XML file
        :param index: index of a specific image
        :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        index = self.image_set_index[iindex]

        import xml.etree.ElementTree as ET
        roi_rec = dict()
        roi_rec['image'] = self.image_path_from_index(index)
        roi_rec['frame_id'] = self.frame_id[iindex]
        if hasattr(self,'frame_seg_id'):
            roi_rec['pattern'] = self.image_path_from_index(self.pattern[iindex])
            roi_rec['frame_seg_id'] = self.frame_seg_id[iindex]
            roi_rec['frame_seg_len'] = self.frame_seg_len[iindex]

        if self.det_vid == 'DET':
            filename = os.path.join(self.data_path, 'Annotations', 'DET', index + '.xml')
        else:
            filename = os.path.join(self.data_path, 'Annotations', 'VID', index + '.xml')

        tree = ET.parse(filename)
        size = tree.find('size')
        roi_rec['height'] = float(size.find('height').text)
        roi_rec['width'] = float(size.find('width').text)
        #im_size = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION).shape
        #assert im_size[0] == roi_rec['height'] and im_size[1] == roi_rec['width']

        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        valid_objs = np.zeros((num_objs), dtype=np.bool)

        class_to_index = dict(zip(self.classes_map, range(self.num_classes)))
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = np.maximum(float(bbox.find('xmin').text), 0)
            y1 = np.maximum(float(bbox.find('ymin').text), 0)
            x2 = np.minimum(float(bbox.find('xmax').text), roi_rec['width']-1)
            y2 = np.minimum(float(bbox.find('ymax').text), roi_rec['height']-1)
            if not class_to_index.has_key(obj.find('name').text):
                continue
            valid_objs[ix] = True
            cls = class_to_index[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        boxes = boxes[valid_objs, :]
        gt_classes = gt_classes[valid_objs]
        overlaps = overlaps[valid_objs, :]

        assert (boxes[:, 2] >= boxes[:, 0]).all()

        roi_rec.update({'boxes': boxes,
                        'gt_classes': gt_classes,
                        'gt_overlaps': overlaps,
                        'max_classes': overlaps.argmax(axis=1),
                        'max_overlaps': overlaps.max(axis=1),
                        'flipped': False})
        return roi_rec

###################################################################################################
    def evaluate_detections(self, detections):
        """
        top level evaluations
        :param detections: result matrix, [bbox, confidence]
        :return: None
        """
        # make all these folders for results
        result_dir = os.path.join(self.result_path, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        self.write_vid_results(detections)
        info = self.do_python_eval()
        return info

    def evaluate_detections_multiprocess(self, detections):
        """
        top level evaluations
        :param detections: result matrix, [bbox, confidence]
        :return: None
        """
        # make all these folders for results
        result_dir = os.path.join(self.result_path, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        self.write_vid_results_multiprocess(detections)
        info = self.do_python_eval_gen()
        return info

    def get_result_file_template(self):
        """
        :return: a string template
        """
        res_file_folder = os.path.join(self.result_path, 'results')
        filename = 'det_' + self.image_set + '_{:s}.txt'
        path = os.path.join(res_file_folder, filename)
        return path

    def write_vid_results(self, all_boxes):
        """
        write results files in pascal devkit path
        :param all_boxes: boxes to be processed [bbox, confidence]
        :return: None
        """
        print 'Writing {} ImageNetVID results file'.format('all')
        filename = self.get_result_file_template().format('all')
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(self.image_set_index):
                for cls_ind, cls in enumerate(self.classes):
                    if cls == '__background__':
                        continue
                    dets = all_boxes[cls_ind][im_ind]
                    if len(dets) == 0:
                        continue
                    # the imagenet expects 0-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:d} {:d} {:.4f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.
                                format(self.frame_id[im_ind], cls_ind, dets[k, -1],
                                       dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3]))

    def write_vid_results_multiprocess(self, detections):
        """
        write results files in pascal devkit path
        :param all_boxes: boxes to be processed [bbox, confidence]
        :return: None
        """
        print 'Writing {} ImageNetVID results file'.format('all')
        filename = self.get_result_file_template().format('all')
        with open(filename, 'wt') as f:
            for detection in detections:
                all_boxes = detection[0]
                frame_ids = detection[1]
                for im_ind in range(len(frame_ids)):
                    for cls_ind, cls in enumerate(self.classes):
                        if cls == '__background__':
                            continue
                        dets = all_boxes[cls_ind][im_ind]
                        if len(dets) == 0:
                            continue
                        # the imagenet expects 0-based indices
                        for k in range(dets.shape[0]):
                            f.write('{:d} {:d} {:.4f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.
                                    format(frame_ids[im_ind], cls_ind, dets[k, -1],
                                           dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3]))
    
    def do_python_eval(self):
        """
        python evaluation wrapper
        :return: info_str
        """
        info_str = ''
        annopath = os.path.join(self.data_path, 'Annotations', '{0!s}.xml')
        imageset_file = os.path.join(self.data_path, 'ImageSets', self.image_set + '.txt')
        annocache = os.path.join(self.cache_path, self.name + '_annotations.pkl')

        filename = self.get_result_file_template().format('all')
        ap = vid_eval(filename, annopath, imageset_file, self.classes_map, annocache, ovthresh=0.5)
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('AP for {} = {:.4f}'.format(cls, ap[cls_ind-1]))
            info_str += 'AP for {} = {:.4f}\n'.format(cls, ap[cls_ind-1])
        print('Mean AP@0.5 = {:.4f}'.format(np.mean(ap)))
        info_str += 'Mean AP@0.5 = {:.4f}\n\n'.format(np.mean(ap))
        return info_str

    def do_python_eval_gen(self):
        """
        python evaluation wrapper
        :return: info_str
        """
        info_str = ''
        annopath = os.path.join(self.data_path, 'Annotations', '{0!s}.xml')
        imageset_file = os.path.join(self.data_path, 'ImageSets', self.image_set + '_eval.txt')
        annocache = os.path.join(self.cache_path, self.name + '_annotations.pkl')

        with open(imageset_file, 'w') as f:
            for i in range(len(self.pattern)):
                for j in range(self.frame_seg_len[i]):
                    f.write((self.pattern[i] % (self.frame_seg_id[i] + j)) + ' ' + str(self.frame_id[i] + j) + '\n')

        filename = self.get_result_file_template().format('all')
        ap = vid_eval(filename, annopath, imageset_file, self.classes_map, annocache, ovthresh=0.5)
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('AP for {} = {:.4f}'.format(cls, ap[cls_ind-1]))
            info_str += 'AP for {} = {:.4f}\n'.format(cls, ap[cls_ind-1])
        print('Mean AP@0.5 = {:.4f}'.format(np.mean(ap)))
        info_str += 'Mean AP@0.5 = {:.4f}\n\n'.format(np.mean(ap))
        return info_str

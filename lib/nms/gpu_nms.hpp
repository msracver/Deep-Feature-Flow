// ------------------------------------------------------------------
// Deep Feature Flow
// Copyright (c) 2017 Microsoft
// Licensed under The MIT License
// Modified from MATLAB Faster R-CNN (https://github.com/shaoqingren/faster_rcnn)
// ------------------------------------------------------------------
// Based on:
// Faster R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License
// https://github.com/shaoqingren/faster_rcnn
// ------------------------------------------------------------------

void _nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh, int device_id);

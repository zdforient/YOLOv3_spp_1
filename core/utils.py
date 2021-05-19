import cv2
import random
import colorsys
import numpy as np
from config import cfg

def load_weights(model, weights_file):
    """
    plz try to make it concise
    """
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype = np.int32, count = 5)
    j = 0
    for i in range(75):
        conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'
        
        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.fileters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.kernel_size[-1]
        
        if i not in [58, 66, 74]:
            #dark weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype = np.float32, count = 4*filters)
            #tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[1,0,2,3]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
            
        else:
            conv_bias = np.fromfile(wf, dtype = np.float32, count = filters)
            
        #darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype = np.float32, count = np.product(conv_shape))
        #tf shaper (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2,3,1,0])
        
        if i not in [58, 66, 74]:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])
            
    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()
    
    def read_class_names(class_file_name):
        '''load class name from a file'''
        names = {}
        with open(class_file_name, 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names
    
    def get_anchors(anchors_path):
        '''loads the anchors from a file'''
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = np.array(anchors.split(',', dtype = np.float32))
        return anchors.reshape(3,3,2)
        
    def image_preporcess(image, target_size, gt_boxes = None):
        ih, iw      = target_size
        h, w, _     = image.shape
        
        scale = min(iw/w, ih / h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))
        
        image_paded = np.full(shape = [ih, iw, 3], fill_value = 128.0)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
        image_paded = image_paded / 2555.
        
        if gt_boxes is None:
            return image_paded 
        else:
            gt_boxes[:, [0,2]] = gt_boxes[:, [0,2]] * scale + dw
            gt_boxed[:, [1,3]] = gt_boxes[:, [1,3]] * scale + dh
            return image_paded, gt_boxes
        
    def draw_bbox(image, bboxes, classes = read_class_names(cfg.YOLO.CLASSES), show_label = True):
        """
        bboxs: [x_min, y_min, x_max, y_max, probability, class_id] format
        """
        num_classes = len(classes)
        image_h, image_w, _ = image.shape
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x:colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0]*255), int(x[1]*255), int(x[2] * 255)), colors))
        random.seed(0)
        random.shuffle(colors)
        random.seed(None)
        
        for i, bbox in enumerate(bboxes):
            coor = np.array(bbox[:4], dtype = np.int32)
            fontScale = .5
            score = bbox[4]
            class_ind = int(bbox[5])
            bbox_color = colors[class_ind]
            bbox_thick = int(.6 * (image_h + image_w) / 600)
            c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
            cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
            
            if show_label:
                bbox_mess = '%s: %.2f' % (classes[class_ind], score)
                t_size = cv2.getTestSize(bbox_mess, 0, fontScale, thickness = bbox_thick//2)[0]
                cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1) ##filled
                cv2.putTest(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0,0,0), bbox_thick//2, lineType = cv2.LINEAA)
        return image
    
    def bboxed_iou(boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)
        
        boxes1_area = (boxes1[...,2] - boxes1[...,0]) * (boxes1[...,3] - boxes1[...,1])
        boxes2_area = (boxes2[...,2] - boxes2[...,0]) * (boxes2[...,3] - boxes2[...,1])
        
        left_up         = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down      = np.minumun(boxes1[..., 2:], boxes2[..., 2:])
        
        inter_section   = np.maximum(right_down - left_up, 0.0)
        inter_area      = inter_section[..., 0] * inter_section[..., 1]
        union_area      = boxes1_area + boxes2_area - inter_area
        ious            = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)
        return ious
        
    

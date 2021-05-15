#set var values


from easydict import Easydict as edict

__C				= edict()

cfg				= __C
__C.YOLO			= edict()

#set class name
__C.YOLO.CLASSES		= "./data/classes/class.names"
__C.YOLO.ANCHORS		= "./data/anchors/anchors.txt"
__C.YOLO.STRIDES		= [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE 	= 3
__C.YOLO.IOU_LOSS_THRESH	= 0.5

#train options
__C.TRAIN			= edict()
#annotation path
__C.TRAIN.ANNOT_PATH		= "./data/dataset/train.txt"
__C.TRAIN.BATCH_SIZE		= 4
#input image size a*a
__C.TRAIN.INPUT_SIZE		= [416, 608]
#data argumentation
__C.TRAIN_DATA_AUG		= True
#learning rate
__C.TRAIN.LR_INIT		= 1e-3
__C.TRAIN_LR.END		= 1e-6
#warm up?????????
__C.TRAIN.WARMUP_EPOCHS		= 2
__C.TRAIN.EPOCHS		= 30

#test options
__C.TEST			= edict()
__C.TEST.ANNOT_PATH		= "./data/dataset/test.txt"
__C.TEST.BATCH_SIZE 		= 2
__C.TEST.INPUT_SIZE		= [416,608]
__C.TEST.DATA_AUG		= False
__C.TEST.DETECT_IMAGE_PATH	= "./data/dataset/detection/"
__C.TEST.SCORE_THREDHOLD	= .4
__C.TEST.IOU_THRESHOLD		= .45

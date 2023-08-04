import numpy as np
import pickle
num_class = 2  # 类别数
with open('output\ppyolov2_r50vd_dcn\pretrain\ppyolov2_r50vd_dcn_365e_coco.pdparams','rb') as f:  # 预训练模型
    obj = f.read()
weights = pickle.loads(obj, encoding = 'latin1')

weights['yolo_head.yolo_output.0.weight'] = np.zeros([num_class*3+18,1024,1,1],dtype = 'float32')
weights['yolo_head.yolo_output.0.bias'] = np.zeros([num_class*3+18],dtype = 'float32')
weights['yolo_head.yolo_output.1.weight'] = np.zeros([num_class*3+18,512,1,1],dtype = 'float32')
weights['yolo_head.yolo_output.1.bias'] = np.zeros([num_class*3+18],dtype = 'float32')
weights['yolo_head.yolo_output.2.weight'] = np.zeros([num_class*3+18,256,1,1],dtype = 'float32')
weights['yolo_head.yolo_output.2.bias'] = np.zeros([num_class*3+18],dtype = 'float32')

f = open('output\ppyolov2_r50vd_dcn\pretrain\ppyolov2_r50vd_dcn_365e_coco.pdparams','wb')
pickle.dump(weights,f)
f.close()


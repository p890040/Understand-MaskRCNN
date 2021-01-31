from mask_rcnn import TinyMaskRcnn
from rpn.anchor import TinyAnchor

import torch
import numpy as np
import cv2
import sys, os
sys.path.append('..')
from utils import drawDetection

def load_tiny_dataset(path):
    f = open(os.path.join(path, 'annotation.txt'))
    text = f.readlines()
    category = text[0].strip().split(';')
    category_names, category_ids= [],[]
    for c in category:
        names, ids = c.split(':')
        category_names.append(names)
        category_ids.append(ids)
    
    dataset_dict=[]
    for s in text[1:]:
        if(s[0]=='#' and not('#END' in s)):
            class_ids=[]
            bboxes=[]
            masks=[]
            img_name = s.strip().split(':')[1]
            img_shape = cv2.imread(os.path.join(path, img_name)).shape[:2] #H, W
            continue
        elif('#END' in s):
            data={}
            data.update({'img_name':img_name})
            data.update({'img_shape':img_shape})
            data.update({'class_id':class_ids})
            data.update({'bbox':bboxes})
            data.update({'mask':masks})
            dataset_dict.append(data)
        else:
            cls_id, x1, y1, x2, y2, mask = s.strip().split(',') #x_min, y_min, x_max, y_max
            class_ids.append(int(cls_id))
            bboxes.append([float(x1),float(y1),float(x2),float(y2)])
            mask = cv2.imread(os.path.join(path, 'masks', mask), 0)
            mask = cv2.resize(mask, (28,28))
            mask = mask>0
            masks.append(mask)
            
    f.close()
    return dataset_dict

def train_net(data_path, max_epoch = 300):
    dataset_dict = load_tiny_dataset(data_path)
    
    model = TinyMaskRcnn(TinyAnchor('rpn/1280_800_default_anchors.npy').anchor, True, 4)
    model.to('cuda')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    epoch = 0
    while(epoch < max_epoch):
        for data in dataset_dict:
            img_name = data['img_name']
            img = cv2.imread(os.path.join(data_path, img_name))
            img = (img/255)
            gt_boxes =  np.array(data['bbox'], dtype=np.float32)
            gt_ids =  np.array(data['class_id'])
            gt_masks= np.array(data['mask'])
            
            image_input = torch.tensor(img[None,...], dtype=torch.float32, device=torch.device('cuda:0')).permute(0,3,1,2)
            Gts = [torch.tensor(gt_boxes).cuda(), torch.tensor(gt_ids).cuda(), torch.tensor(gt_masks).cuda()]
            
            optimizer.zero_grad()
            outputs = model(image_input, Gts)
            _, _, losses = outputs
            print(losses)
            loss = sum(losses.values())
            loss.backward()
            optimizer.step()
            
        print(f'({epoch}/{max_epoch})')  
        epoch+=1
        
    torch.save(model.state_dict(), 'model_final.pth')
    del model, optimizer

def inference(data_path, score_thresh=0.8):
    model = TinyMaskRcnn(TinyAnchor('rpn/1280_800_default_anchors.npy').anchor, False, 4)
    model_state_dict = torch.load('model_final.pth')
    model.load_state_dict(model_state_dict)
    model.to('cuda')
    
    os.makedirs('results', exist_ok=True)
    for file in os.listdir(data_path):
        if(file[-4:]!='.png' and file[-4:]!='.jpg' and file[-4:]!='.bmp'): continue
        img = cv2.imread(os.path.join(data_path, file))
        img_ = img.copy()
        img = (img/255)
        image_input = torch.tensor(img[None,...], dtype=torch.float32, device=torch.device('cuda:0')).permute(0,3,1,2)
        outputs = model(image_input)
        boxes, scores, classes, mask_full = outputs[0].cpu().numpy(), outputs[1].cpu().numpy(), outputs[2].cpu().numpy(), outputs[3].cpu().numpy()

        keep = scores>score_thresh
        boxes, scores, classes, mask_full=boxes[keep], scores[keep], classes[keep], mask_full[keep]

        out_name = os.path.join('results', file[:-4]+'.jpg')
        drawDetection(img_, out_name, boxes, scores=scores, class_ids=classes, class_names = ['Apple','Banana','Cocona'], masks=mask_full)     
        print(f'Inference {file} ok')
    del model
    

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.getcwd()), 'dataset')
    train_net(data_path, max_epoch = 300)
    inference(data_path, score_thresh = 0.8)












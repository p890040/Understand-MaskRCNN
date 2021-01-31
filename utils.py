import numpy as np
import os
import cv2
import math

#RGB order
color_map = [[255,0,0], # red
             [0,255,0], # green
          [255,255,0],  # yellow
          [255,165,0],  # orange
          [0,0,255],    # blue
          [75,0,130],   # indigo
          [238,130,238],# violet
          [127,127,127],# grey
          [255,255,255]]# white
# [255, 153, 255] Pink
for i in range(20):
    color_map.append([np.random.randint(256), np.random.randint(256), np.random.randint(256)])
for i in range(len(color_map)):
    color_map[i] = color_map[i][::-1]
    

def drawDetection(img, out_name, boxes, scores=None, class_ids=None, class_names=None, masks=None, keypoints=None, mini_mask=False, mode='class_color'):
    if class_ids is None: class_ids = np.zeros(boxes.shape[0])
    boxes = boxes.astype(np.int)
    for i in range(boxes.shape[0]):
        color = color_map[class_ids[i]] if mode == 'class_color' else [np.random.randint(256), np.random.randint(256), np.random.randint(256)]
        cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), color, 2)
        if(class_names is not None):
            class_name = class_names[class_ids[i]]
            cv2.putText(img, f'{class_name}', (boxes[i][0], boxes[i][1]+15), cv2.FONT_HERSHEY_PLAIN, 1, [30, 30, 30], 1, cv2.LINE_AA)
        if(scores is not None):
            cv2.putText(img, f'{scores[i]:.2f}', (boxes[i][0], boxes[i][1]), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0], 1, cv2.LINE_AA)
        if(masks is not None):
            img[masks[i]==True] = (img[masks[i]==True] + color)//2
        if(keypoints is not None):
            for j in range(keypoints.shape[1]):
                x = keypoints[i][j][0]
                y = keypoints[i][j][1]
                cv2.circle(img,(x, y), 1, [255, 153, 255], 3)
                cv2.putText(img, 'p'+str(j+1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 153, 255], 1, cv2.LINE_AA)
                # cv2.putText(img, f'{pred_keypoints[i][j][2]:.2f}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
    cv2.imwrite(out_name, img)
    



from turtle import color
import cv2
from cv2 import VideoCapture
import numpy as np
from zipfile import ZipFile

from utils.torch_utils import select_device
import socket
from models.models import *
from utils.datasets import *
from utils.general import *
import time
from datetime import datetime
import os
import shutil
import opt

IP = "133.22.128.249"
PORT = 8080

file_name  = 'main.zip'
send_check = False

def load_classes(path):
    with open(path,'r') as f:
        names = f.read().split("\n")
    return list(filter(None,names))

out, source, weights, view_img, save_txt, imgsz, cfg, names = \
    opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names

device = select_device(opt.device)
half = device.type != 'cpu'

model = Darknet(cfg,imgsz)
model.load_state_dict(torch.load(weights[0], map_location=device)['model'])

model.to(device).eval()
if half:
    model.half()

names = load_classes(names)
colors = (0,0,255)

def detect(img0):
    H,W,_ =img0.shape
    img0_copy = img0.copy()
    img = letterbox(img0,new_shape=imgsz,auto_size=64)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() ==3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

    center = []
    for i, det in enumerate(pred):  # detections per image

        # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Write results
            for *xyxy, conf, cls in det:
                label = names[int(cls)]
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                x_center = (x1+x2)//2
                y_center = (y1+y2)//2
                center.append([x_center,y_center])
                #if not is_in_parking_line(x_center,y_center):
                #   continue
                label = '%s %.2f' % (names[int(cls)],conf)
                # cv2.rectangle(im0, (x1, y1), (x2, y2), colors, 2)
                # cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
    status = {}
    for key,value in points_dict_c9.items():
        x,y,w,h = value[0]
        cv2.rectangle(img0,(x,y),(x+w,y+w),(0,255,0),3)
        cv2.putText(img0,str(key),(x+10,y+30),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)

        for [x_cen,y_cen] in center:
            if x<= x_cen<=x+w and y<=y_cen<=y+h:
                cv2.rectangle(img0,(x,y),(x+w,y+w),(0,255,0),3)
                cv2.putText(img0,str(key),(x+10,y+30),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)
                status[key] = 1
                break
            else:
                status[key] = 0
    return img0,status

points_dict_c9 = {}
with open("slot_C9.txt","r") as f:
    points = []
    count = 0
    for line in f.read().split():
        x,y,w,h = list(map(int,line.split(",")))
        points.append([x,y,w,h])
        count +=1
        points_dict_c9[int(count)] = points
        points = []

print(points_dict_c9)
if __name__ == '__main__':
    check = 1
    rotate = 0
    frame = 0
    skip = 10
    status = {}

    with torch.no_grad():
        t = time.time()
        while True:
            today = f'{datetime.now().year}_{datetime.now().month}_{datetime.now().day}'
            timenow = f'{datetime.now().hour}_{datetime.now().minute}_{datetime.now().second}'            
            path = r"rtsp://admin:bk123456@192.168.0.55:554/Streaming/channels/1"
            cap = VideoCapture(path)
            
            t = time.time()
            frame += 1
            ret ,img0 = cap.read()
            img0_copy = img0.copy()
            if not ret:
                print(f"Camera fail {today}")
                continue
            img0_copy = cv2.resize(img0_copy,(1280,720))
            cv2.imwrite('C9_original.jpg',img0_copy)
            img0_deteted ,status = detect(img0)

            img0_deteted = cv2.resize(img0_deteted,(1280,720))
            cv2.imwrite('C9.jpg',img0_deteted)

            if datetime.now().hour ==25 and not os.path.isdir(f'C9_Image/{today}__original'):
                os.mkdir(f'C9_Image/{today}__original')
                os.mkdir(f'C9_Image/{today}__detected')

            if 25< datetime.now().hour<26:
                cv2.imwrite(f'C9_Image/{today}__original/{timenow}_original.jpg',img0_copy)
                cv2.imwrite(f'C9_Image/{today}_detected/{timenow}_detected.jpg',img0_deteted)
                print("saved")

            img0_detected = cv2.resize(img0_deteted, (1280, 720))
            cv2.imshow("Image", img0_deteted)
        
            if datetime.now().hour == 25 and not os.path.isfile(f'C9_Image/C9__{today}__original.zip'):
                with ZipFile(f'C9_Image/C9__{today}__original.zip', 'w') as z:
                    lst_dir = os.listdir(f'C9_Image/{today}__original')
                    for i in lst_dir:
                        z.write(f'C9_Image/{today}__original/{i}')
                    #shutil.rmtree(f'C9_Image/{today}')
                        
                with ZipFile(f'C9_Image/C9__{today}__detected.zip', 'w') as z:
                    lst_dir = os.listdir(f'C9_Image/{today}__detected')
                    for i in lst_dir:
                        z.write(f'C9_Image/{today}__detected/{i}')
                        
                send_check = True
            
            with open('frame.txt', 'w+') as f:
                for key, value in status.items():
                    f.write(f"{str(key)} 70 {str(value)}\n")

            with ZipFile(file_name, 'w') as z:
                z.write('C9_original.jpg')
                z.write('C9.jpg')
                z.write('frame.txt')
                                    
            s = socket.socket()
            
            try:
                s.connect((IP, PORT))
                print('connected')
            except:
                print('connect failed')
                
            try:
                s.send(file_name.encode("utf-8"))
                with open(file_name, 'rb') as f:
                    data = f.read()
                    s.sendall(data)
                    print('sent main.zip')
                
                if send_check:
                    s.send(f'C9_Image/C9__{today}__original.zip'.encode("utf-8"))
                    with open(f'C9_Image/C9__{today}__original.zip', 'rb') as f:
                        data = f.read()
                        s.sendall(data)
                        
                    s.send(f'C9_Image/C9__{today}__detected.zip'.encode("utf-8"))
                    with open(f'C9_Image/C9__{today}__detected.zip', 'rb') as f:
                        data = f.read()
                        s.sendall(data)
                        
                    print('sent image')                           
                    send_check = False
            except:
                print('send failed')
            s.close()
            
            key = cv2.waitKey(1)
            print("FPS: ", 1 // (time.time() - t), '\n')
            if key == ord("q"):
                print(status)
                break
            if key == ord("c"):
                check = -check
            if key == ord("r"):
                rotate += 1
            if key == ord("n"):
                frame += skip * 25
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            elif key == ord("p") and frame > skip * 25:
                frame -= skip * 25
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            if key == 32:
                cv2.waitKey()

from unicodedata import name

from cv2 import VideoCapture
from utils.torch_utils import select_device

from models.models import *
from utils.datasets import *
from utils.general import *
import time

import opt


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


out, source, weights, view_img, save_txt, imgsz, cfg, names = \
    opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names
# Initialize
device = select_device("cpu")
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = Darknet(cfg, imgsz)
model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
# model = attempt_load(weights, map_location=device)  # load FP32 model
# imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
model.to(device).eval()
if half:
    model.half()  # to FP16

# Get names and colors
names = load_classes(names)
colors = (0, 0, 255)


# Run inference
# img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
# _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

car_list = []
def detect(frame, allow_class, conf_thres):
    img0 = frame
    H, W, _ = img0.shape
    img = letterbox(img0, new_shape=imgsz, auto_size=64)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, opt.iou_thres, classes=allow_class, agnostic=opt.agnostic_nms)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        im0 = img0.copy()

        # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in det:
                label = names[int(cls)]
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                center_x = int((x1+x2)/2)
                center_y = int((y1+y2)/2)
                # cv2.rectangle(im0, (x1, y1), (x2, y2), colors, 2)
                # cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)
                cv2.circle(im0,(center_x,center_y),5,(0,255,0),-1)
                car_list.append([center_x,center_y])
        
    try:
        return im0
    except:
        return img0
    


if __name__ == '__main__':
    name_path = "ch01_00000000000000900.jpg"
    img = cv2.imread("inference/images/"+ name_path)
    #img = cv2.resize(img, (1920,1080))
    
    with torch.no_grad():

        img = detect(img,[2], 0.35)
        slot_list = []
        index_list = []
        index_list_busy = []
        f = open("Slot_C9.txt","r")
        for index , line in enumerate(f.readlines()):
            slot = line.strip()
            x ,y ,w , h = map(int,slot.split(","))
            slot_list.append([index+1,x,y,w,h])

        for slot in slot_list:
            index,x_s,y_s,w_s,h_s = slot
            status = False
            for x_center ,y_center in car_list:
                if x_s < x_center <x_s+w_s and y_s <y_center<y_s +h_s:
                    cv2.rectangle (img,(x_s,y_s),(x_s+w_s,y_s+h_s),(0,0,255),3)
                    cv2.putText(img,f"{index}",(x_s,y_s),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
                    status = True
                    index_list.append(index)
                    index_list_busy.append(index)
                    break
            if not status:
                index_list.append(index)
                cv2.rectangle (img,(x_s,y_s),(x_s+w_s,y_s+h_s),(0,255,0),3)
                cv2.putText(img,f"{index}",(x_s,y_s),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
            
           
                    
        
        
        # f = open(r'inference\\output\\Send_file.txt','w+')
        # f.write("slot,status\n")
        # for i in index_list:
        #     if i in index_list_busy:
        #         a = "busy"
        #     else:
        #         a = "free"
        #     f.write(f"{ i },{ a }\n")
        # f.close()

        cv2.imwrite(r"inference/output/"+ name_path,img)
        #img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
        cv2.imshow("Image", img)
        cv2.waitKey()


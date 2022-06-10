import cv2
#cap = cv2.VideoCapture(r"D:\yolor\inference\output\C9.jpg")
#img = cap.read()[1]
img = cv2.imread("inference/images/test_drone.jpg")
img = cv2.resize(img, (1280, 720))
rois = cv2.selectROIs("image", img, showCrosshair=False, fromCenter=False)
f = open("spot_file_downpark.txt", "w+")
for roi in rois:
    x, y, w, h = roi
    f.write(f"{x},{y},{w},{h}\n")
f.close()
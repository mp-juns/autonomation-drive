import cv2
import time

DEVICE = 0
WIDTH = 640
HEIGHT = 480
FPS = 30

cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("[INFO] opened:", cap.isOpened())

if not cap.isOpened():
    raise RuntimeError("Camera open failed")

for i in range(30):
    ret, frame = cap.read()
    print(i, ret, None if frame is None else frame.shape)

    if ret and frame is not None:
        cv2.imwrite("test_frame_mjpg.jpg", frame)
        print("[INFO] saved test_frame_mjpg.jpg")
        break

    time.sleep(0.1)

cap.release()

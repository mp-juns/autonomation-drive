import cv2
import time

DEVICE = 0
WIDTH = 640
HEIGHT = 480
FPS = 30

pipeline = (
    f"v4l2src device=/dev/video{DEVICE} ! "
    f"image/jpeg,width={WIDTH},height={HEIGHT},framerate={FPS}/1 ! "
    "jpegdec ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink drop=1 max-buffers=1"
)

print("[INFO] pipeline:", pipeline)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

print("[INFO] opened:", cap.isOpened())

if not cap.isOpened():
    raise RuntimeError("GStreamer camera open failed")

for i in range(30):
    ret, frame = cap.read()
    print(i, ret, None if frame is None else frame.shape)

    if ret and frame is not None:
        cv2.imwrite("test_frame_gst.jpg", frame)
        print("[INFO] saved test_frame_gst.jpg")
        break

    time.sleep(0.1)

cap.release()

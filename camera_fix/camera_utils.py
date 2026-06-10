import cv2
import time


def open_camera(device=0, width=640, height=480, fps=30):
    """
    Raspberry Pi + UVC USB Camera 안정화용 카메라 오픈 함수.
    1) V4L2 + MJPG 강제
    2) 실패 시 GStreamer MJPG pipeline fallback
    """
    print(f"[INFO] Opening camera /dev/video{device}")

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        time.sleep(0.3)

        ret, frame = cap.read()
        if ret and frame is not None:
            print("[INFO] Camera ready: V4L2 + MJPG")
            return cap

        print("[WARN] V4L2 opened but frame read failed. Trying GStreamer.")
        cap.release()
    else:
        print("[WARN] Basic V4L2 open failed. Trying GStreamer.")
        cap.release()

    pipeline = (
        f"v4l2src device=/dev/video{device} ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        "jpegdec ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=1 max-buffers=1"
    )

    print("[INFO] GStreamer pipeline:", pipeline)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Camera open failed: V4L2 and GStreamer both failed")

    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise RuntimeError("Camera opened with GStreamer but frame read failed")

    print("[INFO] Camera ready: GStreamer")
    return cap


def preprocess_frame(frame, width=100, height=60):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (width, height))
    normalized = resized.astype("float32") / 255.0
    return normalized.reshape(height, width, 1)

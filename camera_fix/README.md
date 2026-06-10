# Raspberry Pi UVC Camera Fix Codes

## 1. 지원 포맷 확인
```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
```

## 2. V4L2 단독 테스트
```bash
v4l2-ctl -d /dev/video0 \
  --set-fmt-video=width=640,height=480,pixelformat=MJPG \
  --stream-mmap --stream-count=30
```

## 3. Python 테스트 순서
```bash
python3 test_cam_mjpg.py
python3 test_cam_gst.py
python3 test_cam_yuyv.py
```

## 4. 실제 코드 적용
`Final_RunMain.py`에서 기존:
```python
cap = cv2.VideoCapture(0)
```

변경:
```python
from camera_utils import open_camera
cap = open_camera(device=0, width=640, height=480, fps=30)
```

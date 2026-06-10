# autonomation-drive

Raspberry Pi 기반 자율주행 RC카 프로젝트입니다.  
수업 목표는 정해진 트랙을 10바퀴 안정적으로 완주하는 것이었고, 최종 평가에서 1등을 달성했습니다.

## 핵심 흐름

1. **차량 플랫폼 안정화**
   - 초기 차량은 차체가 길고 무게중심이 맞지 않아 컨트롤러로도 회전이 어려웠습니다.
   - 하드웨어를 다시 출력할 시간이 부족했기 때문에 무게중심을 후방으로 이동시키고, 회전 시 모터 속도 차이를 키워 주행 가능한 플랫폼으로 만들었습니다.

2. **수동 주행 데이터 수집**
   - MIT App Inventor 기반 조이스틱 제어로 차량을 직접 운전하며 학습 데이터를 수집했습니다.
   - 차량이 안정화된 뒤 수집한 데이터가 모델 학습에 더 유효했습니다.

3. **CNN-LSTM 실험**
   - 연속 프레임 정보를 활용하기 위해 CNN-LSTM 기반 하이브리드 모델을 실험했습니다.
   - 하지만 Raspberry Pi 실주행에서는 추론 지연과 jitter가 조향 반응성에 영향을 주었습니다.

4. **최종 모델 단순화**
   - 실주행에서는 복잡한 시계열 모델보다 빠른 반응성이 더 중요하다고 판단했습니다.
   - 최종적으로 NVIDIA End-to-End Driving 스타일의 단일 프레임 CNN 구조로 단순화했습니다.

5. **TFLite INT8 양자화**
   - CNN 모델을 TensorFlow Lite로 변환하고 INT8 양자화를 적용해 Raspberry Pi에서의 추론 지연을 줄였습니다.
   - 이 최적화가 최종 주행 안정성과 완주 시간 개선의 핵심이었습니다.

## 디렉터리

```text
src/
  quantize_cnn_int8.py              # CNN 모델 INT8 TFLite 양자화 스크립트
  run_tflite_cnn_ultrasonic.py      # TFLite CNN + 초음파 정지 실주행 코드
  convert_tflite_hybrid_float16.py  # 기존 CNN-LSTM float16 변환 스크립트
  Final_RunMain_Hybrid_Ultrasonic.py# 기존 CNN-LSTM + 초음파 통합 코드

camera_fix/
  Raspberry Pi UVC 카메라 MJPG/GStreamer 안정화 테스트 코드

docs/
  project_process.md                # 면접/포트폴리오용 프로젝트 프로세스 정리
  raspi_patch_notes.txt             # 라즈베리파이 추론 패치 메모
```

## 대표 실행 예시

### 1. CNN INT8 양자화

```bash
python3 src/quantize_cnn_int8.py \
  --model Model/model_cnn_best.h5 \
  --output Model/model_cnn_int8.tflite \
  --representative-dir data/images \
  --width 200 \
  --height 66 \
  --channels 3
```

### 2. Raspberry Pi 실주행

```bash
python3 src/run_tflite_cnn_ultrasonic.py \
  --model Model/model_cnn_int8.tflite \
  --width 200 \
  --height 66 \
  --camera-size 320 240 \
  --speed 0.30 \
  --stop-distance 30
```

from ultralytics import YOLO
import os
#print("p路徑：", sys.executable)

os.chdir("/home/lab602.demo/.pipeline/")  # 確保工作目錄正確

model = YOLO("yolov8m-seg.pt")

model.train(
    data="/home/lab602.demo/.pipeline/training_data/10-fold/rip_currents_k0.yaml",
    epochs=200,
    imgsz=640,
    batch=16,
    project="/home/lab602.demo/.pipeline/training_data/Result",
    name="result",
    exist_ok=True,
    optimizer='SGD',
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    patience=50,
    degrees=5,
    translate=0.1,
    fliplr=0.5,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
)



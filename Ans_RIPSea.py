import os
import cv2
import numpy as np
from ultralytics import YOLO

print("start")

# --- 1. 設定參數 ---
# ROI 區域 (只看海面，避開傘跟天空)
ROI_Y_MIN_RATIO = 0.15 
ROI_Y_MAX_RATIO = 0.37 

# 輸出的專案路徑
PROJECT_PATH = "/home/lab602.demo/.pipeline/training_data/ANS_RIP"
NAME = "dataWaiao_filtered" 

# 建立輸出資料夾
save_dir = os.path.join(PROJECT_PATH, NAME)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 載入模型
model = YOLO(r"/home/lab602.demo/.pipeline/training_data/Result/200/weights/best.pt")
source_folder = r"/home/lab602.demo/.pipeline/training_data/dataWaiao"

# --- 2. 推論 (關鍵修改：加入 stream=True) ---
print(f"開始處理圖片串流...")

# stream=True: 變成產生器模式，節省記憶體，適合大量圖片
results = model.predict(
    source=source_folder,
    conf=0.1,
    save=False,     # 關閉自動存檔，由我們手動畫
    augment=False,  # 關閉 TTA 避免警告
    verbose=False,
    stream=True     # <--- 關鍵修改：避免記憶體爆掉
)

# --- 3. 手動過濾與繪圖 ---
saved_count = 0
processed_count = 0

# 因為是 stream=True，這裡的 r 會一張一張吐出來
for r in results:
    processed_count += 1
    
    # 讀取原始圖片
    img_display = r.orig_img.copy()
    img_h, img_w = img_display.shape[:2]
    
    # 計算 ROI 邊界
    roi_y_min = int(img_h * ROI_Y_MIN_RATIO)
    roi_y_max = int(img_h * ROI_Y_MAX_RATIO)
    
    # 標記是否有畫框
    has_valid_detection = False
    
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        # 計算中心點 Y
        center_y = (y1 + y2) // 2
        
        # === 過濾邏輯 ===
        if roi_y_min < center_y < roi_y_max:
            has_valid_detection = True
            
            # 畫框 (紅色)
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 畫標籤
            label = f"rip-seg {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img_display, (x1, y1 - 20), (x1 + w, y1), (0, 0, 255), -1)
            cv2.putText(img_display, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # --- 4. 存檔邏輯 ---
    if has_valid_detection:
        img_name = os.path.basename(r.path)
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, img_display)
        saved_count += 1
        print(f"[{processed_count}] 已保留：{img_name}")
    
    # 每處理 100 張顯示一下進度，不然不知道跑到哪了
    if processed_count % 100 == 0:
        print(f"...已處理 {processed_count} 張圖片")

print(f"全部處理完成，共保留 {saved_count} 張圖片。")
print(f"圖片儲存於：{save_dir}")
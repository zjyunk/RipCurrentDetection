import os
from ultralytics import YOLO

print("start")

# --- 1. 設定感興趣區域 (ROI) ---
# 設定你要判斷的 Y 軸範圍 (比例 0.0 ~ 1.0)
# 例如：只看畫面中間偏下的海面區域，避開天空和沙灘
ROI_Y_MIN_RATIO = 0.15  # 區域頂部 (30% 位置)
ROI_Y_MAX_RATIO = 0.37  # 區域底部 (80% 位置)

# 驗證模型
model = YOLO(r"/home/lab602.demo/.pipeline/training_data/Result/200/weights/best.pt")

# 訓練資料
source_folder = r"/home/lab602.demo/.pipeline/training_data/dataWaiao"

# 推論 (save=True 會畫出所有偵測到的框，包含區域外的)
results = model.predict(
    source=source_folder,
    conf=0.1,
    save=True,
    augment=False,
    project="/home/lab602.demo/.pipeline/training_data/ANS_RIP",
    name="dataWaiao4",
)

# 取得輸出路徑
output_dir = results[0].save_dir

# --- 2. 進行區域過濾 ---
for r in results:
    img_h, img_w = r.orig_shape # 取得原始圖片高度與寬度
    
    # 計算像素單位的 ROI 邊界
    roi_y_min = int(img_h * ROI_Y_MIN_RATIO)
    roi_y_max = int(img_h * ROI_Y_MAX_RATIO)

    # 統計「在區域內」的有效偵測數量
    valid_detections_in_roi = 0

    for box in r.boxes:
        # 取得框的座標 (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # 計算框的中心點 Y 座標
        center_y = (y1 + y2) // 2
        
        # 判斷中心點是否在 ROI 範圍內
        if roi_y_min < center_y < roi_y_max:
            valid_detections_in_roi += 1

    # --- 3. 刪除邏輯修改 ---
    # 根據原始檔名拼出輸出圖檔的檔名
    img_name = os.path.basename(r.path)
    save_path = os.path.join(output_dir, img_name)
    
    # 條件改為：如果「ROI 內」沒有偵測到東西，就刪除圖片
    # (即使 YOLO 在區域外有偵測到東西，我們也不要這張圖)
    if valid_detections_in_roi == 0:
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"已刪除為偵測結果)：{img_name}")
        else:
            print(f"找不到要刪的圖片：{save_path}")
    else:
        print(f"共偵測到 {valid_detections_in_roi} 個結果)：{img_name}")

print("偵測結束，僅保留特定區域內有偵測結果的圖片。")
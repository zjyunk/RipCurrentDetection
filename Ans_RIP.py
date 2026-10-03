import os
from ultralytics import YOLO

print("start")
# 驗證模型.best
model = YOLO(r"/home/lab602.demo/.pipeline/training_data/Result/200/weights/best.pt")

# 訓練資料
source_folder = r"/home/lab602.demo/.pipeline/training_data/dataWaiao"


# 推論
results = model.predict(
    source=source_folder,
    conf=0.1,
    save=True,
    augment=False,
    project="/home/lab602.demo/.pipeline/training_data/ANS_RIP",
    name="test",
    #name="test1"
)


#results = model.predict(source=source_folder, conf=0.25, save=True, augment=True)

# 預設輸出資料夾
output_dir = results[0].save_dir  # 這會自動抓到目前 predict 資料夾的路徑

# 移除未判斷圖
for r in results:
    if len(r.boxes) == 0:
        # 根據原始檔名拼出輸出圖檔的檔名
        img_name = os.path.basename(r.path)
        save_path = os.path.join(output_dir, img_name)

        # 如果存在，就刪掉
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"已刪除：{save_path}")
        else:
            print(f"找不到要刪的圖片：{save_path}")

print("處理完成，已保留所有有偵測結果的圖片。")


# Rip Current Segmentation (YOLOv8-seg)

以 YOLOv8m-seg 訓練離岸流（rip current）實例分割模型，並對實地拍攝的海岸影像做推論與篩選。

## 資料集

訓練影像與標註未納入版本控制（體積過大），存放於 Google Drive：

<https://drive.google.com/drive/folders/1op6D2dje7u8djRDMGvTosLEMrPV9V_jR>

資料集與 `training logs/` 來自上游的 **YOLO-Rip-seg** 專案，著作權歸原作者。本 repo 僅包含自行撰寫的程式碼與自行訓練的結果。

下載後需還原的目錄（皆已列入 `.gitignore`）：

```
images/            訓練影像
labels/            YOLO segmentation 標註
with_rips/         有離岸流的原始影像
without_rips/      無離岸流的原始影像
10-fold/           k-fold 的 yaml 與 train/val 清單
dataWaiao/         外澳海灘待推論影像（6,994 張）
test data/         測試影像、影片與標註
```

## 程式

| 檔案 | 用途 |
|---|---|
| `RIP.py` | 訓練 YOLOv8m-seg，200 epochs、imgsz 640、SGD |
| `fix_labels.py` | 標註清理：class id 強制為 0、濾掉 bbox 格式與越界座標 |
| `Ans_RIP.py` | 批次推論，刪除無偵測結果的輸出圖 |
| `Ans_RIP copy.py` | 同上，額外以 ROI（畫面高度 15%–37%）過濾非海面誤判 |
| `Ans_RIPSea.py` | ROI 過濾版的串流推論（`stream=True`），手動繪框、省記憶體 |
| `outfulong.py` | 將推論結果圖序列合成 5 fps 影片 |
| `videoECC.py` | 以 ECC 演算法做影片幀間對齊，消除鏡頭晃動 |

腳本內路徑為訓練機的絕對路徑，換機器需自行修改。

## 訓練結果

`Result/result/` 內含訓練曲線、混淆矩陣、PR/F1 曲線與 `results.csv`（200 epochs）。

模型權重（`best.pt` / `last.pt`，各約 55 MB）未納入版本控制。

## 環境

```bash
pip install -r requirements.txt
```

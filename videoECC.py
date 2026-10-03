import cv2
import numpy as np

# === 參數設定 ===
video_path = r"/home/lab602.demo/.pipeline/training_data/test data/videos (complete)/DJI_0095.MP4"         # 原始影片路徑
output_path = "/home/lab602.demo/.pipeline/training_data/test data/videos_ECC/ECC_DJI_0095.MP4" # 對齊後輸出影片
warp_mode = cv2.MOTION_TRANSLATION  # 僅平移 (可改為 MOTION_EUCLIDEAN、MOTION_AFFINE、MOTION_HOMOGRAPHY)

# === 開啟影片 ===
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# === 取第一幀當作參考 ===
ret, first_frame = cap.read()
if not ret:
    print("無法讀取影片")
    cap.release()
    exit()

prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
out.write(first_frame)

# === 初始化平移矩陣 ===
warp_matrix = np.eye(2, 3, dtype=np.float32)

# === ECC 迭代條件 ===
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)

frame_idx = 1
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    try:
        # 計算平移矩陣
        cc, warp_matrix = cv2.findTransformECC(prev_gray, gray, warp_matrix, warp_mode, criteria)

        # 對齊影像
        aligned = cv2.warpAffine(frame, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        out.write(aligned)
        prev_gray = gray

    except cv2.error as e:
        print(f"Frame {frame_idx}: ECC failed, skip.")
        out.write(frame)
        continue

    frame_idx += 1

cap.release()
out.release()
print("ECC 對齊完成，輸出檔案：", output_path)

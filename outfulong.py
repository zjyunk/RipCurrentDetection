import cv2, glob, os

img_dir = "/home/lab602.demo/.pipeline/training_data/ANS_RIP/0530fulong2"
imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

if len(imgs) == 0:
    print("找不到任何圖片，請確認路徑！")
    exit()

# 讀取第一張影像確認尺寸
frame = cv2.imread(imgs[0])
h, w = frame.shape[:2]

# 建立影片輸出 (5 fps，可改)
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 5, (w, h))

for i, f in enumerate(imgs):
    img = cv2.imread(f)
    if img is None:
        continue
    out.write(img)
    if i % 20 == 0:
        print(f"已處理 {i+1}/{len(imgs)} 張")

out.release()
print("影片輸出完成：output.mp4")
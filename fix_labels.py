import os
import glob

# 標籤路徑
label_root = "/home/lab602.demo/.pipeline/training_data/labels"

def is_bbox_format(values):
    """判斷是不是 YOLO bbox 格式: class cx cy w h"""
    return len(values) == 5

def is_segment_format(values):
    """判斷是不是 segmentation polygon 格式: class x1 y1 x2 y2 ..."""
    return len(values) > 5 and len(values) % 2 == 1

def fix_labels():
    txt_files = glob.glob(os.path.join(label_root, "**/*.txt"), recursive=True)
    print(f"找到 {len(txt_files)} 個標籤檔")

    for txt_path in txt_files:
        fixed_lines = []
        with open(txt_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            # 強制 class id = 0
            parts[0] = "0"

            # bbox 格式 → 忽略
            if is_bbox_format(parts):
                print(f"{txt_path} 有 bbox 格式 (忽略這行): {line.strip()}")
                continue

            # segmentation 格式 → 檢查座標合法性
            if is_segment_format(parts):
                coords = list(map(float, parts[1:]))
                if all(0 <= c <= 1 for c in coords):
                    fixed_lines.append(" ".join([parts[0]] + list(map(str, coords))))
                else:
                    print(f"{txt_path} 有座標超出 [0,1] (忽略): {line.strip()}")
            else:
                print(f"{txt_path} 格式不明 (忽略): {line.strip()}")

        # 覆寫修正後的檔案
        with open(txt_path, "w") as f:
            f.write("\n".join(fixed_lines) + ("\n" if fixed_lines else ""))

        if not fixed_lines:
            print(f"{txt_path} → 全部行被忽略，檔案清空")

if __name__ == "__main__":
    fix_labels()

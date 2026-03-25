"""Download SSD MobileNet v2 COCO model files for the SSD detector backend.

Run from the project root:
    python scripts/download_ssd_model.py
"""
import sys
import urllib.request
import tarfile
from pathlib import Path
import shutil

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
TAR_FILE = MODEL_DIR / "ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
# TensorFlow官方模型下载地址（永久有效）
TAR_URL = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
# 目标文件名映射
FILES = {
    "ssd_mobilenet_v2_coco.pb": "ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb",
    "ssd_mobilenet_v2_coco.pbtxt": (
        "https://raw.githubusercontent.com/dbloisi/dnn_example/master/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
    )
}


def download(url: str, dest: Path):
    print(f"Downloading {dest.name} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"done  ({dest.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return False


def extract_tar(tar_path: Path, dest_dir: Path, file_map: dict):
    """解压tar.gz文件并提取需要的模型文件"""
    print(f"\nExtracting model files from {tar_path.name}...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for dest_name, src_name in file_map.items():
                if dest_name.endswith(".pb"):  # 只处理pb文件，pbtxt单独下载
                    dest_path = dest_dir / dest_name
                    if dest_path.exists():
                        print(f"Already exists: {dest_path}")
                        continue
                    # 提取指定文件
                    tar.extract(src_name, path=dest_dir)
                    # 重命名并移动到models目录
                    src_path = dest_dir / src_name
                    src_path.rename(dest_path)
                    print(f"Extracted: {dest_path}")
            # 清理临时目录
            temp_dir = dest_dir / "ssd_mobilenet_v2_coco_2018_03_29"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        return True
    except Exception as e:
        print(f"Extraction failed: {e}", file=sys.stderr)
        return False


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True
    
    # 1. 处理pb模型文件（通过tar.gz包）
    pb_dest = MODEL_DIR / "ssd_mobilenet_v2_coco.pb"
    if not pb_dest.exists():
        # 下载tar.gz包
        if not TAR_FILE.exists():
            ok = download(TAR_URL, TAR_FILE)
            all_ok = all_ok and ok
        if all_ok and TAR_FILE.exists():
            # 解压并提取pb文件
            ok = extract_tar(TAR_FILE, MODEL_DIR, FILES)
            all_ok = all_ok and ok
        # 删除tar.gz包（可选）
        if TAR_FILE.exists():
            TAR_FILE.unlink()
    else:
        print(f"Already exists: {pb_dest}")
    
    # 2. 处理pbtxt配置文件
    pbtxt_dest = MODEL_DIR / "ssd_mobilenet_v2_coco.pbtxt"
    if not pbtxt_dest.exists():
        ok = download(FILES["ssd_mobilenet_v2_coco.pbtxt"], pbtxt_dest)
        all_ok = all_ok and ok
    else:
        print(f"Already exists: {pbtxt_dest}")
    
    if all_ok:
        print("\n✅ SSD model ready. Start the app with detector_backend='ssd'.")
    else:
        print("\n❌ Some downloads failed. Check your internet connection and retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()
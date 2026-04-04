"""
Benchmark CSV 数据分析脚本

使用方法:
    conda activate cv-group-py311
    python scripts/analyze_benchmark.py

功能:
    1. 读取 logs/ 目录下所有 CSV 文件
    2. 按 (检测器, 手部跟踪, 深度估计) 组合分组
    3. 计算每个组合的: 平均推理时间、检出率、平均置信度、平均距离
    4. 生成汇总表格，用于填入报告 Section 7.2
    5. 检查是否有遗漏的组合
"""

import pandas as pd
import os
from pathlib import Path


def short_detector(name: str) -> str:
    """简写检测器名称，用于表格显示"""
    return name.replace('.pt', '').replace('yolov8', 'yolo8')


def short_hand(name: str) -> str:
    """简写手部跟踪器名称"""
    if 'yolo_pose' in name:
        return 'yolo_pose'
    if 'holistic' in name:
        return 'holistic'
    if 'mediapipe' in name:
        return 'mediapipe'
    return name


def load_all_csvs(log_dir: str = "logs") -> pd.DataFrame:
    """读取所有 CSV 并添加组合标签"""
    dfs = []
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith('.csv'):
            continue
        # Skip non-benchmark CSV files (e.g. analysis summaries)
        if 'summary' in fname or 'analysis' in fname:
            continue
        fpath = os.path.join(log_dir, fname)
        df = pd.read_csv(fpath)
        df['source_file'] = fname
        # 简化 backend 名称
        df['det_short'] = df['detector'].apply(short_detector)
        df['hand_short'] = df['hand_tracker'].apply(short_hand)
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"logs/ 目录中没有找到 CSV 文件")
    return pd.concat(dfs, ignore_index=True)


def analyze_all(df: pd.DataFrame) -> pd.DataFrame:
    """按 (det_short, hand_short, depth_backend) 分组计算统计量"""
    grouped = df.groupby(['det_short', 'hand_short', 'depth_backend'])

    stats = grouped.agg(
        total_frames=('timestamp', 'count'),
        det_ms_mean=('det_ms', 'mean'),
        det_ms_std=('det_ms', 'std'),
        det_conf_mean=('det_conf', 'mean'),
        det_found_rate=('det_found', 'mean'),
        hand_ms_mean=('hand_ms', 'mean'),
        hand_ms_std=('hand_ms', 'std'),
        hand_found_rate=('hand_found', 'mean'),
        depth_ms_mean=('depth_ms', 'mean'),
        pixel_dist_mean=('pixel_dist_cm', 'mean'),
        depth_dist_mean=('depth_dist_cm', 'mean'),
    ).round(2).reset_index()

    return stats


def print_detection_table(stats: pd.DataFrame):
    """打印检测器对比表格 (Scheme 1)"""
    print("\n" + "=" * 80)
    print("Scheme 1 — Object Detection 对比")
    print("=" * 80)
    print(f"{'检测器':12s} | {'手部跟踪':12s} | {'深度':6s} | "
          f"{'帧数':6s} | {'延迟ms':8s} | {'检出率%':8s} | {'置信度':8s}")
    print("-" * 80)

    # 只取 yolo11s + mediapipe 的组合作为基准对比
    baseline = stats[
        (stats['det_short'] == 'yolo11s') &
        (stats['hand_short'] == 'mediapipe')
    ]

    for _, row in stats.drop_duplicates('det_short').iterrows():
        det_row = stats[stats['det_short'] == row['det_short']]
        det_ms = det_row['det_ms_mean'].mean()
        det_found = det_row['det_found_rate'].mean() * 100
        det_conf = det_row['det_conf_mean'].mean()
        n_frames = det_row['total_frames'].sum()

        print(f"{row['det_short']:12s} | "
              f"{'—':12s} | "
              f"{'—':6s} | "
              f"{n_frames:6d} | "
              f"{det_ms:8.1f} | "
              f"{det_found:8.1f} | "
              f"{det_conf:8.3f}")

    print()


def print_hand_table(stats: pd.DataFrame):
    """打印手部跟踪对比表格 (Scheme 2)"""
    print("=" * 80)
    print("Scheme 2 — Hand Tracking 对比")
    print("=" * 80)
    print(f"{'检测器':12s} | {'手部跟踪':12s} | {'深度':6s} | "
          f"{'帧数':6s} | {'延迟ms':8s} | {'检出率%':8s}")
    print("-" * 80)

    for _, row in stats.drop_duplicates('hand_short').iterrows():
        hand_row = stats[stats['hand_short'] == row['hand_short']]
        hand_ms = hand_row['hand_ms_mean'].mean()
        hand_found = hand_row['hand_found_rate'].mean() * 100
        n_frames = hand_row['total_frames'].sum()

        print(f"{'—':12s} | "
              f"{row['hand_short']:12s} | "
              f"{'—':6s} | "
              f"{n_frames:6d} | "
              f"{hand_ms:8.1f} | "
              f"{hand_found:8.1f}")

    print()


def print_depth_table(stats: pd.DataFrame):
    """打印深度估计对比表格 (Scheme 3)"""
    print("=" * 80)
    print("Scheme 3 — Depth Estimation 对比")
    print("=" * 80)
    print(f"{'检测器':12s} | {'手部跟踪':12s} | {'深度':6s} | "
          f"{'帧数':6s} | {'延迟ms':8s} | {'像素距离cm':12s} | {'深度距离cm':12s}")
    print("-" * 80)

    for _, row in stats.drop_duplicates('depth_backend').iterrows():
        depth_row = stats[stats['depth_backend'] == row['depth_backend']]
        depth_ms = depth_row['depth_ms_mean'].mean()
        pixel_dist = depth_row['pixel_dist_mean'].mean()
        depth_dist = depth_row['depth_dist_mean'].mean()
        n_frames = depth_row['total_frames'].sum()

        print(f"{'—':12s} | {'—':12s} | "
              f"{row['depth_backend']:6s} | "
              f"{n_frames:6d} | "
              f"{depth_ms:8.1f} | "
              f"{pixel_dist:12.1f} | "
              f"{depth_dist:12.1f}")

    print()


def print_full_matrix(stats: pd.DataFrame):
    """打印完整组合矩阵"""
    print("=" * 80)
    print("完整组合矩阵 (检测器 × 手部跟踪 × 深度估计)")
    print("=" * 80)

    dets = sorted(stats['det_short'].unique())
    hands = sorted(stats['hand_short'].unique())
    depths = sorted(stats['depth_backend'].unique())

    # 表头
    header_depth = " | ".join([f"{d:^20}" for d in depths])
    print(f"{'':20s} | {header_depth}")
    print("-" * (22 + 23 * len(depths)))

    for det in dets:
        row_strs = []
        for dep in depths:
            # 这个检测器+这个深度下，有哪些手部跟踪的数据
            cell_rows = stats[
                (stats['det_short'] == det) &
                (stats['depth_backend'] == dep)
            ]
            if cell_rows.empty:
                row_strs.append(f"{'—':^20}")
            else:
                hand_results = []
                for _, r in cell_rows.iterrows():
                    hand_results.append(
                        f"{r['hand_short']}:{r['total_frames']:.0f}f"
                    )
                row_strs.append(f"{', '.join(hand_results):^20}")
        print(f"{det:20s} | " + " | ".join(row_strs))

    print()


def check_missing(stats: pd.DataFrame):
    """检查报告需要的组合是否都有数据"""
    print("=" * 80)
    print("组合完整性检查")
    print("=" * 80)

    all_dets = ['yolo11s', 'yolo8n', 'yolo8s', 'yolo8m', 'ssd']
    all_hands = ['mediapipe', 'yolo_pose', 'holistic']
    all_depths = ['pixel', 'midas']

    missing = []
    for det in all_dets:
        for hand in all_hands:
            for dep in all_depths:
                found = stats[
                    (stats['det_short'] == det) &
                    (stats['hand_short'] == hand) &
                    (stats['depth_backend'] == dep)
                ]
                if found.empty:
                    missing.append((det, hand, dep))

    if missing:
        print(f"[WARN] 缺少 {len(missing)} 个组合:")
        for m in missing:
            print(f"   {m[0]} + {m[1]} + {m[2]}")
        print()
        print("建议: 重新测量缺失的组合，每个组合:")
        print("  1. 按 1/2/3 切换到目标组合")
        print("  2. 等待 2-3 秒让系统稳定")
        print("  3. 按 b 保存")
    else:
        print("[OK] 所有组合都已完成测量！")

    print()
    return missing


def save_results(stats: pd.DataFrame, output_path: str = "logs/analysis_summary.csv"):
    """保存汇总结果到 CSV"""
    stats.to_csv(output_path, index=False)
    print(f"汇总结果已保存到: {output_path}")


def main():
    print("=" * 80)
    print("Benchmark 数据分析脚本")
    print("=" * 80)
    print()

    try:
        df = load_all_csvs()
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先运行 python run.py 并按 b 保存数据到 logs/ 目录")
        return

    print(f"共读取 {len(df)} 帧数据，来自 {df['source_file'].nunique()} 个 CSV 文件")
    print()

    stats = analyze_all(df)

    print_detection_table(stats)
    print_hand_table(stats)
    print_depth_table(stats)
    print_full_matrix(stats)

    missing = check_missing(stats)

    # 保存汇总
    save_results(stats)

    if not missing:
        print("=" * 80)
        print("[OK] 数据完整，可以填入报告 Section 7.2!")
        print("=" * 80)


if __name__ == "__main__":
    main()

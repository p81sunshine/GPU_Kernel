"""
运行 reduction_benchmark 于多个输入规模，解析各 kernel 耗时并绘制性能对比图。
用法: python3 plot_reduction.py
"""

import subprocess
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

BINARY = "./build/bin/reduction_benchmark"

# 与 NVIDIA 官方 slide 一致的元素数量（2^17 ~ 2^25）
SIZES = [
    131072,
    262144,
    524288,
    1048576,
    2097152,
    4194304,
    8388608,
    16777216,
    33554432,
]

# kernel 显示名称（与输出行匹配用的前缀关键字）
KERNELS = [
    ("Kernel 1", "1: Interleaved\nDivergent Branches"),
    ("Kernel 2", "2: Interleaved\nBank Conflicts"),
    ("Kernel 3", "3: Sequential\nAddressing"),
    ("Kernel 4", "4: 2x shmem\nSequential"),
    ("Kernel 5", "5: 2x shmem\nWarp Unroll"),
    ("Kernel 6", "6: 2x shmem\nLoop Unroll"),
    ("Kernel 7", "7: Add-on-load\n1x shmem"),
    ("Kernel 8", "8: Grid-stride\n1/4 grid"),
]

# 与 NVIDIA slide 风格接近的颜色
COLORS = [
    "#1f3c88",  # 深蓝
    "#e91e8c",  # 品红
    "#f0c800",  # 黄
    "#00bcd4",  # 青
    "#7b1fa2",  # 紫
    "#b71c1c",  # 深红
    "#2e7d32",  # 深绿
    "#ff6f00",  # 橙
]

MARKERS = ["o", "s", "^", "x", "*", "D", "+", "v"]


def run_benchmark(n: int) -> dict[str, float]:
    """运行一次 benchmark，返回 {kernel_prefix: time_ms}"""
    try:
        result = subprocess.run(
            [BINARY, str(n)],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        print(f"[ERROR] 找不到 {BINARY}，请先编译：cmake --build build --target reduction_benchmark")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"[WARN] n={n} 超时，跳过")
        return {}

    times = {}
    # 在 Performance 段中查找 "Kernel N: ..." 行，提取第一个浮点数（Time ms）
    in_perf = False
    for line in result.stdout.splitlines():
        if "Performance for" in line:
            in_perf = True
            continue
        if not in_perf:
            continue
        # 匹配以 "Kernel" 开头的数据行
        m = re.match(r"^(Kernel \d+).*?(\d+\.\d+)\s+\d+\.\d+", line)
        if m:
            times[m.group(1)] = float(m.group(2))
    return times


def main():
    print(f"共 {len(SIZES)} 个规模 × {len(KERNELS)} 个 kernel，开始采集数据...\n")

    # data[kernel_prefix][size_idx] = time_ms
    data = {k: [] for k, _ in KERNELS}

    for i, n in enumerate(SIZES):
        print(f"  [{i+1}/{len(SIZES)}] n = {n:>10,} ...", end=" ", flush=True)
        times = run_benchmark(n)
        for prefix, _ in KERNELS:
            data[prefix].append(times.get(prefix, float("nan")))
        print("done")

    # ── 绘图 ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("#f8f8f8")
    ax.set_facecolor("#ffffff")

    x = np.array(SIZES)

    for (prefix, label), color, marker in zip(KERNELS, COLORS, MARKERS):
        y = np.array(data[prefix])
        mask = ~np.isnan(y)
        if mask.sum() == 0:
            continue
        ax.plot(
            x[mask], y[mask],
            label=label,
            color=color,
            marker=marker,
            markersize=7,
            linewidth=2,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(SIZES[0] * 0.8, SIZES[-1] * 1.2)

    # x 轴刻度显示原始数字
    ax.set_xticks(SIZES)
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{v:.3f}" if v < 1 else f"{v:.2f}"
    ))

    ax.set_xlabel("# Elements", fontsize=12, labelpad=8)
    ax.set_ylabel("Time (ms)", fontsize=12, labelpad=8)
    ax.set_title("GPU Parallel Reduction — Performance Comparison", fontsize=14, fontweight="bold", pad=14)

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(
        loc="upper left",
        fontsize=9,
        framealpha=0.85,
        ncol=2,
        handlelength=2.5,
    )

    plt.tight_layout()
    out = "reduction_perf.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n图表已保存至 {out}")


if __name__ == "__main__":
    main()

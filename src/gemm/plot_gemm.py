"""
运行 gemm benchmark 于多个矩阵规模，解析各 kernel 的 GFLOPS 并绘制性能对比图。
用法: python3 src/gemm/plot_gemm.py
"""

import subprocess
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

BINARY = "./build/bin/gemm"

# M=N=K 的规模列表
SIZES = [256, 512, 1024, 2048, 4096]

KERNELS = [
    ("gemm_naive",   "Naive\n(16×16 block)"),
    ("gemm_2d_tile", "2D Tiling\n(32×32 tile)"),
    ("cublas",       "cuBLAS"),
]

COLORS = [
    "#1f3c88",  # 深蓝
    "#e91e8c",  # 品红
    "#f0a500",  # 橙黄
]

MARKERS = ["o", "s", "^"]


def run_benchmark(size: int) -> dict[str, float]:
    """运行一次 benchmark，返回 {kernel_name: gflops}"""
    try:
        result = subprocess.run(
            [BINARY, "--bench", str(size), str(size), str(size)],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError:
        print(f"[ERROR] 找不到 {BINARY}，请先编译：cmake --build build --target gemm")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"[WARN] size={size} 超时，跳过")
        return {}

    gflops = {}
    for line in result.stdout.splitlines():
        # 格式: [bench] gemm_naive M=... GFLOPS=2414.18
        m = re.match(r"^\[bench\]\s+(\S+)\s+.*GFLOPS=([\d.]+)", line)
        if m:
            gflops[m.group(1)] = float(m.group(2))
    return gflops


def main():
    print(f"共 {len(SIZES)} 个规模 × {len(KERNELS)} 个 kernel，开始采集数据...\n")

    data = {name: [] for name, _ in KERNELS}

    for i, size in enumerate(SIZES):
        print(f"  [{i+1}/{len(SIZES)}] M=N=K={size} ...", end=" ", flush=True)
        gflops = run_benchmark(size)
        for name, _ in KERNELS:
            data[name].append(gflops.get(name, float("nan")))
        print("done")

    # ── 绘图 ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#f8f8f8")
    ax.set_facecolor("#ffffff")

    x = np.array(SIZES)

    for (name, label), color, marker in zip(KERNELS, COLORS, MARKERS):
        y = np.array(data[name])
        mask = ~np.isnan(y)
        if mask.sum() == 0:
            continue
        ax.plot(
            x[mask], y[mask],
            label=label,
            color=color,
            marker=marker,
            markersize=8,
            linewidth=2.2,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.9,
        )
        # 在每个数据点旁标注 GFLOPS 值
        for xi, yi in zip(x[mask], y[mask]):
            ax.annotate(
                f"{yi:.0f}",
                xy=(xi, yi),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=color,
            )

    ax.set_xscale("log", base=2)
    ax.set_xlim(SIZES[0] * 0.7, SIZES[-1] * 1.4)
    ax.set_ylim(bottom=0)

    ax.set_xticks(SIZES)
    ax.get_xaxis().set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{int(v)}")
    )
    plt.setp(ax.get_xticklabels(), fontsize=10)

    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:.0f}")
    )

    ax.set_xlabel("Matrix Size (M=N=K)", fontsize=12, labelpad=8)
    ax.set_ylabel("GFLOPS", fontsize=12, labelpad=8)
    ax.set_title("GEMM Performance Comparison", fontsize=14, fontweight="bold", pad=14)

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(
        loc="upper left",
        fontsize=10,
        framealpha=0.88,
        handlelength=2.5,
    )

    plt.tight_layout()
    out = "src/gemm/gemm_perf.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n图表已保存至 {out}")


if __name__ == "__main__":
    main()

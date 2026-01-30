#!/usr/bin/env python3
import argparse
import json
import math
import os
import uuid
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

# Keep matplotlib caches in a writable location.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns  # type: ignore
except Exception:
    sns = None

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

try:
    import plotly.graph_objects as go  # type: ignore
except Exception:
    go = None

try:
    from bokeh.plotting import figure  # type: ignore
    from bokeh.models import ColumnDataSource, Whisker  # type: ignore
    from bokeh.io import export_png  # type: ignore
except Exception:
    figure = None
    ColumnDataSource = None
    Whisker = None
    export_png = None

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
LINE_STYLES = ["-", "--", ":", "-."]
# Weight common publication markers (circle/triangle/diamond) more heavily.
MARKERS = ["o", "o", "o", "D", "D", "^", "^", "v", "v", "s", "P", "X", ">", "<", "h", "H", "x", "*"]
FONTS = ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"]
DEFAULT_SIZES = [
    (800, 600),
    (900, 700),
    (1000, 700),
    (1200, 800),
    (1400, 900),
    (1600, 900),
    (1800, 1000),
    (2000, 1200),
]

DEFAULT_EXCLUDE_PREFIX = "synth_"

GRAYSCALE_PALETTE = [
    "#000000",
    "#222222",
    "#444444",
    "#666666",
    "#888888",
]

COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ab",
]


def trend_sigmoid(x, rng):
    # 4-parameter logistic style curve
    x = np.asarray(x, dtype=float)
    xmin, xmax = np.min(x), np.max(x)
    if xmax - xmin < 1e-6:
        return np.full_like(x, rng.uniform(0.5, 2.0))
    top = rng.uniform(5, 200)
    bottom = rng.uniform(0.1, 5)
    mid = rng.uniform(xmin + 0.2 * (xmax - xmin), xmin + 0.8 * (xmax - xmin))
    slope = rng.uniform(0.5, 2.5)
    return bottom + (top - bottom) / (1 + np.exp(-slope * (x - mid) / (xmax - xmin)))

ENGINE_WEIGHTS = {
    "mpl": 0.4,
    "seaborn": 0.2,
    "pandas": 0.15,
    "plotly": 0.15,
    "bokeh": 0.1,
}
AUTO_SKIP_ENGINES = {"plotly", "bokeh"}


def rand_choice(rng, items):
    return items[int(rng.integers(0, len(items)))]


def weighted_choice(rng, choices):
    items = list(choices.items())
    labels = [item[0] for item in items]
    weights = np.array([item[1] for item in items], dtype=float)
    weights = weights / weights.sum()
    idx = rng.choice(len(labels), p=weights)
    return labels[int(idx)]


def map_linestyle_plotly(style):
    return {"-": "solid", "--": "dash", ":": "dot", "-.": "dashdot"}.get(style, "solid")


def map_linestyle_bokeh(style):
    return {"-": "solid", "--": "dashed", ":": "dotted", "-.": "dotdash"}.get(style, "solid")


def map_marker_plotly(marker):
    return {
        "o": "circle",
        "s": "square",
        "D": "diamond",
        "^": "triangle-up",
        "v": "triangle-down",
        ">": "triangle-right",
        "<": "triangle-left",
        "P": "cross",
        "X": "x",
        "h": "hexagon",
        "H": "hexagon2",
        "x": "x",
        "*": "star",
    }.get(marker, "circle")


def map_marker_bokeh(marker):
    return {
        "o": "circle",
        "s": "square",
        "D": "diamond",
        "^": "triangle",
        "v": "inverted_triangle",
        ">": "triangle",
        "<": "triangle",
        "P": "cross",
        "X": "x",
        "h": "hex",
        "H": "hex",
        "x": "x",
        "*": "asterisk",
    }.get(marker, "circle")


def available_engines():
    engines = ["mpl"]
    if sns is not None:
        engines.append("seaborn")
    if pd is not None:
        engines.append("pandas")
    if go is not None:
        engines.append("plotly")
    if figure is not None and export_png is not None:
        engines.append("bokeh")
    return engines


def load_size_distribution(images_dir: Path, max_samples=200):
    if Image is None or not images_dir.exists():
        return None
    sizes = []
    for img in images_dir.iterdir():
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        if img.name.startswith(DEFAULT_EXCLUDE_PREFIX):
            continue
        try:
            with Image.open(img) as im:
                sizes.append(im.size)
        except Exception:
            continue
        if len(sizes) >= max_samples:
            break
    if not sizes:
        return None
    counts = Counter(sizes)
    return counts


def load_size_distribution_with_filter(images_dir: Path, max_samples=500, exclude_prefix=DEFAULT_EXCLUDE_PREFIX):
    if Image is None or not images_dir.exists():
        return None
    sizes = []
    for img in images_dir.iterdir():
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        if exclude_prefix and img.name.startswith(exclude_prefix):
            continue
        try:
            with Image.open(img) as im:
                sizes.append(im.size)
        except Exception:
            continue
        if len(sizes) >= max_samples:
            break
    if not sizes:
        return None
    return Counter(sizes)


def compute_label_stats(labels_dir: Path, exclude_prefix=DEFAULT_EXCLUDE_PREFIX):
    if not labels_dir.exists():
        return None
    line_counts = []
    points_per_line = []
    top_ratios = []
    bottom_ratios = []
    error_presence = {
        "any": 0,
        "top_only": 0,
        "bottom_only": 0,
        "both": 0,
        "none": 0,
        "total": 0,
    }
    plot_heights = []

    for path in labels_dir.iterdir():
        if path.suffix.lower() != ".json":
            continue
        if exclude_prefix and path.name.startswith(exclude_prefix):
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        line_counts.append(len(data))

        for line in data:
            points = line.get("points", [])
            data_points = []
            anchors = {}
            for p in points:
                label = p.get("label", "")
                if label:
                    anchors[label] = p
                else:
                    data_points.append(p)
            points_per_line.append(len(data_points))

            ymin = anchors.get("ymin")
            ymax = anchors.get("ymax")
            plot_h = None
            if ymin and ymax:
                try:
                    plot_h = abs(float(ymax["y"]) - float(ymin["y"]))
                except Exception:
                    plot_h = None
            if plot_h and plot_h > 1:
                plot_heights.append(plot_h)

            for p in data_points:
                top = float(p.get("topBarPixelDistance", 0) or 0)
                bottom = float(p.get("bottomBarPixelDistance", 0) or 0)
                has_top = top > 0
                has_bottom = bottom > 0
                error_presence["total"] += 1
                if has_top and has_bottom:
                    error_presence["both"] += 1
                    error_presence["any"] += 1
                elif has_top:
                    error_presence["top_only"] += 1
                    error_presence["any"] += 1
                elif has_bottom:
                    error_presence["bottom_only"] += 1
                    error_presence["any"] += 1
                else:
                    error_presence["none"] += 1

                if plot_h and plot_h > 1:
                    if has_top:
                        top_ratios.append(top / plot_h)
                    if has_bottom:
                        bottom_ratios.append(bottom / plot_h)

    stats = {
        "line_counts": line_counts,
        "points_per_line": points_per_line,
        "error_top_ratios": top_ratios,
        "error_bottom_ratios": bottom_ratios,
        "error_presence": error_presence,
        "plot_heights": plot_heights,
    }
    return stats


def sample_size(rng, size_counts):
    if not size_counts:
        return rand_choice(rng, DEFAULT_SIZES)
    sizes = list(size_counts.keys())
    weights = np.array([size_counts[s] for s in sizes], dtype=float)
    weights = weights / weights.sum()
    idx = rng.choice(len(sizes), p=weights)
    return sizes[int(idx)]


def make_timepoints(rng, n, scale="linear", span=None):
    if scale == "log":
        start = rng.uniform(0.1, 1.0)
        stop = rng.uniform(30, 300) if span is None else span
        return np.logspace(math.log10(start), math.log10(stop), n)
    if scale == "dense_early":
        max_span = rng.uniform(20, 360) if span is None else span
        n_early = max(3, int(n * rng.uniform(0.4, 0.7)))
        n_late = n - n_early
        early = np.sort(rng.uniform(0, max_span * 0.1, size=n_early))
        late = np.linspace(max_span * 0.15, max_span, n_late)
        return np.sort(np.concatenate([early, late]))
    if rng.random() < 0.35:
        base = np.linspace(0, rng.uniform(20, 360) if span is None else span, n)
        jitter = rng.normal(0, 0.05 * base.max(), size=n)
        vals = np.clip(base + jitter, 0, None)
        return np.sort(vals)
    return np.linspace(0, rng.uniform(20, 360) if span is None else span, n)


def trend_exponential_decay(x, rng):
    k = rng.uniform(0.01, 0.08)
    a = rng.uniform(30, 300)
    c = rng.uniform(0, 20)
    return a * np.exp(-k * x) + c


def trend_peak(x, rng):
    peak = rng.uniform(0.1 * x.max(), 0.5 * x.max())
    rise = np.exp(-((x - peak) ** 2) / (2 * (0.15 * x.max() + 1) ** 2))
    scale = rng.uniform(30, 200)
    baseline = rng.uniform(0, 20)
    return baseline + scale * rise


def trend_linear(x, rng, allow_negative=False):
    slope = rng.uniform(-1.5, 1.5)
    intercept = rng.uniform(-20, 80)
    y = slope * x + intercept
    if not allow_negative:
        y = y - y.min() + rng.uniform(5, 25)
    return y


def trend_sawtooth(x, rng):
    period = rng.uniform(2, 6)
    amp = rng.uniform(80, 800)
    baseline = rng.uniform(10, 80)
    return baseline + amp * (1 - ((x / period) % 1))


def trend_step(x, rng):
    n = len(x)
    if n < 3:
        return np.full_like(x, rng.uniform(10, 80))
    steps = int(rng.integers(2, 5))
    steps = min(steps, n - 1)
    cuts = np.sort(rng.choice(x[1:-1], size=steps - 1, replace=False))
    levels = rng.uniform(10, 80, size=steps)
    y = np.zeros_like(x)
    for i, xi in enumerate(x):
        idx = np.searchsorted(cuts, xi, side="right")
        y[i] = levels[idx]
    return y


def trend_oscillation(x, rng):
    amp = rng.uniform(5, 40)
    freq = rng.uniform(0.02, 0.2)
    baseline = rng.uniform(20, 80)
    return baseline + amp * np.sin(2 * math.pi * freq * x)


def generate_series(x, rng, allow_negative=False, pattern=None):
    if pattern == "sawtooth":
        y = trend_sawtooth(x, rng)
    elif pattern == "peak":
        y = trend_peak(x, rng)
    elif pattern == "linear":
        y = trend_linear(x, rng, allow_negative=allow_negative)
    elif pattern == "sigmoid":
        y = trend_sigmoid(x, rng)
    else:
        funcs = [
            trend_exponential_decay,
            trend_peak,
            trend_linear,
            trend_sigmoid,
            trend_sawtooth,
            trend_step,
            trend_oscillation,
        ]
        func = rng.choice(funcs)
        if func is trend_linear:
            y = func(x, rng, allow_negative=allow_negative)
        else:
            y = func(x, rng)
    noise = rng.normal(0, 0.03 * (np.max(y) - np.min(y) + 1), size=len(x))
    return y + noise


def make_errors(x, y, rng, scale="linear", style="mixed"):
    n = len(x)
    if style == "none":
        return np.zeros(n), np.zeros(n)
    has_errors = rng.random() < 0.8
    if not has_errors:
        return np.zeros(n), np.zeros(n)
    if scale == "log":
        frac = rng.uniform(0.05, 0.3, size=n)
        low = y * frac
        high = y * frac * rng.uniform(0.8, 1.3, size=n)
        low = np.minimum(low, y * 0.8)
        low = np.maximum(low, y * 0.05)
    else:
        span = (np.max(y) - np.min(y) + 1)
        base = rng.uniform(0.02, 0.2) * span
        jitter = rng.uniform(0.0, 0.7, size=n)
        low = base * (0.4 + jitter)
        high = base * (0.4 + rng.uniform(0.0, 0.7, size=n))
    mask = rng.random(size=n) < 0.15
    low[mask] = 0
    mask = rng.random(size=n) < 0.15
    high[mask] = 0
    return low, high


def build_line_name(rng, idx):
    prefixes = ["Dose", "Placebo", "Cohort", "Group", "Line", "Treatment", "PK", "Study"]
    suffixes = ["A", "B", "C", "High", "Low", "ER", "XR", "Control"]
    unit = rng.choice(["mg", "mg/kg", "mg/m2", "µg/mL"])
    amount = rng.choice([1, 3, 5, 10, 15, 30, 50, 100, 300, 700, 1500])
    if rng.random() < 0.35:
        return f"{amount} {unit} - Part {rng.integers(1, 3)}"
    if rng.random() < 0.5:
        return f"{rng.choice(prefixes)} {amount} {unit} (n={rng.choice([3,6,8,10,27,35])})"
    return f"{rng.choice(prefixes)}_{idx}_{rng.choice(suffixes)}"


def make_style(rng, grayscale=False, color_override=None):
    if color_override:
        color = color_override
    elif grayscale:
        color = rand_choice(rng, GRAYSCALE_PALETTE)
    else:
        color = rand_choice(rng, COLOR_PALETTE)
    marker = rand_choice(rng, MARKERS)
    linestyle = rand_choice(rng, LINE_STYLES)
    markerface = "none" if rng.random() < 0.45 else color
    if rng.random() < 0.2:
        linewidth = rng.uniform(0.6, 1.1)
    elif rng.random() < 0.25:
        linewidth = rng.uniform(2.4, 3.6)
    else:
        linewidth = rng.uniform(1.2, 2.2)
    if rng.random() < 0.25:
        elinewidth = rng.uniform(0.6, 0.9)
    elif rng.random() < 0.2:
        elinewidth = rng.uniform(1.6, 2.4)
    else:
        elinewidth = rng.uniform(0.8, 1.4)
    return {
        "color": color,
        "marker": marker,
        "linestyle": linestyle,
        "markerface": markerface,
        "linewidth": linewidth,
        "markersize": rng.uniform(4, 8),
        "capsize": rng.uniform(2, 6),
        "capthick": rng.uniform(0.8, 1.2),
        "elinewidth": elinewidth,
    }


def load_visual_stats(path: Path):
    if not path or not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def derive_visual_priors(vstats):
    if not vstats:
        return {}
    priors = {}
    if vstats.get("grid_present"):
        priors["grid_prob"] = float(np.mean(vstats["grid_present"]))
    if vstats.get("marker_scores"):
        m = float(np.mean(vstats["marker_scores"]))
        # scale marker presence into a probability
        priors["marker_prob"] = max(0.2, min(0.95, m * 8))
    if vstats.get("cap_scores"):
        c = float(np.mean(vstats["cap_scores"]))
        priors["cap_prob"] = max(0.1, min(0.9, c * 10))
    if vstats.get("dominant_colors"):
        counts = []
        for c in vstats["dominant_colors"]:
            est = max(1, min(8, int(round(c / 4))))
            counts.append(est)
        priors["series_counts"] = counts
    if vstats.get("bg_colors"):
        bg = Counter(vstats["bg_colors"])
        priors["bg_palette"] = [
            tuple(int(x) for x in k.split("-")) for k, _ in bg.most_common(10)
        ]
    return priors


def pick_pattern(rng, stats=None):
    patterns = {
        "rise_decay": 0.18,
        "small_range": 0.1,
        "dotted_open": 0.08,
        "negative_response": 0.1,
        "multi_line_clutter": 0.15,
        "log_pk": 0.15,
        "log_multi": 0.12,
        "sawtooth": 0.07,
        "log_sparse": 0.05,
        "dense_early": 0.07,
        "mono_decay": 0.06,
        "dual_axis": 0.05,
        "inset": 0.06,
        "pre_dose": 0.05,
        "bql_drop": 0.05,
        "grouped_bar": 0.08,
        "scatter_only": 0.05,
        "line_end_labels": 0.05,
        "sigmoid": 0.06,
        "stepped": 0.05,
        "boxplot": 0.05,
    }
    if stats and stats.get("error_presence") and stats["error_presence"].get("total", 0) > 0:
        any_prob = stats["error_presence"]["any"] / stats["error_presence"]["total"]
        if any_prob < 0.5:
            patterns["sawtooth"] += 0.05
            patterns["log_sparse"] += 0.05
            patterns["rise_decay"] = max(0.1, patterns["rise_decay"] - 0.05)
    if stats and stats.get("line_counts"):
        avg_lines = float(np.mean(stats["line_counts"]))
        if avg_lines >= 3.5:
            patterns["multi_line_clutter"] += 0.05
            patterns["log_multi"] += 0.03
    return weighted_choice(rng, patterns)


def generate_spec(rng, size_counts, stats=None, visual_priors=None, pattern_override=None):
    width, height = sample_size(rng, size_counts)
    dpi = int(rand_choice(rng, [96, 100, 110, 120]))
    pattern = pattern_override or pick_pattern(rng, stats)

    log_x = pattern in {"log_pk"} and rng.random() < 0.2
    log_y = pattern in {"log_pk", "log_multi", "log_sparse", "mono_decay", "bql_drop"} or (pattern == "inset" and rng.random() < 0.5)
    if pattern in {"grouped_bar", "boxplot"}:
        log_y = False

    if stats and stats.get("points_per_line"):
        base_points = int(rng.choice(stats["points_per_line"]))
        base_points = max(3, min(base_points, 20))
    else:
        base_points = int(rng.integers(5, 14))

    box_tick_labels = None
    bar_tick_labels = None
    if pattern == "dense_early":
        n_points = max(base_points, int(rng.integers(10, 18)))
        x = make_timepoints(rng, n_points, scale="dense_early", span=rng.uniform(40, 200))
    elif pattern == "pre_dose":
        n_points = max(base_points, int(rng.integers(8, 16)))
        post = make_timepoints(rng, n_points - 2, scale="dense_early", span=rng.uniform(40, 200))
        pre = -np.sort(rng.uniform(1, rng.uniform(10, 60), size=2))
        x = np.sort(np.concatenate([pre, post]))
    elif pattern == "grouped_bar":
        n_points = int(rng.integers(4, 7))
        x = np.arange(n_points, dtype=float)
        if rng.random() < 0.7:
            start = int(rng.integers(0, 3))
            step = int(rand_choice(rng, [1, 2, 3, 4, 6]))
            bar_tick_labels = [str(start + step * i) for i in range(n_points)]
        else:
            bar_tick_labels = [str(i) for i in range(n_points)]
    elif pattern == "boxplot":
        n_points = max(5, int(rng.integers(5, 9)))
        x = np.arange(n_points, dtype=float)
        if rng.random() < 0.7:
            start = int(rng.integers(0, 3))
            step = int(rand_choice(rng, [2, 3, 4, 6]))
            box_tick_labels = [f"{start + step * i} months" for i in range(n_points)]
        else:
            box_tick_labels = [str(i) for i in range(n_points)]
    elif pattern == "sawtooth":
        n_points = max(base_points, int(rng.integers(6, 10)))
        x = make_timepoints(rng, n_points, scale="linear", span=rng.uniform(5, 10))
    elif pattern == "sigmoid":
        n_points = max(base_points, int(rng.integers(6, 12)))
        x = make_timepoints(rng, n_points, scale="linear", span=rng.uniform(10, 60))
    elif pattern == "small_range":
        n_points = min(base_points, int(rng.integers(4, 7)))
        x = make_timepoints(rng, n_points, scale="linear", span=rng.uniform(20, 60))
    elif pattern in {"log_sparse", "log_multi", "mono_decay", "bql_drop"}:
        n_points = max(base_points, int(rng.integers(6, 12)))
        x = make_timepoints(rng, n_points, scale="linear", span=rng.uniform(50, 350))
    else:
        n_points = base_points
        x = make_timepoints(rng, n_points, scale="log" if log_x else "linear")

    if stats and stats.get("line_counts"):
        base_lines = int(rng.choice(stats["line_counts"]))
        base_lines = max(1, min(base_lines, 8))
    elif visual_priors and visual_priors.get("series_counts"):
        base_lines = int(rng.choice(visual_priors["series_counts"]))
        base_lines = max(1, min(base_lines, 8))
    else:
        base_lines = int(rng.integers(1, 5))

    if pattern == "multi_line_clutter":
        num_lines = max(base_lines, int(rng.integers(4, 8)))
    elif pattern in {"log_multi", "log_pk"}:
        num_lines = max(base_lines, int(rng.integers(3, 7)))
    elif pattern in {"small_range", "dotted_open"}:
        num_lines = 2
    elif pattern == "mono_decay":
        num_lines = 1
    elif pattern == "bql_drop":
        num_lines = max(3, base_lines)
    elif pattern == "grouped_bar":
        num_lines = int(rng.integers(2, 4))
    elif pattern == "scatter_only":
        num_lines = int(rng.integers(3, 5))
    elif pattern == "sigmoid":
        num_lines = int(rng.integers(2, 4))
    elif pattern == "stepped":
        num_lines = int(rng.integers(1, 3))
    elif pattern == "boxplot":
        num_lines = int(rng.integers(1, 4))
    else:
        num_lines = base_lines

    lines = []
    single_hue = rng.random() < 0.2
    base_color = rand_choice(rng, ["#1f77b4", "#1c5aa6", "#2c7fb8", "#0f4c81"])

    for idx in range(num_lines):
        allow_negative = (not log_y) and (pattern == "negative_response" or rng.random() < 0.15)
        if pattern == "sawtooth":
            y = generate_series(x, rng, allow_negative=allow_negative, pattern="sawtooth")
        elif pattern == "rise_decay":
            y = generate_series(x, rng, allow_negative=allow_negative, pattern="peak")
        elif pattern == "small_range":
            base = rng.uniform(7.5, 8.3)
            y = base + rng.normal(0, 0.1, size=len(x))
        elif pattern == "mono_decay":
            y = trend_exponential_decay(x, rng)
        elif pattern == "bql_drop":
            y = generate_series(x, rng, allow_negative=allow_negative)
            # force a few points near BQL
            bql = np.min(y) + 0.05 * (np.max(y) - np.min(y))
            drop_idx = rng.choice(len(y), size=max(1, len(y)//6), replace=False)
            y[drop_idx] = bql * rng.uniform(0.2, 1.0, size=len(drop_idx))
        elif pattern == "grouped_bar":
            base = rng.uniform(0.15, 0.4)
            trend = np.linspace(base, base * rng.uniform(0.5, 0.9), len(x))
            y = trend + rng.normal(0, 0.02, size=len(x))
        elif pattern == "boxplot":
            # placeholder medians (actual box stats computed below)
            base = rng.uniform(0.2, 1.0)
            trend = np.linspace(base, base * rng.uniform(0.6, 1.2), len(x))
            y = trend + rng.normal(0, 0.05, size=len(x))
        elif pattern == "sigmoid":
            y = generate_series(x, rng, allow_negative=allow_negative, pattern="sigmoid")
        elif pattern == "stepped":
            y = trend_step(x, rng)
        else:
            y = generate_series(x, rng, allow_negative=allow_negative)

        if log_y:
            y = np.maximum(y, rng.uniform(0.1, 5))

        if pattern in {"log_pk", "log_multi", "log_sparse", "mono_decay", "dense_early", "bql_drop"}:
            grayscale = rng.random() < 0.4
        else:
            grayscale = rng.random() < 0.1
        if pattern == "boxplot":
            grayscale = rng.random() < 0.8
        if pattern in {"dotted_open"} and idx == 0:
            style = make_style(rng, grayscale=grayscale, color_override=base_color if single_hue else None)
            style["linestyle"] = ":"
            style["markerface"] = "none"
        else:
            style = make_style(rng, grayscale=grayscale, color_override=base_color if single_hue else None)

        if pattern == "scatter_only":
            style["linestyle"] = "None"
        if pattern == "stepped":
            style["drawstyle"] = "steps-post"
        if pattern == "grouped_bar":
            style["marker"] = None
        if pattern == "boxplot":
            style["marker"] = None

        # Drop markers for some lines based on visual priors
        if visual_priors and "marker_prob" in visual_priors:
            if rng.random() > visual_priors["marker_prob"]:
                style["marker"] = None
        else:
            if rng.random() < 0.15:
                style["marker"] = None

        if rng.random() < 0.2:
            style["markerface"] = "none"
        if rng.random() < 0.15:
            style["capsize"] = 0
            style["capthick"] = 0

        label = build_line_name(rng, idx)
        lines.append({
            "label": label,
            "x": x,
            "y": y,
            "err_low": None,
            "err_high": None,
            "err_style": "none" if pattern in {"sawtooth", "log_sparse"} else "mixed",
            "style": style,
        })

    xlabel = rand_choice(rng, ["Time (h)", "Weeks", "Days", "Months", "Time (days)"])
    ylabel = rand_choice(rng, [
        "Concentration (ng/mL)",
        "A1c (%)",
        "C-peptide/glucose ratio",
        "Response (log2 fold change)",
        "Serum concentration (ng/mL)",
    ])
    if pattern == "negative_response":
        ylabel = "C-peptide AUC 0–2 h response (log2 fold change)"
    if log_y:
        ylabel = rand_choice(rng, [
            "Serum concentration (ng/mL)",
            "Mean plasma concentration (ng/mL)",
            "Limoniplmab Serum Concentration (µg/mL)",
        ])
    title = rand_choice(rng, ["Figure 1", "Panel A", "Dose response", "Pharmacokinetic profile"])

    if visual_priors and visual_priors.get("bg_palette"):
        bg = rand_choice(rng, visual_priors["bg_palette"])
        bg_color = f"#{bg[0]*16:02x}{bg[1]*16:02x}{bg[2]*16:02x}"
    else:
        bg_color = rand_choice(rng, ["white", "#f7f7f7", "#fcfcfc"])

    # bar offsets for grouped bars
    if pattern == "grouped_bar":
        base_x = lines[0]["x"]
        if len(base_x) > 1:
            diffs = np.diff(np.sort(base_x))
            spacing = float(np.median(diffs)) if np.all(diffs > 0) else 1.0
        else:
            spacing = 1.0
        if not np.isfinite(spacing) or spacing <= 0:
            spacing = 1.0
        group_width = 0.8 * spacing
        bar_width = group_width / max(1, len(lines))
        offsets = np.linspace(-group_width / 2 + bar_width / 2,
                              group_width / 2 - bar_width / 2,
                              len(lines))
        for line, off in zip(lines, offsets):
            line["x_draw"] = line["x"] + off
            line["bar_width"] = bar_width
    elif pattern == "boxplot":
        box_width = 0.7 / max(1, len(lines))
        offsets = np.linspace(-0.35 + box_width / 2, 0.35 - box_width / 2, len(lines))
        for line, off in zip(lines, offsets):
            line["x_draw"] = line["x"] + off
            line["box_width"] = box_width
    else:
        for line in lines:
            line["x_draw"] = line["x"]

    # assign axis for dual-axis plots
    if pattern == "dual_axis":
        for i, line in enumerate(lines):
            line["axis"] = "right" if i % 2 == 1 else "left"
    else:
        for line in lines:
            line["axis"] = "left"

    if pattern == "boxplot":
        for line in lines:
            stats_list = []
            medians = []
            err_low = []
            err_high = []
            for base in line["y"]:
                n = int(rng.integers(12, 30))
                scale = max(0.05, float(abs(base) * rng.uniform(0.12, 0.35)))
                samples = rng.normal(loc=base, scale=scale, size=n)
                q1 = np.percentile(samples, 25)
                med = np.percentile(samples, 50)
                q3 = np.percentile(samples, 75)
                iqr = q3 - q1
                whis_lo = max(samples.min(), q1 - 1.5 * iqr)
                whis_hi = min(samples.max(), q3 + 1.5 * iqr)
                stats_list.append({
                    "med": float(med),
                    "q1": float(q1),
                    "q3": float(q3),
                    "whislo": float(whis_lo),
                    "whishi": float(whis_hi),
                    "fliers": [],
                })
                medians.append(med)
                err_low.append(max(0.0, med - whis_lo))
                err_high.append(max(0.0, whis_hi - med))
            line["box_stats"] = stats_list
            line["y"] = np.array(medians)
            line["err_low"] = np.array(err_low)
            line["err_high"] = np.array(err_high)
            line["err_style"] = "fixed"

    spec = {
        "pattern": pattern,
        "width": width,
        "height": height,
        "dpi": dpi,
        "log_x": log_x,
        "log_y": log_y,
        "lines": lines,
        "xlabel": xlabel if rng.random() < 0.7 else "",
        "ylabel": ylabel if rng.random() < 0.7 else "",
        "title": title if rng.random() < 0.4 else "",
        "grid": rng.random() < (min(0.9, max(0.1, visual_priors.get("grid_prob", 0.55))) if visual_priors else 0.55),
        "legend": rng.random() < 0.6,
        "legend_outside": rng.random() < 0.3,
        "legend_frame": rng.random() < 0.6,
        "background": bg_color,
        "font": rand_choice(rng, FONTS),
        "panel_label": rand_choice(rng, list("ABCDEFG")) if rng.random() < 0.35 else "",
        "annotations": {},
        "inset": pattern in {"inset"} or (pattern in {"log_pk", "log_multi"} and rng.random() < 0.15),
        "twin_y": pattern in {"dual_axis"},
        "pre_dose": pattern in {"pre_dose"},
        "chart_type": ("bar" if pattern == "grouped_bar"
                       else "box" if pattern == "boxplot"
                       else "scatter" if pattern == "scatter_only" else "line"),
        "box_xtick_labels": box_tick_labels,
        "bar_xtick_labels": bar_tick_labels,
        "despine": rng.random() < 0.4,
        "ticks_in": rng.random() < 0.5,
        "rotate_x": rng.random() < 0.35,
        "margins": {
            "l": int(rng.uniform(70, 110)),
            "r": int(rng.uniform(20, 50)),
            "t": int(rng.uniform(30, 60)),
            "b": int(rng.uniform(60, 90)),
        },
    }

    spec["annotations"]["significance"] = rng.random() < 0.2
    spec["annotations"]["arrow_label"] = rng.random() < 0.25
    spec["annotations"]["lloq"] = rng.random() < 0.25 and log_y
    spec["annotations"]["caption"] = rng.random() < 0.15
    spec["annotations"]["bql_line"] = pattern in {"bql_drop"} or (log_y and rng.random() < 0.2)
    spec["annotations"]["dose_lines"] = pattern in {"pre_dose", "dense_early"} and rng.random() < 0.4
    spec["annotations"]["letters"] = rng.random() < 0.25
    spec["annotations"]["week_text"] = rng.random() < 0.2
    spec["annotations"]["counts_under_x"] = (pattern in {"line_end_labels", "small_range"} and rng.random() < 0.5
                                            and not log_y)
    spec["annotations"]["line_end_labels"] = pattern in {"line_end_labels"} and rng.random() < 0.7

    return spec


def compute_axes_ranges(spec):
    xs = np.concatenate([line.get("x_draw", line["x"]) for line in spec["lines"]])
    ys = np.concatenate([line["y"] for line in spec["lines"]])
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    # Include precomputed error extents (e.g., boxplot whiskers) when available.
    for line in spec["lines"]:
        err_low = line.get("err_low")
        err_high = line.get("err_high")
        if err_low is None or err_high is None:
            continue
        if np.any(err_low) or np.any(err_high):
            y_min = min(y_min, float(np.min(line["y"] - err_low)))
            y_max = max(y_max, float(np.max(line["y"] + err_high)))
    if spec["log_y"]:
        y_min = max(y_min, 0.1)
        y_max = max(y_max, y_min * 1.2)
        y_min *= 0.6
        y_max *= 1.6
    else:
        pad = 0.1 * (y_max - y_min + 1)
        y_min -= pad
        y_max += pad
        if spec.get("chart_type") == "bar" and y_min > 0:
            y_min = 0.0
    if spec["log_x"]:
        x_min = max(x_min, 0.1)
        x_max = max(x_max, x_min * 1.2)
        x_min *= 0.8
        x_max *= 1.2
    else:
        pad = 0.05 * (x_max - x_min + 1)
        x_min -= pad
        x_max += pad
    return x_min, x_max, y_min, y_max


def _sample_ratio(rng, ratios, default_range=(0.01, 0.12)):
    if ratios:
        return float(rng.choice(ratios))
    return float(rng.uniform(default_range[0], default_range[1]))


def apply_errors(spec, rng, stats=None, visual_priors=None):
    y_min, y_max = spec["ranges"][2], spec["ranges"][3]
    plot_h = spec["height"] - spec["margins"]["t"] - spec["margins"]["b"]
    plot_h = max(plot_h, 1)

    error_presence = None
    if stats and stats.get("error_presence") and stats["error_presence"].get("total", 0) > 0:
        err = stats["error_presence"]
        total = err["total"]
        error_presence = {
            "any": err["any"] / total,
            "both": err["both"] / total,
            "top_only": err["top_only"] / total,
            "bottom_only": err["bottom_only"] / total,
        }

    cap_prob = visual_priors.get("cap_prob") if visual_priors else None

    for line in spec["lines"]:
        if line.get("err_style") == "fixed":
            if line.get("err_low") is None:
                line["err_low"] = np.zeros(len(line["x"]))
            if line.get("err_high") is None:
                line["err_high"] = np.zeros(len(line["x"]))
            continue
        if line.get("err_style") == "none":
            line["err_low"] = np.zeros(len(line["x"]))
            line["err_high"] = np.zeros(len(line["x"]))
            continue

        low = np.zeros(len(line["x"]))
        high = np.zeros(len(line["x"]))
        for i, (xi, yi) in enumerate(zip(line["x"], line["y"])):
            if error_presence:
                r = rng.random()
                if r > error_presence["any"]:
                    continue
                # decide if both or one-sided
                r2 = rng.random()
                if r2 < error_presence["both"]:
                    has_top, has_bottom = True, True
                elif r2 < error_presence["both"] + error_presence["top_only"]:
                    has_top, has_bottom = True, False
                else:
                    has_top, has_bottom = False, True
            else:
                base_prob = cap_prob if cap_prob is not None else 0.7
                has_top = rng.random() < base_prob
                has_bottom = rng.random() < base_prob

            top_ratio = _sample_ratio(rng, stats.get("error_top_ratios") if stats else None)
            bottom_ratio = _sample_ratio(rng, stats.get("error_bottom_ratios") if stats else None)
            top_px = top_ratio * plot_h if has_top else 0.0
            bottom_px = bottom_ratio * plot_h if has_bottom else 0.0

            if spec["log_y"]:
                log_span = math.log10(y_max) - math.log10(y_min)
                if has_top and top_px > 0:
                    factor = 10 ** (top_px / plot_h * log_span)
                    high[i] = max(0.0, yi * (factor - 1))
                if has_bottom and bottom_px > 0:
                    factor = 10 ** (bottom_px / plot_h * log_span)
                    low[i] = max(0.0, yi * (1 - 1 / factor))
            else:
                data_per_px = (y_max - y_min) / plot_h
                if has_top and top_px > 0:
                    high[i] = top_px * data_per_px
                if has_bottom and bottom_px > 0:
                    low[i] = bottom_px * data_per_px

            if spec.get("pattern") in {"rise_decay", "dense_early"} and not spec.get("log_y"):
                x_span = float(spec["ranges"][1] - spec["ranges"][0] + 1e-9)
                x_norm = (xi - spec["ranges"][0]) / x_span
                if x_norm <= 0.3:
                    boost = rng.uniform(1.2, 1.8)
                    high[i] *= boost
                    low[i] *= boost

        line["err_low"] = low
        line["err_high"] = high

    # Ensure bar chart ranges include error bars so they aren't clipped.
    if spec.get("chart_type") == "bar" and not spec.get("log_y"):
        x_min, x_max, y_min, y_max = spec["ranges"]
        for line in spec["lines"]:
            if line.get("err_low") is None or line.get("err_high") is None:
                continue
            if np.any(line["err_low"]) or np.any(line["err_high"]):
                y_min = min(y_min, float(np.min(line["y"] - line["err_low"])))
                y_max = max(y_max, float(np.max(line["y"] + line["err_high"])))
        pad = 0.05 * (y_max - y_min + 1)
        y_min -= pad
        y_max += pad
        if y_min > 0:
            y_min = 0.0
        spec["ranges"] = (x_min, x_max, y_min, y_max)


def compute_pixel_from_spec(spec, x_val, y_val):
    width = spec["width"]
    height = spec["height"]
    margins = spec["margins"]
    plot_w = width - margins["l"] - margins["r"]
    plot_h = height - margins["t"] - margins["b"]
    x_min, x_max, y_min, y_max = spec["ranges"]

    if spec["log_x"]:
        x_min_l = math.log10(x_min)
        x_max_l = math.log10(x_max)
        x_val_l = math.log10(max(x_val, x_min * 1e-6))
        fx = (x_val_l - x_min_l) / (x_max_l - x_min_l + 1e-9)
    else:
        fx = (x_val - x_min) / (x_max - x_min + 1e-9)
    if spec["log_y"]:
        y_min_l = math.log10(y_min)
        y_max_l = math.log10(y_max)
        y_val_l = math.log10(max(y_val, y_min * 1e-6))
        fy = (y_val_l - y_min_l) / (y_max_l - y_min_l + 1e-9)
    else:
        fy = (y_val - y_min) / (y_max - y_min + 1e-9)
    x_pix = margins["l"] + fx * plot_w
    y_pix = margins["t"] + (1 - fy) * plot_h
    return float(x_pix), float(y_pix)


def compute_pixel_mpl(ax, fig_height, x, y):
    x_disp, y_disp = ax.transData.transform((x, y))
    return float(x_disp), float(fig_height - y_disp)


def annotate_common(ax, rng, spec):
    if spec["panel_label"]:
        ax.text(0.02, 0.98, spec["panel_label"], transform=ax.transAxes,
                ha="left", va="top", fontsize=rng.uniform(12, 18), fontweight="bold")
    if spec["annotations"].get("significance"):
        line = spec["lines"][0]
        x = line["x"]
        y = line["y"]
        if len(x) >= 2:
            i = int(rng.integers(0, len(x) - 1))
            j = i + 1
            y_level = max(y[i], y[j]) + rng.uniform(2, 10)
            ax.plot([x[i], x[i], x[j], x[j]], [y_level - 1, y_level, y_level, y_level - 1],
                    color="black", linewidth=1)
            ax.text((x[i] + x[j]) / 2, y_level + rng.uniform(1, 3), "*",
                    ha="center", va="bottom", fontsize=rng.uniform(12, 16))
    if spec["annotations"].get("arrow_label"):
        line = spec["lines"][0]
        idx = int(rng.integers(0, len(line["x"])))
        ax.annotate("stimulated values", xy=(line["x"][idx], line["y"][idx]),
                    xytext=(line["x"][idx], line["y"][idx] + rng.uniform(10, 30)),
                    arrowprops=dict(arrowstyle="->", lw=1),
                    fontsize=rng.uniform(8, 11))
    if spec["annotations"].get("letters"):
        line = spec["lines"][0]
        letters = ["a", "b", "c"]
        for j, letter in enumerate(letters):
            idx = int(rng.integers(0, len(line["x"])))
            ax.text(line["x"][idx], line["y"][idx] - rng.uniform(0.5, 2.0),
                    letter, fontsize=9, color="black")
    if spec["annotations"].get("week_text"):
        ax.text(0.72, 0.85, f"Week {rng.integers(1, 8)}", transform=ax.transAxes, fontsize=10)
    if spec["annotations"].get("line_end_labels"):
        for line in spec["lines"]:
            x_last = line["x"][-1]
            y_last = line["y"][-1]
            ax.text(x_last, y_last, f" {line['label']}", fontsize=8, va="center")


def build_labels_from_pixels(lines, anchors):
    lines_out = []
    for line in lines:
        points_out = list(line["points"])
        points_out.extend(anchors)
        lines_out.append({
            "label": {"lineName": line["label"]},
            "points": points_out,
        })
    return lines_out


def render_mpl(rng, spec, out_images, out_labels, engine):
    fig = plt.figure(figsize=(spec["width"] / spec["dpi"], spec["height"] / spec["dpi"]), dpi=spec["dpi"])
    plt.rcParams["font.family"] = spec["font"]
    if engine == "seaborn" and sns is not None:
        sns.set_theme(style=rand_choice(rng, ["whitegrid", "ticks", "white"]))
    elif engine == "pandas":
        plt.style.use(rand_choice(rng, ["default", "classic", "seaborn-v0_8"]))
    else:
        plt.style.use(rand_choice(rng, ["default", "seaborn-v0_8", "classic"]))

    ax = fig.add_subplot(1, 1, 1)
    ax2 = None
    ax.set_facecolor(spec["background"])
    for spine in ax.spines.values():
        spine.set_linewidth(rng.uniform(0.8, 1.2))

    if spec["log_x"]:
        ax.set_xscale("log")
    if spec["log_y"]:
        ax.set_yscale("log")
    if spec.get("twin_y"):
        ax2 = ax.twinx()
        if spec["log_y"]:
            ax2.set_yscale("log")

    if spec["xlabel"]:
        ax.set_xlabel(spec["xlabel"])
    if spec["ylabel"]:
        ax.set_ylabel(spec["ylabel"])
    if spec["title"]:
        ax.set_title(spec["title"])

    if spec.get("ticks_in"):
        ax.tick_params(direction="in", top=True, right=not bool(ax2))
        if ax2:
            ax2.tick_params(direction="in", top=True, right=True)
    if spec.get("rotate_x"):
        rot = int(rand_choice(rng, [30, 45, 60]))
        ax.tick_params(axis="x", labelrotation=rot)
        if ax2:
            ax2.tick_params(axis="x", labelrotation=rot)
    if spec.get("despine"):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax2:
            ax2.spines["top"].set_visible(False)

    if spec.get("chart_type") != "line":
        engine = "mpl"

    if engine == "pandas" and pd is not None and spec.get("chart_type") == "line":
        df = pd.DataFrame({line["label"]: line["y"] for line in spec["lines"]}, index=spec["lines"][0]["x"])
        styles = []
        colors = []
        for line in spec["lines"]:
            if line["style"]["marker"]:
                styles.append(f"{line['style']['linestyle']}{line['style']['marker']}")
            else:
                styles.append(f"{line['style']['linestyle']}")
            colors.append(line["style"]["color"])
        df.plot(ax=ax, style=styles, color=colors, legend=False)
        for line in spec["lines"]:
            target_ax = ax2 if line.get("axis") == "right" and ax2 else ax
            target_ax.errorbar(
                line["x"], line["y"],
                yerr=np.vstack([line["err_low"], line["err_high"]]),
                fmt="none",
                ecolor=line["style"]["color"],
                elinewidth=line["style"]["elinewidth"],
                capsize=line["style"]["capsize"],
                capthick=line["style"]["capthick"],
            )
    elif engine == "seaborn" and sns is not None and spec.get("chart_type") == "line":
        for line in spec["lines"]:
            target_ax = ax2 if line.get("axis") == "right" and ax2 else ax
            drawstyle = line["style"].get("drawstyle")
            sns.lineplot(
                x=line["x"],
                y=line["y"],
                marker=line["style"]["marker"] if line["style"]["marker"] else None,
                linestyle=line["style"]["linestyle"],
                color=line["style"]["color"],
                markersize=line["style"]["markersize"],
                drawstyle=drawstyle if drawstyle else "default",
                ax=target_ax,
                label=line["label"],
            )
            err_kwargs = {}
            if drawstyle:
                err_kwargs["drawstyle"] = drawstyle
            target_ax.errorbar(
                line["x"], line["y"],
                yerr=np.vstack([line["err_low"], line["err_high"]]),
                fmt="none",
                ecolor=line["style"]["color"],
                elinewidth=line["style"]["elinewidth"],
                capsize=line["style"]["capsize"],
                capthick=line["style"]["capthick"],
                **err_kwargs,
            )
        if spec.get("despine"):
            sns.despine(ax=ax)
    else:
        for line in spec["lines"]:
            has_err = np.any(line["err_low"]) or np.any(line["err_high"])
            has_marker = line["style"]["marker"] is not None
            target_ax = ax2 if line.get("axis") == "right" and ax2 else ax
            x_draw = line.get("x_draw", line["x"])
            drawstyle = line["style"].get("drawstyle")
            if spec.get("chart_type") == "box":
                width = line.get("box_width", 0.5)
                stats_list = line.get("box_stats", [])
                if stats_list:
                    target_ax.bxp(
                        stats_list,
                        positions=x_draw,
                        widths=width,
                        showfliers=False,
                        patch_artist=True,
                        boxprops=dict(
                            facecolor=line["style"]["color"],
                            alpha=0.3,
                            edgecolor=line["style"]["color"],
                            linewidth=line["style"]["linewidth"],
                        ),
                        whiskerprops=dict(
                            color=line["style"]["color"],
                            linewidth=max(line["style"]["elinewidth"], 0.8),
                        ),
                        capprops=dict(
                            color=line["style"]["color"],
                            linewidth=max(line["style"]["elinewidth"], 0.8),
                        ),
                        medianprops=dict(color="black", linewidth=max(line["style"]["linewidth"], 1.2)),
                    )
                continue
            if spec.get("chart_type") == "bar":
                width = line.get("bar_width", 0.2)
                target_ax.bar(x_draw, line["y"], width=width,
                              color=line["style"]["color"], edgecolor=line["style"]["color"],
                              linewidth=line["style"]["linewidth"],
                              alpha=0.85, label=line["label"])
                if has_err:
                    target_ax.errorbar(
                        x_draw, line["y"],
                        yerr=np.vstack([line["err_low"], line["err_high"]]),
                        fmt="none",
                        ecolor=line["style"]["color"],
                        elinewidth=line["style"]["elinewidth"],
                        capsize=line["style"]["capsize"],
                        capthick=line["style"]["capthick"],
                    )
                continue

            if has_err:
                err_kwargs = {}
                if drawstyle:
                    err_kwargs["drawstyle"] = drawstyle
                if has_marker:
                    target_ax.errorbar(
                        x_draw, line["y"],
                        yerr=np.vstack([line["err_low"], line["err_high"]]),
                        fmt=line["style"]["marker"],
                        color=line["style"]["color"],
                        linestyle=line["style"]["linestyle"],
                        linewidth=line["style"]["linewidth"],
                        markersize=line["style"]["markersize"],
                        markerfacecolor=line["style"]["markerface"],
                        markeredgecolor=line["style"]["color"],
                        elinewidth=line["style"]["elinewidth"],
                        capsize=line["style"]["capsize"],
                        capthick=line["style"]["capthick"],
                        label=line["label"],
                        **err_kwargs,
                    )
                else:
                    target_ax.plot(
                        x_draw, line["y"],
                        color=line["style"]["color"],
                        linestyle=line["style"]["linestyle"],
                        linewidth=line["style"]["linewidth"],
                        drawstyle=drawstyle if drawstyle else "default",
                        label=line["label"],
                    )
                    target_ax.errorbar(
                        x_draw, line["y"],
                        yerr=np.vstack([line["err_low"], line["err_high"]]),
                        fmt="none",
                        ecolor=line["style"]["color"],
                        elinewidth=line["style"]["elinewidth"],
                        capsize=line["style"]["capsize"],
                        capthick=line["style"]["capthick"],
                        **err_kwargs,
                    )
            else:
                target_ax.plot(
                    x_draw, line["y"],
                    marker=line["style"]["marker"] if has_marker else None,
                    color=line["style"]["color"],
                    linestyle=line["style"]["linestyle"],
                    linewidth=line["style"]["linewidth"],
                    markersize=line["style"]["markersize"],
                    markerfacecolor=line["style"]["markerface"],
                    markeredgecolor=line["style"]["color"],
                    drawstyle=drawstyle if drawstyle else "default",
                    label=line["label"],
                )

    x_min, x_max, y_min, y_max = spec["ranges"]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    if ax2:
        ax2.set_ylim(y_min, y_max)

    if spec.get("chart_type") == "box" and spec.get("box_xtick_labels"):
        base_ticks = spec["lines"][0]["x"]
        ax.set_xticks(base_ticks)
        ax.set_xticklabels(spec["box_xtick_labels"])
    if spec.get("chart_type") == "bar" and spec.get("bar_xtick_labels"):
        base_ticks = spec["lines"][0]["x"]
        ax.set_xticks(base_ticks)
        ax.set_xticklabels(spec["bar_xtick_labels"])

    if spec["grid"]:
        ax.grid(
            True,
            linestyle=rand_choice(rng, [":", "--", "-"]),
            linewidth=float(rng.uniform(0.4, 0.9)),
            alpha=float(rng.uniform(0.25, 0.7)),
        )

    if spec["annotations"].get("lloq"):
        y_ref = y_min + 0.15 * (y_max - y_min)
        ax.axhline(y_ref, color="gray", linestyle="--", linewidth=1)
        ax.text(0.98, 0.02, "LLOQ", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    if spec["annotations"].get("bql_line"):
        y_ref = y_min + 0.1 * (y_max - y_min)
        ax.axhline(y_ref, color="gray", linestyle="--", linewidth=1, label="BQL level")

    annotate_common(ax, rng, spec)

    if spec["annotations"].get("dose_lines"):
        xs = np.quantile(spec["lines"][0]["x"], [0.1, 0.3, 0.5, 0.7])
        for xv in xs:
            ax.axvline(xv, color="lightgray", linestyle="--", linewidth=0.8)

    if spec["annotations"].get("counts_under_x") and not spec["log_y"]:
        counts_a = rng.integers(20, 80, size=len(spec["lines"][0]["x"]))
        counts_b = rng.integers(20, 80, size=len(spec["lines"][0]["x"]))
        y_text = y_min - 0.08 * (y_max - y_min)
        for xv, ca, cb in zip(spec["lines"][0]["x"], counts_a, counts_b):
            ax.text(xv, y_text, f"N = {ca}", ha="center", va="top", fontsize=8)
            ax.text(xv, y_text - 0.04 * (y_max - y_min), f"N = {cb}", ha="center", va="top", fontsize=8)

    if spec["annotations"].get("caption"):
        fig.text(0.02, 0.02, "Mean serum concentration versus time (± SD).", fontsize=8)

    if spec["legend"] and spec.get("chart_type") != "box":
        if spec.get("legend_outside"):
            fig.subplots_adjust(right=0.78)
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=spec.get("legend_frame", True))
        else:
            ax.legend(loc=rand_choice(rng, ["best", "upper right", "upper left", "lower right", "lower left"]),
                      frameon=spec.get("legend_frame", True))

    if spec.get("inset"):
        inset = fig.add_axes([0.55, 0.35, 0.35, 0.35])
        inset.set_facecolor("white")
        for line in spec["lines"]:
            inset.plot(line["x"], line["y"], color=line["style"]["color"], linewidth=1)
        if spec["log_y"]:
            inset.set_yscale("log")
        inset.set_xticks([])
        inset.set_yticks([])

    if spec.get("chart_type") in {"bar", "box"}:
        left = spec["margins"]["l"] / spec["width"]
        right = 1 - (spec["margins"]["r"] / spec["width"])
        bottom = spec["margins"]["b"] / spec["height"]
        top = 1 - (spec["margins"]["t"] / spec["height"])
        # Clamp to sane bounds
        left = min(max(left, 0.05), 0.45)
        right = min(max(right, 0.55), 0.95)
        bottom = min(max(bottom, 0.05), 0.45)
        top = min(max(top, 0.55), 0.95)
        if left >= right:
            left, right = 0.1, 0.9
        if bottom >= top:
            bottom, top = 0.1, 0.9
        fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    else:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="(?i).*tight[_ ]layout.*",
                    category=UserWarning,
                )
                fig.tight_layout()
        except Exception:
            pass
    fig.canvas.draw()
    fig_w, fig_h = fig.canvas.get_width_height()

    lines_out = []
    for line in spec["lines"]:
        points = []
        for xi, x_draw, yi, lo, hi in zip(line["x"], line.get("x_draw", line["x"]), line["y"],
                                          line["err_low"], line["err_high"]):
            target_ax = ax2 if line.get("axis") == "right" and ax2 else ax
            x_pix, y_pix = compute_pixel_mpl(target_ax, fig_h, x_draw, yi)
            if hi > 0:
                _, y_upper = compute_pixel_mpl(target_ax, fig_h, x_draw, yi + hi)
                top_dist = max(0.0, y_pix - y_upper)
            else:
                top_dist = 0.0
            if lo > 0:
                _, y_lower = compute_pixel_mpl(target_ax, fig_h, x_draw, yi - lo)
                bottom_dist = max(0.0, y_lower - y_pix)
            else:
                bottom_dist = 0.0
            points.append({
                "x": x_pix,
                "y": y_pix,
                "label": "",
                "topBarPixelDistance": top_dist,
                "bottomBarPixelDistance": bottom_dist,
                "deviationPixelDistance": max(top_dist, bottom_dist),
            })
        lines_out.append({"label": line["label"], "points": points})

    anchors = [
        {"x": compute_pixel_mpl(ax, fig_h, x_min, y_min)[0], "y": compute_pixel_mpl(ax, fig_h, x_min, y_min)[1], "label": "ymin",
         "topBarPixelDistance": 0, "bottomBarPixelDistance": 0, "deviationPixelDistance": 0},
        {"x": compute_pixel_mpl(ax, fig_h, x_min, y_max)[0], "y": compute_pixel_mpl(ax, fig_h, x_min, y_max)[1], "label": "ymax",
         "topBarPixelDistance": 0, "bottomBarPixelDistance": 0, "deviationPixelDistance": 0},
        {"x": compute_pixel_mpl(ax, fig_h, x_min, y_min)[0], "y": compute_pixel_mpl(ax, fig_h, x_min, y_min)[1], "label": "xmin",
         "topBarPixelDistance": 0, "bottomBarPixelDistance": 0, "deviationPixelDistance": 0},
        {"x": compute_pixel_mpl(ax, fig_h, x_max, y_min)[0], "y": compute_pixel_mpl(ax, fig_h, x_max, y_min)[1], "label": "xmax",
         "topBarPixelDistance": 0, "bottomBarPixelDistance": 0, "deviationPixelDistance": 0},
    ]

    image_name = f"synth_{uuid.uuid4().hex}.png"
    label_name = image_name.replace(".png", ".json")
    try:
        fig.savefig(out_images / image_name, dpi=spec["dpi"])
    except Exception as exc:
        plt.close(fig)
        return None
    plt.close(fig)

    with (out_labels / label_name).open("w") as f:
        json.dump(build_labels_from_pixels(lines_out, anchors), f, indent=2)

    return image_name


def render_plotly(rng, spec, out_images, out_labels):
    if go is None:
        raise RuntimeError("plotly not available")
    fig = go.Figure()
    for line in spec["lines"]:
        has_marker = line["style"]["marker"] is not None
        line_style = dict(
            color=line["style"]["color"],
            dash=map_linestyle_plotly(line["style"]["linestyle"]),
        )
        if line["style"].get("drawstyle") == "steps-post":
            line_style["shape"] = "hv"
        scatter_kwargs = dict(
            x=line["x"],
            y=line["y"],
            mode="lines+markers" if has_marker else "lines",
            name=line["label"],
            line=line_style,
            error_y=dict(
                type="data",
                array=line["err_high"],
                arrayminus=line["err_low"],
                visible=True,
            ),
        )
        if has_marker:
            scatter_kwargs["marker"] = dict(
                symbol=map_marker_plotly(line["style"]["marker"]),
                size=line["style"]["markersize"],
                color=line["style"]["color"],
            )
        fig.add_trace(go.Scatter(**scatter_kwargs))
    x_min, x_max, y_min, y_max = spec["ranges"]
    layout = dict(
        width=spec["width"],
        height=spec["height"],
        margin=dict(l=spec["margins"]["l"], r=spec["margins"]["r"], t=spec["margins"]["t"], b=spec["margins"]["b"]),
        xaxis=dict(type="log" if spec["log_x"] else "linear",
                   range=[math.log10(x_min), math.log10(x_max)] if spec["log_x"] else [x_min, x_max]),
        yaxis=dict(type="log" if spec["log_y"] else "linear",
                   range=[math.log10(y_min), math.log10(y_max)] if spec["log_y"] else [y_min, y_max]),
        plot_bgcolor=spec["background"],
        paper_bgcolor="white",
        showlegend=spec["legend"],
        font=dict(family=spec["font"], size=12),
    )
    if spec["xlabel"]:
        layout["xaxis"]["title"] = spec["xlabel"]
    if spec["ylabel"]:
        layout["yaxis"]["title"] = spec["ylabel"]
    if spec["title"]:
        layout["title"] = spec["title"]
    fig.update_layout(layout)

    image_name = f"synth_{uuid.uuid4().hex}.png"
    label_name = image_name.replace(".png", ".json")
    fig.write_image(out_images / image_name)

    lines_out = []
    for line in spec["lines"]:
        points = []
        for xi, yi, lo, hi in zip(line["x"], line["y"], line["err_low"], line["err_high"]):
            x_pix, y_pix = compute_pixel_from_spec(spec, xi, yi)
            if hi > 0:
                _, y_upper = compute_pixel_from_spec(spec, xi, yi + hi)
                top_dist = max(0.0, y_pix - y_upper)
            else:
                top_dist = 0.0
            if lo > 0:
                _, y_lower = compute_pixel_from_spec(spec, xi, yi - lo)
                bottom_dist = max(0.0, y_lower - y_pix)
            else:
                bottom_dist = 0.0
            points.append({
                "x": x_pix,
                "y": y_pix,
                "label": "",
                "topBarPixelDistance": top_dist,
                "bottomBarPixelDistance": bottom_dist,
                "deviationPixelDistance": max(top_dist, bottom_dist),
            })
        lines_out.append({"label": line["label"], "points": points})

    anchors = [
        {"x": compute_pixel_from_spec(spec, x_min, y_min)[0], "y": compute_pixel_from_spec(spec, x_min, y_min)[1], "label": "ymin",
         "topBarPixelDistance": 0, "bottomBarPixelDistance": 0, "deviationPixelDistance": 0},
        {"x": compute_pixel_from_spec(spec, x_min, y_max)[0], "y": compute_pixel_from_spec(spec, x_min, y_max)[1], "label": "ymax",
         "topBarPixelDistance": 0, "bottomBarPixelDistance": 0, "deviationPixelDistance": 0},
        {"x": compute_pixel_from_spec(spec, x_min, y_min)[0], "y": compute_pixel_from_spec(spec, x_min, y_min)[1], "label": "xmin",
         "topBarPixelDistance": 0, "bottomBarPixelDistance": 0, "deviationPixelDistance": 0},
        {"x": compute_pixel_from_spec(spec, x_max, y_min)[0], "y": compute_pixel_from_spec(spec, x_max, y_min)[1], "label": "xmax",
         "topBarPixelDistance": 0, "bottomBarPixelDistance": 0, "deviationPixelDistance": 0},
    ]

    with (out_labels / label_name).open("w") as f:
        json.dump(build_labels_from_pixels(lines_out, anchors), f, indent=2)

    return image_name


def render_bokeh(rng, spec, out_images, out_labels):
    if figure is None or export_png is None:
        raise RuntimeError("bokeh not available")
    p = figure(
        width=spec["width"],
        height=spec["height"],
        x_axis_type="log" if spec["log_x"] else "linear",
        y_axis_type="log" if spec["log_y"] else "linear",
        background_fill_color=spec["background"],
        toolbar_location=None,
    )
    p.min_border_left = spec["margins"]["l"]
    p.min_border_right = spec["margins"]["r"]
    p.min_border_top = spec["margins"]["t"]
    p.min_border_bottom = spec["margins"]["b"]

    x_min, x_max, y_min, y_max = spec["ranges"]
    p.x_range.start = x_min
    p.x_range.end = x_max
    p.y_range.start = y_min
    p.y_range.end = y_max

    if spec["xlabel"]:
        p.xaxis.axis_label = spec["xlabel"]
    if spec["ylabel"]:
        p.yaxis.axis_label = spec["ylabel"]

    for line in spec["lines"]:
        if line["style"].get("drawstyle") == "steps-post":
            p.step(line["x"], line["y"], mode="after",
                   line_color=line["style"]["color"],
                   line_dash=map_linestyle_bokeh(line["style"]["linestyle"]))
        else:
            p.line(line["x"], line["y"], line_color=line["style"]["color"],
                   line_dash=map_linestyle_bokeh(line["style"]["linestyle"]))
        if line["style"]["marker"]:
            fill_color = line["style"]["markerface"]
            fill_alpha = 1.0
            if fill_color == "none":
                fill_color = None
                fill_alpha = 0.0
            p.scatter(
                line["x"],
                line["y"],
                marker=map_marker_bokeh(line["style"]["marker"]),
                size=line["style"]["markersize"],
                line_color=line["style"]["color"],
                fill_color=fill_color,
                fill_alpha=fill_alpha,
            )
        if np.any(line["err_low"]) or np.any(line["err_high"]):
            source = ColumnDataSource(data=dict(
                x=line["x"],
                upper=line["y"] + line["err_high"],
                lower=line["y"] - line["err_low"],
            ))
            whisker = Whisker(base="x", upper="upper", lower="lower", source=source,
                              line_color=line["style"]["color"])
            p.add_layout(whisker)

    image_name = f"synth_{uuid.uuid4().hex}.png"
    label_name = image_name.replace(".png", ".json")
    export_png(p, filename=str(out_images / image_name))

    lines_out = []
    for line in spec["lines"]:
        points = []
        for xi, yi, lo, hi in zip(line["x"], line["y"], line["err_low"], line["err_high"]):
            x_pix, y_pix = compute_pixel_from_spec(spec, xi, yi)
            if hi > 0:
                _, y_upper = compute_pixel_from_spec(spec, xi, yi + hi)
                top_dist = max(0.0, y_pix - y_upper)
            else:
                top_dist = 0.0
            if lo > 0:
                _, y_lower = compute_pixel_from_spec(spec, xi, yi - lo)
                bottom_dist = max(0.0, y_lower - y_pix)
            else:
                bottom_dist = 0.0
            points.append({
                "x": x_pix,
                "y": y_pix,
                "label": "",
                "topBarPixelDistance": top_dist,
                "bottomBarPixelDistance": bottom_dist,
                "deviationPixelDistance": max(top_dist, bottom_dist),
            })
        lines_out.append({"label": line["label"], "points": points})

    anchors = [
        {"x": compute_pixel_from_spec(spec, x_min, y_min)[0], "y": compute_pixel_from_spec(spec, x_min, y_min)[1], "label": "ymin",
         "topBarPixelDistance": 0, "bottomBarPixelDistance": 0, "deviationPixelDistance": 0},
        {"x": compute_pixel_from_spec(spec, x_min, y_max)[0], "y": compute_pixel_from_spec(spec, x_min, y_max)[1], "label": "ymax",
         "topBarPixelDistance": 0, "bottomBarPixelDistance": 0, "deviationPixelDistance": 0},
        {"x": compute_pixel_from_spec(spec, x_min, y_min)[0], "y": compute_pixel_from_spec(spec, x_min, y_min)[1], "label": "xmin",
         "topBarPixelDistance": 0, "bottomBarPixelDistance": 0, "deviationPixelDistance": 0},
        {"x": compute_pixel_from_spec(spec, x_max, y_min)[0], "y": compute_pixel_from_spec(spec, x_max, y_min)[1], "label": "xmax",
         "topBarPixelDistance": 0, "bottomBarPixelDistance": 0, "deviationPixelDistance": 0},
    ]

    with (out_labels / label_name).open("w") as f:
        json.dump(build_labels_from_pixels(lines_out, anchors), f, indent=2)

    return image_name


def main():
    parser = argparse.ArgumentParser(description="Synthetic error-bar dataset generator.")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--images", type=Path, default=Path("synthetic_data/images"))
    parser.add_argument("--labels", type=Path, default=Path("synthetic_data/labels"))
    parser.add_argument("--engine", type=str, default="auto", choices=["auto", "mpl", "seaborn", "pandas", "plotly", "bokeh"])
    parser.add_argument("--engine-weights", type=str, default=None,
                        help="Override engine weights, e.g. mpl:0.4,seaborn:0.2,pandas:0.2,plotly:0.1,bokeh:0.1")
    parser.add_argument("--use-existing-sizes", action="store_true")
    parser.add_argument("--learn-from-labels", action="store_true",
                        help="Compute sampling stats from existing label files (excluding synth_ by default).")
    parser.add_argument("--stats", type=Path, default=Path("synthetic_data/learned_stats.json"),
                        help="Path to read/write stats JSON for learned distributions.")
    parser.add_argument("--visual-stats", type=Path, default=None,
                        help="Path to visual stats JSON generated by analyze_images.py.")
    parser.add_argument("--exclude-prefix", type=str, default=DEFAULT_EXCLUDE_PREFIX,
                        help="Filename prefix to exclude when learning stats (default: synth_).")
    parser.add_argument("--size-sample", type=int, default=500,
                        help="Max number of images to sample for size distribution.")
    parser.add_argument("--manifest", type=Path, default=Path("synthetic_data/manifest.jsonl"))
    parser.add_argument("--min-bar", type=int, default=0,
                        help="Minimum number of grouped bar charts to generate.")
    parser.add_argument("--min-box", type=int, default=0,
                        help="Minimum number of box plots to generate.")
    args = parser.parse_args()

    args.images.mkdir(parents=True, exist_ok=True)
    args.labels.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    stats = None
    if args.stats and args.stats.exists():
        try:
            stats = json.loads(args.stats.read_text())
        except Exception:
            stats = None
    if stats is None and args.learn_from_labels:
        stats = compute_label_stats(args.labels, exclude_prefix=args.exclude_prefix)
        if stats is None:
            stats = None
        if stats is not None and args.use_existing_sizes:
            size_counts = load_size_distribution_with_filter(args.images, max_samples=args.size_sample,
                                                             exclude_prefix=args.exclude_prefix)
            if size_counts:
                stats["size_counts"] = {f"{k[0]}x{k[1]}": int(v) for k, v in size_counts.items()}
        if stats is not None and args.stats:
            args.stats.write_text(json.dumps(stats))

    if args.use_existing_sizes:
        if stats and stats.get("size_counts"):
            size_counts = Counter({tuple(map(int, k.split("x"))): v for k, v in stats["size_counts"].items()})
        else:
            size_counts = load_size_distribution_with_filter(args.images, max_samples=args.size_sample,
                                                             exclude_prefix=args.exclude_prefix)
    else:
        size_counts = None

    visual_priors = {}
    if args.visual_stats:
        visual_priors = derive_visual_priors(load_visual_stats(args.visual_stats))

    engines = available_engines()
    if args.engine == "auto" and not args.engine_weights:
        engines = [e for e in engines if e not in AUTO_SKIP_ENGINES]
    weights = {k: v for k, v in ENGINE_WEIGHTS.items() if k in engines}
    if args.engine_weights:
        custom = {}
        for part in args.engine_weights.split(","):
            part = part.strip()
            if not part:
                continue
            name, value = part.split(":", 1)
            name = name.strip()
            if name in engines:
                try:
                    custom[name] = float(value)
                except ValueError:
                    continue
        if custom:
            weights = custom
    if not weights:
        weights = {"mpl": 1.0}

    manifest_handle = None
    if args.manifest:
        manifest_handle = args.manifest.open("w")

    generated = 0
    attempts = 0
    max_attempts = args.count * 2
    pattern_counts = Counter()
    targets = {}
    if args.min_bar > 0:
        targets["grouped_bar"] = args.min_bar
    if args.min_box > 0:
        targets["boxplot"] = args.min_box

    while generated < args.count and attempts < max_attempts:
        attempts += 1
        forced_pattern = None
        if targets:
            remaining = args.count - generated
            needs = {p: targets[p] - pattern_counts.get(p, 0) for p in targets}
            needs = {p: n for p, n in needs.items() if n > 0}
            if needs:
                need_total = sum(needs.values())
                if remaining <= need_total:
                    forced_pattern = weighted_choice(rng, needs)
                else:
                    prob_force = min(1.0, need_total / float(remaining))
                    if rng.random() < prob_force:
                        forced_pattern = weighted_choice(rng, needs)

        spec = generate_spec(
            rng,
            size_counts,
            stats,
            visual_priors=visual_priors,
            pattern_override=forced_pattern,
        )
        spec["ranges"] = compute_axes_ranges(spec)
        apply_errors(spec, rng, stats, visual_priors=visual_priors)
        # Force Matplotlib for bar, box, or scatter charts to ensure correct rendering
        if spec.get("chart_type") in {"bar", "box", "scatter"}:
            engine = "mpl"
        elif args.engine == "auto":
            engine = weighted_choice(rng, weights)
        else:
            engine = args.engine

        try:
            if engine == "plotly":
                image_name = render_plotly(rng, spec, args.images, args.labels)
            elif engine == "bokeh":
                image_name = render_bokeh(rng, spec, args.images, args.labels)
            else:
                image_name = render_mpl(rng, spec, args.images, args.labels, engine)
        except Exception as exc:
            if args.engine != "auto":
                print(f"Engine {engine} failed ({exc}); falling back to mpl.")
            engine = "mpl"
            image_name = render_mpl(rng, spec, args.images, args.labels, engine)

        if not image_name:
            continue

        generated += 1
        pattern_counts[spec["pattern"]] += 1

        if manifest_handle:
            manifest_handle.write(json.dumps({
                "image": image_name,
                "engine": engine,
                "pattern": spec["pattern"],
                "width": spec["width"],
                "height": spec["height"],
                "log_x": spec["log_x"],
                "log_y": spec["log_y"],
                "num_lines": len(spec["lines"]),
            }) + "\n")

    if manifest_handle:
        manifest_handle.close()

    print(f"Generated {generated} images into {args.images} and labels into {args.labels}")


if __name__ == "__main__":
    main()

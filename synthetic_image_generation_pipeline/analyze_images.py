#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def quantize_color(rgb, bins=16):
    step = 256 // bins
    return tuple(int(c // step) for c in rgb)


def dequantize_color(q, bins=16):
    step = 256 // bins
    return tuple(int(c * step + step / 2) for c in q)


def downscale(img, max_side=512):
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)


def compute_sobel_edges(gray):
    # gray: float32 [H, W]
    g = gray
    gx = (
        -1 * g[:-2, :-2] + 1 * g[:-2, 2:]
        -2 * g[1:-1, :-2] + 2 * g[1:-1, 2:]
        -1 * g[2:, :-2] + 1 * g[2:, 2:]
    )
    gy = (
        1 * g[:-2, :-2] + 2 * g[:-2, 1:-1] + 1 * g[:-2, 2:]
        -1 * g[2:, :-2] - 2 * g[2:, 1:-1] - 1 * g[2:, 2:]
    )
    mag = np.sqrt(gx * gx + gy * gy)
    # pad to original size
    mag = np.pad(mag, ((1, 1), (1, 1)), mode="edge")
    return mag


def connected_components(binary):
    # Simple 4-neighborhood connected components. Returns list of sizes.
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    sizes = []
    for y in range(h):
        for x in range(w):
            if not binary[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            size = 0
            while stack:
                cy, cx = stack.pop()
                size += 1
                if cy > 0 and binary[cy - 1, cx] and not visited[cy - 1, cx]:
                    visited[cy - 1, cx] = True
                    stack.append((cy - 1, cx))
                if cy + 1 < h and binary[cy + 1, cx] and not visited[cy + 1, cx]:
                    visited[cy + 1, cx] = True
                    stack.append((cy + 1, cx))
                if cx > 0 and binary[cy, cx - 1] and not visited[cy, cx - 1]:
                    visited[cy, cx - 1] = True
                    stack.append((cy, cx - 1))
                if cx + 1 < w and binary[cy, cx + 1] and not visited[cy, cx + 1]:
                    visited[cy, cx + 1] = True
                    stack.append((cy, cx + 1))
            sizes.append(size)
    return sizes


def analyze_image(path, max_side=512, color_bins=16):
    img = Image.open(path).convert("RGB")
    img = downscale(img, max_side=max_side)
    w, h = img.size
    arr = np.array(img)

    # Background color from border pixels
    border = np.concatenate([
        arr[0, :, :],
        arr[-1, :, :],
        arr[:, 0, :],
        arr[:, -1, :],
    ], axis=0)
    q_border = [quantize_color(tuple(px), bins=color_bins) for px in border]
    bg_q = Counter(q_border).most_common(1)[0][0]
    bg = np.array(dequantize_color(bg_q, bins=color_bins))

    # Foreground mask (non-background)
    diff = np.abs(arr.astype(np.int16) - bg.astype(np.int16)).sum(axis=2)
    fg = diff > 30
    fg_count = int(fg.sum())

    # Dominant colors among foreground (favor saturated colors for line series)
    if fg_count > 0:
        fg_pixels = arr[fg]
        # compute saturation proxy
        max_c = fg_pixels.max(axis=1).astype(np.float32) + 1e-6
        min_c = fg_pixels.min(axis=1).astype(np.float32)
        sat = (max_c - min_c) / max_c
        saturated = sat > 0.2
        if saturated.any():
            use_pixels = fg_pixels[saturated]
        else:
            use_pixels = fg_pixels
        q_fg = [quantize_color(tuple(px), bins=color_bins) for px in use_pixels]
        fg_counts = Counter(q_fg)
        # dominant clusters (>=0.5% of foreground)
        threshold = max(10, int(0.005 * len(use_pixels)))
        dominant = [c for c, v in fg_counts.items() if v >= threshold]
        dominant_colors = len(dominant)
    else:
        fg_counts = Counter()
        dominant_colors = 0

    # Edge map
    gray = arr.mean(axis=2).astype(np.float32)
    edges = compute_sobel_edges(gray)
    thr = edges.mean() + edges.std()
    edge_bin = edges > thr
    edge_density = float(edge_bin.mean())

    # Grid detection
    row_sum = edge_bin.sum(axis=1)
    col_sum = edge_bin.sum(axis=0)
    row_ratio = row_sum / float(w)
    col_ratio = col_sum / float(h)
    # grid lines tend to span a good portion of the plot area
    strong_rows = np.where(row_ratio > 0.3)[0]
    strong_cols = np.where(col_ratio > 0.3)[0]
    # ignore very top/bottom and very left/right rows (axes/labels)
    def _filter_band(idxs, size, margin=0.06):
        lo = int(size * margin)
        hi = int(size * (1 - margin))
        return idxs[(idxs >= lo) & (idxs <= hi)]
    strong_rows = _filter_band(strong_rows, h)
    strong_cols = _filter_band(strong_cols, w)
    grid_present = int(len(strong_rows) >= 3 or len(strong_cols) >= 3)

    # Log-like axis detection by irregular spacing of strong rows/cols
    log_y_score = 0.0
    if len(strong_rows) >= 4:
        gaps = np.diff(np.sort(strong_rows))
        if gaps.size > 0 and gaps.mean() > 0:
            log_y_score = float(gaps.std() / gaps.mean())
    log_x_score = 0.0
    if len(strong_cols) >= 4:
        gaps = np.diff(np.sort(strong_cols))
        if gaps.size > 0 and gaps.mean() > 0:
            log_x_score = float(gaps.std() / gaps.mean())

    # Marker presence via small connected components
    comp_sizes = connected_components(edge_bin.astype(np.uint8))
    small_components = [s for s in comp_sizes if 5 <= s <= 60]
    marker_score = float(sum(small_components)) / float(edge_bin.sum() + 1)

    # Error bar cap indicator: count short horizontal edge runs
    cap_count = 0
    cap_len_total = 0
    for row in edge_bin:
        run = 0
        for val in row:
            if val:
                run += 1
            else:
                if 3 <= run <= 15:
                    cap_count += 1
                    cap_len_total += run
                run = 0
        if 3 <= run <= 15:
            cap_count += 1
            cap_len_total += run
    cap_score = float(cap_len_total) / float(w * h)

    return {
        "width": w,
        "height": h,
        "bg_color": bg_q,
        "dominant_colors": dominant_colors,
        "fg_ratio": float(fg_count) / float(w * h),
        "edge_density": edge_density,
        "grid_present": grid_present,
        "log_y_score": log_y_score,
        "log_x_score": log_x_score,
        "marker_score": marker_score,
        "cap_score": cap_score,
        "fg_color_hist": {f"{c[0]}-{c[1]}-{c[2]}": int(v) for c, v in fg_counts.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze plot images for visual characteristics.")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-prefix", type=str, default=None)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--color-bins", type=int, default=16)
    args = parser.parse_args()

    if Image is None:
        raise RuntimeError("Pillow is required (pip install pillow).")

    stats = {
        "image_count": 0,
        "sizes": Counter(),
        "bg_colors": Counter(),
        "dominant_colors": [],
        "fg_ratios": [],
        "edge_density": [],
        "grid_present": [],
        "log_y_scores": [],
        "log_x_scores": [],
        "marker_scores": [],
        "cap_scores": [],
    }

    for path in sorted(args.images.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        if args.exclude_prefix and path.name.startswith(args.exclude_prefix):
            continue
        # Read each image one by one
        info = analyze_image(path, max_side=args.max_side, color_bins=args.color_bins)
        stats["image_count"] += 1
        stats["sizes"][f"{info['width']}x{info['height']}"] += 1
        stats["bg_colors"][f"{info['bg_color'][0]}-{info['bg_color'][1]}-{info['bg_color'][2]}"] += 1
        stats["dominant_colors"].append(info["dominant_colors"])
        stats["fg_ratios"].append(info["fg_ratio"])
        stats["edge_density"].append(info["edge_density"])
        stats["grid_present"].append(info["grid_present"])
        stats["log_y_scores"].append(info["log_y_score"])
        stats["log_x_scores"].append(info["log_x_score"])
        stats["marker_scores"].append(info["marker_score"])
        stats["cap_scores"].append(info["cap_score"])

    # Convert counters to plain dicts
    stats["sizes"] = dict(stats["sizes"])
    stats["bg_colors"] = dict(stats["bg_colors"])

    args.output.write_text(json.dumps(stats, indent=2))
    print(f"Analyzed {stats['image_count']} images. Wrote stats to {args.output}")


if __name__ == "__main__":
    main()

"""Hàm vẽ dùng chung cho các notebook: heatmap (attention, positional encoding), đường loss,
embedding 2 chiều, phân phối xác suất. Gom về một chỗ để các notebook không phải chép lại code vẽ.

Quy ước màu:

* Đại lượng không âm (trọng số attention): thang một màu xanh – nhạt là 0, đậm là lớn.
* Đại lượng có dấu (giá trị positional encoding): thang phân kỳ đỏ (âm) – xám (0) – xanh (dương).
* Các nhóm / các đường khác nhau: bảng màu phân loại theo thứ tự cố định (xanh dương, cam, xanh
  ngọc...). Biểu đồ phân tán chỉ nên có tối đa 3 nhóm để màu luôn dễ phân biệt, kể cả với người
  mù màu; mỗi nhóm còn có hình marker riêng và mỗi điểm có nhãn chữ.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
SURFACE, INK, INK_SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "vietformer_blues", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
).with_extremes(bad=GRID)
DIVERGING = LinearSegmentedColormap.from_list(
    "vietformer_red_blue", ["#8f2626", "#e34948", "#f4bcb9", "#f0efec", "#b7d3f6", "#2a78d6", "#104281"]
).with_extremes(bad=GRID)


def set_style() -> None:
    """Áp dụng phong cách vẽ chung: nền sáng, trục và lưới mảnh, bảng màu cố định."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 100,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titlecolor": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=COLORS),
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "text.color": INK,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
        }
    )


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def plot_matrix(
    matrix,
    ax=None,
    title: str | None = None,
    xlabels=None,
    ylabels=None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cmap=None,
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = False,
    fmt: str = ".2f",
    colorbar: bool = True,
    aspect: str = "auto",
):
    """Vẽ một ma trận thành heatmap. `annotate=True` ghi giá trị vào từng ô (dùng cho ma trận nhỏ).

    Ô có giá trị -inf (vị trí bị causal mask chặn) được tô xám nhạt và ghi "−∞".
    """
    m = np.ma.masked_invalid(_to_numpy(matrix))
    if ax is None:
        _, ax = plt.subplots(figsize=(min(1.5 + 0.55 * m.shape[1], 11), min(1.0 + 0.45 * m.shape[0], 8)))
    image = ax.imshow(m, cmap=cmap or SEQUENTIAL, vmin=vmin, vmax=vmax, aspect=aspect, interpolation="nearest")
    if xlabels is not None:
        ax.set_xticks(range(len(xlabels)), xlabels, rotation=90)
    if ylabels is not None:
        ax.set_yticks(range(len(ylabels)), ylabels)
    ax.tick_params(length=0)
    if annotate:
        raw = _to_numpy(matrix)
        for i in range(raw.shape[0]):
            for j in range(raw.shape[1]):
                value = raw[i, j]
                if np.isfinite(value):
                    r, g, b, _ = image.cmap(image.norm(value))
                    dark = 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5
                    ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=8, color="white" if dark else INK)
                else:
                    ax.text(j, i, "−∞", ha="center", va="center", fontsize=8, color=INK_SECONDARY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if colorbar:
        bar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=0)
    return ax


def plot_attention(weights, tokens, ax=None, title: str | None = None, annotate: bool = False, colorbar: bool = True):
    """Heatmap attention: hàng = token đang xét (query), cột = token được nhìn tới (key)."""
    return plot_matrix(
        weights, ax=ax, title=title, xlabels=tokens, ylabels=tokens, vmin=0.0, vmax=1.0,
        annotate=annotate, colorbar=colorbar, aspect="equal",
        xlabel="token được nhìn tới (key)", ylabel="token đang xét (query)",
    )


def plot_attention_grid(maps, tokens, title: str | None = None, cell_size: float = 2.7):
    """Lưới heatmap attention: mỗi hàng là một layer, mỗi cột là một head.

    `maps`: list (n_layer phần tử) các tensor (n_head, T, T) – ví dụ `[m[0] for m in model.attention_maps()]`.
    """
    layers = [_to_numpy(m) for m in maps]
    n_layer, n_head, T = len(layers), layers[0].shape[0], len(tokens)
    fig, axes = plt.subplots(n_layer, n_head, figsize=(cell_size * n_head + 0.8, cell_size * n_layer + 0.6), squeeze=False)
    for l, layer in enumerate(layers):
        for h in range(n_head):
            ax = axes[l][h]
            plot_matrix(layer[h], ax=ax, vmin=0.0, vmax=1.0, colorbar=False, aspect="equal")
            ax.set_title(f"layer {l} · head {h}", fontsize=9)
            ax.set_xticks(range(T), tokens if l == n_layer - 1 else [""] * T, rotation=90, fontsize=7)
            ax.set_yticks(range(T), tokens if h == 0 else [""] * T, fontsize=7)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_loss(history: dict, ax=None, title: str = "Loss trong quá trình huấn luyện"):
    """Đường train loss và val loss theo bước huấn luyện; đánh dấu điểm val loss thấp nhất."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["step"], history["train_loss"], color=COLORS[0], label="train")
    ax.plot(history["step"], history["val_loss"], color=COLORS[1], label="val")
    best_step, best = history["best_step"], history["best_val_loss"]
    ax.scatter([best_step], [best], s=50, color=COLORS[1], edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.annotate(f"val thấp nhất: {best:.2f} (bước {best_step})", (best_step, best), textcoords="offset points",
                xytext=(6, -16), fontsize=9, color=INK_SECONDARY)
    ax.set(xlabel="bước huấn luyện", ylabel="cross-entropy loss", title=title)
    ax.grid(axis="y")
    ax.legend()
    return ax


def plot_loss_comparison(histories: dict, key: str = "val_loss", ax=None, title: str | None = None):
    """So sánh một loại loss (mặc định val) của nhiều lần huấn luyện: {tên: history}."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for color, (name, history) in zip(COLORS, histories.items()):
        ax.plot(history["step"], history[key], color=color, label=name)
    ax.set(xlabel="bước huấn luyện", ylabel=key.replace("_", " "), title=title)
    ax.grid(axis="y")
    ax.legend()
    return ax


def _label_points(ax, points: np.ndarray, labels, fontsize: int = 9) -> None:
    """Ghi nhãn chữ cạnh từng điểm mà không để các nhãn đè lên nhau.

    Thử lần lượt bốn chỗ quanh điểm (phải-trên, phải-dưới, trái-trên, trái-dưới); nếu chỗ nào cũng đã có nhãn khác
    thì đẩy dần nhãn xuống dưới.
    """
    ax.autoscale_view()
    px_per_pt = ax.figure.dpi / 72
    char_w, line_h, gap = 0.62 * fontsize * px_per_pt, 1.2 * fontsize * px_per_pt, 3
    placed = []  # các hộp nhãn đã đặt (x0, y0, x1, y1), đơn vị pixel

    def is_free(x0, y0, width):
        return not any(x0 < bx1 + gap and x0 + width + gap > bx0 and y0 < by1 and y0 + line_h > by0
                       for bx0, by0, bx1, by1 in placed)

    for i in np.argsort(-points[:, 1]):  # đặt nhãn từ điểm cao nhất xuống
        x_px, y_px = ax.transData.transform(points[i])
        width = char_w * len(labels[i])
        candidates = [(6, 3), (6, -line_h), (-6 - width, 3), (-6 - width, -line_h)]
        dx, dy = next(((dx, dy) for dx, dy in candidates if is_free(x_px + dx, y_px + dy, width)), candidates[0])
        while not is_free(x_px + dx, y_px + dy, width) and dy > -8 * line_h:
            dy -= 0.5 * line_h
        placed.append((x_px + dx, y_px + dy, x_px + dx + width, y_px + dy + line_h))
        ax.annotate(labels[i], points[i], textcoords="offset points", xytext=(dx / px_per_pt, dy / px_per_pt),
                    fontsize=fontsize, color=INK)


def plot_embeddings_2d(points, labels, groups=None, ax=None, title: str | None = None):
    """Vẽ các điểm 2 chiều (sau PCA/t-SNE) kèm nhãn chữ; mỗi nhóm có một màu và một kiểu marker."""
    points = _to_numpy(points)
    groups = list(groups) if groups is not None else [""] * len(labels)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5.5))
    for k, name in enumerate(dict.fromkeys(groups)):  # các nhóm theo thứ tự xuất hiện
        idx = [i for i, g in enumerate(groups) if g == name]
        ax.scatter(points[idx, 0], points[idx, 1], s=70, color=COLORS[k], marker=MARKERS[k],
                   edgecolors=SURFACE, linewidths=1.5, label=name or None, zorder=2)
    _label_points(ax, points, list(labels))
    ax.set_xticks([])
    ax.set_yticks([])
    if any(groups):
        ax.legend(loc="best")
    if title:
        ax.set_title(title)
    return ax


def plot_token_probs(tokens, probs, ax=None, title: str | None = None):
    """Biểu đồ cột ngang: xác suất của các token ứng viên, token xác suất cao nhất ở trên cùng."""
    probs = _to_numpy(probs)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 0.32 * len(tokens) + 1))
    ax.barh(range(len(tokens)), probs, color=COLORS[0], height=0.7)
    ax.set_yticks(range(len(tokens)), tokens)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.tick_params(axis="y", length=0)
    ax.text(probs[0], 0, f" {probs[0]:.2f}", va="center", fontsize=8, color=INK_SECONDARY)
    ax.set(xlabel="xác suất", title=title)
    ax.grid(axis="x")
    return ax

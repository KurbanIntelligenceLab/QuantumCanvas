import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

mpl.rcParams.update({
    "font.family": "sans-serif",
    "mathtext.fontset": "dejavusans",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "axes.labelweight": "bold",
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})

MODEL_COLORS = {
    "FiLM-CNN": "#E69F00",
    "Geometry-only": "#56B4E9",
    "Multimodal-CA": "#009E73",
    "QSN-v2": "#D55E00",
    "Tabular MLP": "#CC79A7",
    "Vision-only ViT": "#0072B2",
}
MODELS = list(MODEL_COLORS)

ACCENT = "#D55E00"
MUTED = "#9aa0a6"

def panel_label(ax, letter, dx=-0.02, dy=1.06):
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes,
            fontweight="bold", fontsize=9, va="top", ha="right")

OOD_SPLITS = {
    "Held-out pairs\n(interpolation)": {
        "EA": [12.2, 3.6, 4.9, 0.4, 7.1, 5.5],
        "$E_{band}$": [-1.6, 0.1, 0.7, -3.7, 7.0, -1.1],
        "$E_{tot}$": [0.6, 4.4, 1.1, -1.5, 6.0, -8.1],
    },
    "By period\n(extrapolation)": {
        "EA": [24.2, 37.4, 61.2, 198.3, 34.9, -2.1],
        "$E_{band}$": [375.8, 772.4, 630.8, 920.3, 447.9, -1.2],
        "$E_{tot}$": [446.0, 1796.5, 882.4, 1366.3, 491.4, 9.4],
    },
    "By type\n(metal $\\rightarrow$ nonmetal)": {
        "EA": [177.3, 234.6, 250.9, 467.7, 307.2, 141.3],
        "$E_{tot}$": [1076.8, 1628.9, 519.7, 3486.6, 1377.8, 190.9],
    },
    "By electronegativity": {
        "EA": [113.4, 171.9, 86.8, 184.5, 135.9, 77.3],
        "$\\|\\boldsymbol{\\mu}\\|$": [49.8, 81.7, -0.2, 66.1, 39.0, -1.8],
        "$E_{tot}$": [305.6, 85.4, 43.4, 214.8, 285.6, 127.0],
    },
    "By bond distance": {
        "EA": [2.0, 0.8, 0.4, -4.2, 9.5, -2.4],
        "$E_{band}$": [264.7, 131.6, 63.3, 205.9, 213.9, 61.2],
        "$\\chi$": [6.8, 5.9, 0.1, 1.0, 3.6, 3.6],
        "$\\|\\boldsymbol{\\mu}\\|$": [189.4, 145.3, 40.9, 144.2, 214.8, 19.6],
        "$E_{tot}$": [245.0, 112.2, 74.2, 206.9, 221.0, 39.8],
    },
    "By period difference": {
        "EA": [-13.6, -5.6, -8.9, 2.3, -17.5, -4.3],
        "$E_{tot}$": [36.6, 20.5, 38.0, 60.1, 44.4, 36.7],
    },
}

OOD_AVG = [101.3, 171.7, 95.8, 256.0, 133.7, 32.0]

def fig_ood_gaps():

    keys = ["By type\n(metal $\\rightarrow$ nonmetal)", "By electronegativity",
            "By bond distance", "By period difference"]
    fig = plt.figure(figsize=(6.2, 4.2))
    grid = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.18,
                            left=0.10, right=0.985, top=0.93, bottom=0.15)
    bar_w = 0.13
    for i, key in enumerate(keys):
        ax = fig.add_subplot(grid[i // 2, i % 2])
        targets = OOD_SPLITS[key]
        names = list(targets)
        x = np.arange(len(names))
        for m, model in enumerate(MODELS):
            vals = [targets[t][m] for t in names]
            xs = x + (m - 2.5) * bar_w
            ax.bar(xs, vals, width=bar_w,
                   color=MODEL_COLORS[model], edgecolor="none")
            for xi, v in zip(xs, vals):
                if v >= 1000:
                    ax.text(xi, v * 1.25, f"{v / 1000:.1f}k", rotation=90,
                            ha="center", va="bottom", fontsize=5.8,
                            color="0.2")
        ax.set_yscale("symlog", linthresh=20)
        ax.set_ylim(-20, 30000)
        ax.set_yticks([-20, 0, 20, 100, 1000])
        ax.set_yticklabels(["$-$20", "0", "20", "$10^2$", "$10^3$"])
        ax.axhline(0.0, color="0.25", lw=0.7, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_title(f"({'abcd'[i]}) {key.replace(chr(10), ' ')}",
                     loc="left", pad=4, fontsize=7.6)
        ax.grid(axis="y", ls=":", lw=0.4, color="0.8", zorder=0)
        ax.set_axisbelow(True)
        if i % 2 == 0:
            ax.set_ylabel("Generalization gap (%)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m]) for m in MODELS]
    fig.legend(handles, MODELS, loc="lower center", ncol=6, frameon=False,
               handlelength=1.0, columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))

    fig.savefig("figureS2.pdf")
    plt.close(fig)

SHUFFLE_TARGETS = ["EA", "$E_{band}$", "$\\chi$", "$\\|\\boldsymbol{\\mu}\\|$",
                   "$E_g$", "$E_{HOMO}$", "$E_{LUMO}$", "$\\eta$", "IP",
                   "$E_{rep}$", "$E_{tot}$"]
SHUFFLE = np.array([

    [944.8,  461.2,  444.2, 1336.9, 1095.1, 0.0],
    [1583.5, 2161.6, 654.4, 3416.0, 2898.1, 0.0],
    [1170.1, 678.4, 1087.5, 1341.7, 1333.4, 0.0],
    [560.3,  524.5,    7.1,  657.6,  368.1, 0.0],
    [115.5,   25.0,   12.7,  381.0,  455.7, 0.0],
    [1138.0, 555.9,  801.3, 1337.1, 1144.3, 0.0],
    [949.8,  443.0,  471.3, 1122.0, 1030.3, 0.0],
    [63.0,    25.0,   12.7,  417.4,  455.7, 0.0],
    [1019.5, 717.2,  717.8, 1103.5, 1064.7, 0.0],
    [151.8,  138.1,   44.5,  937.5,  453.9, 0.0],
    [1994.6, 3175.6, 911.5, 5542.4, 3125.7, 0.0],
])

def fig_shuffle_heatmap():
    fig, ax = plt.subplots(figsize=(5.4, 3.3),
                           gridspec_kw=dict(left=0.07, right=0.95,
                                            top=0.91, bottom=0.03))

    data = SHUFFLE[:, :-1]

    model_cols = ["FiLM-CNN", "Geom.-only", "MM-CA", "QSN-v2", "Tab. MLP"]
    norm = colors.SymLogNorm(linthresh=30, vmin=0, vmax=6000)
    im = ax.imshow(data, cmap="YlOrRd", norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(model_cols)))
    ax.set_xticklabels(model_cols, fontsize=7.5)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(np.arange(len(SHUFFLE_TARGETS)))
    ax.set_yticklabels(SHUFFLE_TARGETS)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"+{v:,.0f}" if v else "0",
                    ha="center", va="center", fontsize=6.8,
                    color="white" if v > 800 else "black")

    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.045)
    cb.outline.set_visible(False)

    cb.ax.text(0.5, 0.5, "MAE increase under shuffle (%)", rotation=90,
               ha="center", va="center", transform=cb.ax.transAxes,
               fontsize=7.5, fontweight="bold", color="black",
               path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    fig.savefig("figureS1.pdf")
    plt.close(fig)

CHANNELS = [
    ("ch0: Orbital pop.", 0.216),
    ("ch1: Angular moment", 0.001),
    ("ch2: s/p field", 2.309),
    ("ch3: d/f field", 1.135),
    ("ch4: Dipole field", 15.935),
    ("ch5: Charge asym.", 0.152),
    ("ch6: Charge mag.", 9.666),
    ("ch7: Electron pop.", 2.632),
    ("ch8: Pos. charge", 0.802),
    ("ch9: Neg. charge", 1.886),
]

def fig_channel_importance():
    fig, ax = plt.subplots(figsize=(4.6, 3.3),
                           gridspec_kw=dict(left=0.045, right=0.97,
                                            top=0.975, bottom=0.13))
    labels = [c[0] for c in CHANNELS]
    vals = [c[1] for c in CHANNELS]
    y = np.arange(len(CHANNELS))[::-1]
    cols = [ACCENT if v > 5 else MUTED for v in vals]
    ax.barh(y, vals, height=0.46, color=cols, edgecolor="none")
    for lab, yi, v in zip(labels, y, vals):
        ax.text(v + 0.25, yi, f"+{v:.2f}%" if v >= 0.01 else "+0.00%",
                va="center", fontsize=7, color="black")
        ax.text(0.1, yi + 0.32, lab, ha="left", va="bottom",
                fontsize=6.8, fontweight="bold")
    ax.set_yticks([])
    ax.set_ylim(-0.6, len(CHANNELS) - 0.25)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Mean relative MAE increase when permuted (%)")
    ax.set_xlim(0, 19.5)
    ax.set_xticks([0, 5, 10, 15])
    ax.grid(axis="x", ls=":", lw=0.4, color="0.8", zorder=0)
    ax.set_axisbelow(True)
    fig.savefig("figureS3.pdf")
    plt.close(fig)

def fig_main_ood():
    fig = plt.figure(figsize=(6.0, 2.1))
    grid = fig.add_gridspec(1, 3, wspace=0.16, width_ratios=[1, 1, 1.15],
                            left=0.075, right=0.99, top=0.86, bottom=0.15)
    bar_w = 0.13

    panels = [("(a) Held-out pairs (interp.)", "Held-out pairs\n(interpolation)"),
              ("(b) By period (extrap.)", "By period\n(extrapolation)")]
    for k, (title, key) in enumerate(panels):
        ax = fig.add_subplot(grid[0, k])
        targets = OOD_SPLITS[key]
        names = list(targets)
        x = np.arange(len(names))
        for m, model in enumerate(MODELS):
            vals = [targets[t][m] for t in names]
            xs = x + (m - 2.5) * bar_w
            ax.bar(xs, vals, width=bar_w,
                   color=MODEL_COLORS[model], edgecolor="none")
            for xi, v in zip(xs, vals):
                if v >= 1000:
                    ax.text(xi, v * 1.25, f"{v / 1000:.1f}k", rotation=90,
                            ha="center", va="bottom", fontsize=5.6, color="0.2")
        ax.set_yscale("symlog", linthresh=20)
        ax.set_ylim(-10, 30000)
        ax.set_yticks([-10, 0, 20, 100, 1000])

        if k == 0:
            ax.set_yticklabels(["$-$10", "0", "20", "$10^2$", "$10^3$"])
            ax.set_ylabel("Gap (%)")
        else:
            ax.set_yticklabels([])
        ax.axhline(0.0, color="0.25", lw=0.7, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_title(title, loc="left", pad=4, fontsize=7.2)
        ax.grid(axis="y", ls=":", lw=0.4, color="0.8", zorder=0)
        ax.set_axisbelow(True)

    ax = fig.add_subplot(grid[0, 2])
    y = np.arange(len(MODELS))[::-1]
    ax.barh(y, OOD_AVG, height=0.50,
            color=[MODEL_COLORS[m] for m in MODELS], edgecolor="none")
    for model, yi, v in zip(MODELS, y, OOD_AVG):
        ax.text(v + 8, yi, f"+{v:.0f}%", va="center", fontsize=6.5)
        ax.text(2, yi + 0.34, model, ha="left", va="bottom",
                fontsize=6.3, fontweight="bold")
    ax.set_yticks([])
    ax.set_ylim(-0.6, len(MODELS) - 0.4)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 330)
    ax.set_title("(c) Average over six splits", loc="left", pad=4, fontsize=7.2)
    ax.set_xlabel("Mean gap (%)", labelpad=1)
    ax.grid(axis="x", ls=":", lw=0.4, color="0.8", zorder=0)
    ax.set_axisbelow(True)

    fig.savefig("figure4.pdf")
    plt.close(fig)

if __name__ == "__main__":
    fig_ood_gaps()
    fig_shuffle_heatmap()
    fig_channel_importance()
    fig_main_ood()
    print("wrote figureS1.pdf, figureS2.pdf, figureS3.pdf, figure4.pdf")

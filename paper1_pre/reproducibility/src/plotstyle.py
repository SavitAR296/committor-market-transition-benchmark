"""Common matplotlib style for the manuscript figures (single-column 3.4in / double-column 7in)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.3, "axes.grid": False, "figure.constrained_layout.use": True,
    "font.family": "serif", "mathtext.fontset": "cm",
})
COL = {"blue": "#1f4e79", "red": "#b22222", "orange": "#e07b00", "green": "#2e7d32", "grey": "#7f7f7f",
       "purple": "#6a3d9a", "teal": "#00838f", "light": "#c9d6e3"}
CYCLE = [COL["blue"], COL["red"], COL["orange"], COL["green"], COL["purple"], COL["teal"], COL["grey"]]
W1, W2 = 3.4, 7.0   # inches

"""Page de garde du projet (fond clair, sobre, A4 portrait).

Usage : .venv/bin/python scripts/make_cover.py
Sortie : outputs/page_de_garde.png (A4, 150 dpi).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs", "page_de_garde.png")

BLEU = "#2f5b8f"
GRIS = "#5a6b7c"
CLAIR = "#d7e2ee"
FOND = "#f7f9fc"

rng = np.random.default_rng(7)

# A4 portrait en pouces (8.27 x 11.69), 150 dpi.
fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=150)
fig.patch.set_facecolor(FOND)
ax.set_facecolor(FOND)
ax.set_xlim(0, 100)
ax.set_ylim(0, 140)
ax.axis("off")
for s in ["top", "bottom", "left", "right"]:
    ax.spines[s].set_visible(False)

# Bandeau superieur fin.
ax.add_patch(patches.Rectangle((0, 134), 100, 2.2, color=BLEU, zorder=3))

# Motif reseau discret (haut de page).
nodes = np.column_stack([rng.uniform(12, 88, 16), rng.uniform(96, 128, 16)])
for i in range(len(nodes)):
    d = np.linalg.norm(nodes - nodes[i], axis=1)
    for j in np.where((d > 0) & (d < 22))[0]:
        ax.plot([nodes[i][0], nodes[j][0]], [nodes[i][1], nodes[j][1]],
                color=CLAIR, lw=1.2, zorder=1)
ax.scatter(nodes[:, 0], nodes[:, 1], s=60, color="#9db9d4",
           edgecolors=BLEU, linewidths=1.1, zorder=2)

# Bouclier contour + coche (sobre, centre).
cx, cy, w, h = 50, 76, 26, 30
shield = patches.Polygon(
    [[cx - w / 2, cy + h / 2], [cx + w / 2, cy + h / 2],
     [cx + w / 2, cy - h * 0.05], [cx, cy - h / 2],
     [cx - w / 2, cy - h * 0.05]],
    closed=True, fill=False, edgecolor=BLEU, linewidth=3)
ax.add_patch(shield)
ax.plot([cx - 6, cx - 1.5, cx + 7], [cy + 1, cy - 4, cy + 6],
        color=BLEU, linewidth=3.5, solid_capstyle="round",
        solid_joinstyle="round")

# Titre.
ax.text(50, 52, "IDS Hybride", ha="center", va="center",
        fontsize=44, fontweight="bold", color="#1e3a5c")
ax.text(50, 44.5, "pour la détection d'intrusion", ha="center", va="center",
        fontsize=22, color=GRIS)
ax.plot([30, 70], [39.5, 39.5], color=BLEU, lw=1.5)
ax.text(50, 35.5, "SVM + K-Means  ·  NSL-KDD", ha="center", va="center",
        fontsize=14, color=GRIS)
ax.text(50, 31.5, "TP Intelligence Artificielle", ha="center",
        va="center", fontsize=12, color="#8a99a8")

# Pied de page discret.
ax.text(50, 4, "Analyse de trafic réseau sur dataset (laboratoire)",
        ha="center", va="center", fontsize=10, color="#8a99a8")

fig.tight_layout(pad=0.5)
fig.savefig(OUT, dpi=150, facecolor=FOND)
plt.close(fig)
print(f"[OK] {OUT} ({os.path.getsize(OUT)} octets)")

"""Bar chart of the shared metric (token F1 on present roles) for every system,
from results/comparison.json (written by scripts/compare.py).

    python scripts/plot_comparison.py   -> results/comparison.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results/comparison.json"))
rows = [(n, t["ALL"]["token_f1_when_present"]) for n, t in d["role_answers"].items() if n != "always answer nothing"]
generators = {"RNN encoder-decoder + attention (beam 16)", "FLAN-T5-base (beam 4)"}
names = [n.replace(" (beam 16)", "").replace(" (beam 4)", "") for n, _ in rows]
colors = ["#e07a5f" if n in generators else "#3d5a80" for n, _ in rows]

fig, ax = plt.subplots(figsize=(7.5, 4.6))
bars = ax.barh(names[::-1], [v for _, v in rows][::-1], color=colors[::-1])
for b, (_, v) in zip(bars, rows[::-1]):
    ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)
ax.set_xlim(0, 1)
ax.set_xlabel("token F1 on roles that are present (validation, 1,316 questions)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#3d5a80"), plt.Rectangle((0, 0), 1, 1, color="#e07a5f")],
          labels=["tagger (BIO labels)", "generator (question answering)"], frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.35, -0.16), ncol=2)
fig.tight_layout()
fig.savefig("results/comparison.png", dpi=160)
print("saved results/comparison.png")

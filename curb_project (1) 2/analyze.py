"""
Analysis of results.csv produced by simulate.py.
Writes tables (CSV) and figures (PNG) into ./out/
"""
import json, os
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("out", exist_ok=True)
df = pd.read_csv(os.environ.get("RES","results.csv"))
df["curbs"] = df["curb_sizes"].map(lambda s: [tuple(x) for x in json.loads(s)])

# --- Only square games 2x2 ... 6x6 are analysed.
df = df[df.n1 == df.n2].copy()
df["game"] = df.n1.astype(str) + "x" + df.n2.astype(str)
ORDER = [g for g in ["2x2", "3x3", "4x4", "5x5", "6x6"] if g in set(df.game)]


def cls(a, b):
    """Size class of a CURB. a x b and b x a are MERGED in the output
    (e.g. 3x4 and 4x3 -> "3x4+4x3"), since with i.i.d. payoffs the two
    players are symmetric."""
    a, b = min(a, b), max(a, b)
    return f"{a}x{b}" if a == b else f"{a}x{b}+{b}x{a}"


df["classes"] = df.curbs.map(lambda L: [cls(*x) for x in L])
df["iesds_size"] = df.iesds1.astype(str) + "x" + df.iesds2.astype(str)

# ---------------- Table 1: overview -----------------------------------
rows = []
for g in ORDER:
    d = df[df.game == g]
    nc = d.n_curb.value_counts(normalize=True)
    rows.append({
        "game": g, "N": len(d),
        "mean_#minCURB": d.n_curb.mean(),
        **{f"P(#CURB={k})": nc.get(k, 0) for k in (1, 2, 3)},
        "P(#CURB>=4)": nc[nc.index >= 4].sum(),
        "P(no pure NE)": (d.n_pure_ne == 0).mean(),
        "mean IESDS rows": d.iesds1.mean(), "mean IESDS cols": d.iesds2.mean(),
        "P(IESDS removes nothing)": ((d.iesds1 == d.n1) & (d.iesds2 == d.n2)).mean(),
    })
t1 = pd.DataFrame(rows).set_index("game")
t1.round(4).to_csv("out/table1_overview.csv")

# ---------------- Table 2: CURB size distribution ---------------------
all_cls = sorted({c for L in df.classes for c in L},
                 key=lambda s: (int(s[0]) * int(s[2]), s))
share_curbs, share_games = {}, {}
for g in ORDER:
    d = df[df.game == g]
    cnt = Counter(c for L in d.classes for c in L)
    tot = sum(cnt.values())
    share_curbs[g] = {c: cnt.get(c, 0) / tot for c in all_cls}
    share_games[g] = {c: d.classes.map(lambda L: c in L).mean() for c in all_cls}
t2a = pd.DataFrame(share_curbs).T.loc[ORDER]
t2b = pd.DataFrame(share_games).T.loc[ORDER]
t2a = t2a.loc[:, (t2a > 0).any()]; t2b = t2b[t2a.columns]
t2a.round(4).to_csv("out/table2a_share_of_all_minCURBs.csv")
t2b.round(4).to_csv("out/table2b_share_of_games_containing.csv")

# counts of non-square CURBs, ordered (a x b vs b x a) and merged
rows = []
for g in ORDER:
    cnt = Counter(x for L in df[df.game == g].curbs for x in L if x[0] != x[1])
    for (a, b) in sorted({tuple(sorted(k)) for k in cnt}):
        rows.append({"game": g, "class": cls(a, b), f"count {a}x{b}": cnt.get((a, b), 0),
                     f"count {b}x{a}": cnt.get((b, a), 0)})
t2c = pd.DataFrame([{"game": r["game"], "class": r["class"],
                     "count a<b (row player smaller)": list(r.values())[2],
                     "count a>b (col player smaller)": list(r.values())[3]} for r in rows])
t2c["merged count"] = t2c.iloc[:, 2] + t2c.iloc[:, 3]
t2c.to_csv("out/table2c_nonsquare_merged_counts.csv", index=False)

# ---------------- Hypothesis: does 2x2 grow with game size? -----------
hyp = []
for g in ORDER:
    d = df[df.game == g]
    has22 = d.classes.map(lambda L: "2x2" in L)
    nopne = d.n_pure_ne == 0
    hyp.append({
        "game": g,
        "P(game has a 2x2 CURB)": has22.mean(),
        "share of minCURBs that are 2x2": t2a.loc[g, "2x2"],
        "P(no pure NE)": nopne.mean(),
        "P(has 2x2 | no pure NE)": has22[nopne].mean(),
        "P(has 2x2 | >=1 pure NE)": has22[~nopne].mean(),
        "P(only CURB is full IESDS game | no pure NE)": (
            d[nopne].apply(lambda r: r.n_curb == 1 and r.curbs[0] == (r.iesds1, r.iesds2), axis=1).mean()),
    })
t_h = pd.DataFrame(hyp).set_index("game")
t_h.round(4).to_csv("out/table_hypothesis_2x2.csv")

# ---------------- Table 3: factor (only 1x1 / only 2x2 / only 3x3) ----
def factor(L):
    s = set(L)
    if len(s) == 1:
        c = next(iter(s))
        if c in ("1x1", "2x2", "3x3"):
            return "only " + c
        return "only other size"
    return "mixed sizes"

df["factor"] = df.classes.map(factor)
F_ORDER = ["only 1x1", "only 2x2", "only 3x3", "only other size", "mixed sizes"]
t3 = pd.crosstab(df.game, df.factor, normalize="index").reindex(index=ORDER, columns=F_ORDER).fillna(0)
t3.round(4).to_csv("out/table3_factor_share.csv")

chars = df.groupby(["game", "factor"]).agg(
    N=("seed", "size"),
    mean_n_curb=("n_curb", "mean"),
    mean_pure_ne=("n_pure_ne", "mean"),
    mean_iesds_rows=("iesds1", "mean"),
    mean_iesds_cols=("iesds2", "mean"),
).reindex(pd.MultiIndex.from_product([ORDER, F_ORDER])).dropna()
chars.round(3).to_csv("out/table3b_factor_characteristics.csv")

# save filtered game lists (seeds reproduce the exact payoff matrices)
for f in ["only 1x1", "only 2x2", "only 3x3"]:
    df[df.factor == f][["n1", "n2", "seed", "n_curb", "curb_sizes", "iesds1", "iesds2", "n_pure_ne"]] \
        .to_csv(f"out/games_{f.replace(' ', '_')}.csv", index=False)

# ---------------- Figures ---------------------------------------------
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})

# Fig 1: composition of minimal CURBs by game size (stacked bars)
comp = pd.DataFrame(index=t2a.index)
for c in ["1x1", "2x2", "2x3+3x2", "3x3", "3x4+4x3"]:
    comp[c] = t2a[c] if c in t2a else 0.0
comp["square 4x4 and larger"] = t2a[[c for c in t2a if "+" not in c and int(c[0]) >= 4]].sum(axis=1)
comp["other non-square"] = 1 - comp.sum(axis=1)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
bottom = np.zeros(len(comp))
for i, col in enumerate(comp.columns):
    ax.bar(comp.index, comp[col], bottom=bottom, color=PAL[i], label=col,
           edgecolor="white", linewidth=2, width=0.6)
    bottom += comp[col].values
ax.set_ylabel("share of all minimal CURB sets")
ax.set_title("Size of minimal CURB sets, by game size (uniform[0,1] payoffs)", color=INK, loc="left")
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
fig.tight_layout(); fig.savefig("out/fig1_curb_size_composition.png", dpi=160)

# Fig 2: 2x2 hypothesis (square games)
sq = [g for g in ["2x2", "3x3", "4x4", "5x5", "6x6"] if g in ORDER]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for i, (col, lab) in enumerate([
        ("P(game has a 2x2 CURB)", "P(game has a 2x2 minimal CURB)"),
        ("share of minCURBs that are 2x2", "share of minimal CURBs that are 2x2"),
        ("P(no pure NE)", "P(no pure Nash equilibrium)")]):
    y = t_h.loc[sq, col]
    ax.plot(sq, y, color=PAL[i], lw=2, marker="o", ms=8, label=lab)
    ax.annotate(f"{y.iloc[-1]:.2f}", (len(sq) - 1, y.iloc[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", color=INK)
ax.set_ylim(0, 1); ax.set_ylabel("probability / share")
ax.set_title("Does the 2x2 CURB become more common in larger games?", color=INK, loc="left")
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("out/fig2_2x2_hypothesis.png", dpi=160)

# Fig 3: factor shares
fig, ax = plt.subplots(figsize=(7.5, 4.2))
bottom = np.zeros(len(t3))
for i, col in enumerate(F_ORDER):
    ax.bar(t3.index, t3[col], bottom=bottom, color=PAL[i], label=col,
           edgecolor="white", linewidth=2, width=0.6)
    bottom += t3[col].values
ax.set_ylabel("share of games")
ax.set_title("Games whose minimal CURBs are all one size", color=INK, loc="left")
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
fig.tight_layout(); fig.savefig("out/fig3_factor_shares.png", dpi=160)

pd.set_option("display.width", 200, "display.max_columns", 30)
print(t1.round(3)); print(); print(t2a.round(3)); print(); print(t2c); print(); print(t2b.round(3))
print(); print(t_h.round(3)); print(); print(t3.round(3)); print(); print(chars.round(2))

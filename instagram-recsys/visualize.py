"""
Visualization Module
====================
Generates:
  1. Account-account similarity heatmap (content-based)
  2. User-user similarity heatmap (collaborative)
  3. Recommendation graph (user → recommended accounts)
  4. Evaluation bar chart (precision / recall per user)
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
import os

OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Consistent style
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#0e1117",
    "text.color": "#fafafa",
    "axes.labelcolor": "#fafafa",
    "xtick.color": "#aaaaaa",
    "ytick.color": "#aaaaaa",
    "axes.edgecolor": "#333333",
    "grid.color": "#222222",
    "font.family": "sans-serif",
})

CMAP = "magma"


# ======================================================================
# 1. Account similarity heatmap
# ======================================================================
def plot_account_similarity_heatmap(
    sim_df: pd.DataFrame, save_path: str | None = None,
) -> str:
    """Heatmap of account-account cosine similarity (content-based)."""
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        sim_df,
        cmap=CMAP,
        linewidths=0.3,
        linecolor="#1a1a2e",
        annot=False,
        square=True,
        cbar_kws={"shrink": 0.6, "label": "Cosine Similarity"},
        ax=ax,
    )
    ax.set_title("Account–Account Similarity  (TF-IDF + Cosine)", fontsize=15, pad=12)
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    ax.tick_params(axis="y", labelsize=7, rotation=0)
    plt.tight_layout()
    path = save_path or os.path.join(OUTPUT_DIR, "account_similarity_heatmap.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


# ======================================================================
# 2. User-user similarity heatmap
# ======================================================================
def plot_user_similarity_heatmap(
    sim_df: pd.DataFrame, max_users: int = 25, save_path: str | None = None,
) -> str:
    """Heatmap of user-user cosine similarity (collaborative)."""
    sub = sim_df.iloc[:max_users, :max_users]
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        sub,
        cmap="viridis",
        linewidths=0.4,
        linecolor="#1a1a2e",
        annot=False,
        square=True,
        cbar_kws={"shrink": 0.6, "label": "Cosine Similarity"},
        ax=ax,
    )
    ax.set_title("User–User Similarity  (Collaborative)", fontsize=15, pad=12)
    ax.tick_params(axis="x", labelsize=8, rotation=90)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    plt.tight_layout()
    path = save_path or os.path.join(OUTPUT_DIR, "user_similarity_heatmap.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


# ======================================================================
# 3. Recommendation graph
# ======================================================================
CAT_COLORS = {
    "tech": "#00d4ff",
    "travel": "#22c55e",
    "fitness": "#f97316",
    "food": "#ef4444",
    "fashion": "#a855f7",
    "music": "#eab308",
    "unknown": "#888888",
}


def plot_recommendation_graph(
    user_id: str,
    followed: list[str],
    recommendations: list[tuple[str, float, str]],
    follow_cats: dict[str, str] | None = None,
    save_path: str | None = None,
) -> str:
    """
    Radial graph: user in centre, followed accounts on inner ring,
    recommended accounts on outer ring.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#0e1117")

    # Centre node (user)
    ax.add_patch(plt.Circle((0, 0), 0.12, color="#e11d48", zorder=5))
    ax.text(0, 0, user_id, ha="center", va="center", fontsize=9, fontweight="bold",
            color="white", zorder=6)

    # Inner ring: followed accounts
    n_f = len(followed)
    for i, acct in enumerate(followed):
        angle = 2 * np.pi * i / max(n_f, 1)
        x, y = 0.6 * np.cos(angle), 0.6 * np.sin(angle)
        cat = (follow_cats or {}).get(acct, "unknown")
        col = CAT_COLORS.get(cat, "#888888")
        ax.plot([0, x], [0, y], color="#555555", lw=0.8, zorder=1)
        ax.add_patch(plt.Circle((x, y), 0.065, color=col, zorder=4))
        ax.text(x, y, acct, ha="center", va="center", fontsize=5, color="white", zorder=5)

    # Outer ring: recommendations
    n_r = len(recommendations)
    for i, (acct, score, cat) in enumerate(recommendations):
        angle = 2 * np.pi * i / max(n_r, 1) + np.pi / n_r  # offset
        x, y = 1.15 * np.cos(angle), 1.15 * np.sin(angle)
        col = CAT_COLORS.get(cat, "#888888")
        lw = 0.5 + 2.0 * score  # thicker line = higher score
        ax.plot([0, x], [0, y], color=col, lw=lw, alpha=0.5, ls="--", zorder=1)
        ax.add_patch(plt.Circle((x, y), 0.07, color=col, alpha=0.85, zorder=4))
        ax.text(x, y, acct, ha="center", va="center", fontsize=5, color="white", zorder=5)
        ax.text(x, y - 0.10, f"{score:.2f}", ha="center", va="center", fontsize=5,
                color="#cccccc", zorder=5)

    # Legend
    for i, (cat, col) in enumerate(CAT_COLORS.items()):
        ax.add_patch(plt.Circle((-1.5, 1.4 - i * 0.12), 0.035, color=col))
        ax.text(-1.43, 1.4 - i * 0.12, cat, fontsize=7, va="center", color="#cccccc")

    ax.set_title(f"Recommendations for {user_id}", fontsize=14, color="white", pad=8)
    plt.tight_layout()
    path = save_path or os.path.join(OUTPUT_DIR, f"rec_graph_{user_id}.png")
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ======================================================================
# 4. Evaluation metrics bar chart
# ======================================================================
def plot_evaluation_metrics(eval_df: pd.DataFrame, save_path: str | None = None) -> str:
    """Bar chart of precision@k and recall@k aggregated over users."""
    means = eval_df[["precision@k", "recall@k", "hit"]].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#00d4ff", "#a855f7", "#22c55e"]
    bars = ax.bar(means.index, means.values, color=colors, edgecolor="#222", linewidth=0.8)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=11, color="#fafafa")
    ax.set_ylim(0, max(means.values) * 1.3 + 0.05)
    ax.set_ylabel("Score")
    ax.set_title("Evaluation Metrics  (Leave-One-Out, k = 5)", fontsize=13, pad=10)
    plt.tight_layout()
    path = save_path or os.path.join(OUTPUT_DIR, "evaluation_metrics.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


# ======================================================================
# 5. Method comparison bar chart
# ======================================================================
def plot_method_comparison(
    user_id: str,
    all_recs: dict[str, list[tuple[str, float, str]]],
    save_path: str | None = None,
) -> str:
    """Grouped horizontal bars comparing scores across the three methods."""
    methods = list(all_recs.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(16, 5), sharey=False)
    palette = {"content_based": "#00d4ff", "collaborative": "#a855f7", "hybrid": "#22c55e"}

    for ax, method in zip(axes, methods):
        recs = all_recs[method]
        names = [a for a, _, _ in recs]
        scores = [s for _, s, _ in recs]
        cats = [c for _, _, c in recs]
        colors = [CAT_COLORS.get(c, "#888") for c in cats]
        ax.barh(names[::-1], scores[::-1], color=colors[::-1], edgecolor="#222", height=0.6)
        ax.set_title(method.replace("_", " ").title(), fontsize=11, color=palette.get(method, "#ccc"))
        ax.set_xlim(0, max(scores) * 1.25 + 0.01)
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle(f"Method Comparison — {user_id}", fontsize=14, color="white", y=1.02)
    plt.tight_layout()
    path = save_path or os.path.join(OUTPUT_DIR, f"method_comparison_{user_id}.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dataset import generate_user_follows, get_account_descriptions
    from engine import HybridRecommender, evaluate_recommender

    follows_df = generate_user_follows(n_users=50, seed=42)
    accounts_df = get_account_descriptions()
    engine = HybridRecommender(follows_df, accounts_df, alpha=0.5)

    # 1 & 2  — heatmaps
    p1 = plot_account_similarity_heatmap(engine.cb.get_similarity_matrix())
    p2 = plot_user_similarity_heatmap(engine.cf.get_user_similarity_matrix())
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")

    # 3 — recommendation graph for a demo user
    user = "User1"
    follows = follows_df.loc[follows_df["UserID"] == user, "Followed_Account"].tolist()
    cat_map = dict(zip(accounts_df["account"], accounts_df["category"]))
    follow_cats = {a: cat_map.get(a, "unknown") for a in follows}
    recs = engine.recommend_accounts(user, top_n=8)
    p3 = plot_recommendation_graph(user, follows, recs, follow_cats)
    print(f"Saved: {p3}")

    # 4 — evaluation
    eval_df = evaluate_recommender(follows_df, accounts_df, alpha=0.5, k=5)
    p4 = plot_evaluation_metrics(eval_df)
    print(f"Saved: {p4}")

    # 5 — method comparison
    all_recs = engine.recommend_all_methods(user, top_n=5)
    p5 = plot_method_comparison(user, all_recs)
    print(f"Saved: {p5}")

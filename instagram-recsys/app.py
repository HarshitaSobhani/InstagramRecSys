"""
Streamlit Interface for Instagram Account Recommendation System
================================================================
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import generate_user_follows, get_account_descriptions, ACCOUNT_CATALOG
from engine import HybridRecommender, evaluate_recommender
from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Instagram Rec System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background: linear-gradient(135deg, #1e1e2f 0%, #2d1b4e 100%);
                padding: 16px; border-radius: 12px; border: 1px solid #333; }
    .stMetric label { color: #c4b5fd !important; font-weight: 600 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 2rem !important; }
    .rec-card { background: #16182a; border-radius: 12px; padding: 14px 18px;
                margin-bottom: 8px; border-left: 4px solid; }
    .cat-tech { border-color: #00d4ff; }
    .cat-travel { border-color: #22c55e; }
    .cat-fitness { border-color: #f97316; }
    .cat-food { border-color: #ef4444; }
    .cat-fashion { border-color: #a855f7; }
    .cat-music { border-color: #eab308; }
    .tag { display: inline-block; padding: 2px 10px; border-radius: 999px;
           font-size: 0.75em; font-weight: 600; margin-right: 6px; }
</style>
""", unsafe_allow_html=True)

CAT_COLORS = {
    "tech": "#00d4ff", "travel": "#22c55e", "fitness": "#f97316",
    "food": "#ef4444", "fashion": "#a855f7", "music": "#eab308",
}

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(n_users: int, seed: int):
    follows_df = generate_user_follows(n_users=n_users, seed=seed)
    accounts_df = get_account_descriptions()
    cat_map = dict(zip(accounts_df["account"], accounts_df["category"]))
    return follows_df, accounts_df, cat_map


@st.cache_resource
def build_engine(n_users: int, seed: int, alpha: float):
    follows_df, accounts_df, _ = load_data(n_users, seed)
    return HybridRecommender(follows_df, accounts_df, alpha=alpha)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Settings")
n_users = st.sidebar.slider("Number of users", 20, 100, 50)
alpha = st.sidebar.slider("Hybrid α  (1 = pure CB, 0 = pure CF)", 0.0, 1.0, 0.5, 0.05)
top_n = st.sidebar.slider("Top-N recommendations", 3, 15, 5)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

follows_df, accounts_df, cat_map = load_data(n_users, seed)
engine = build_engine(n_users, seed, alpha)

users = sorted(follows_df["UserID"].unique(), key=lambda x: int(x.replace("User", "")))
selected_user = st.sidebar.selectbox("Select user", users)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Instagram Account Recommendation System")
st.caption("Content-Based · Collaborative · Hybrid  —  built with scikit-learn & Streamlit")

# ---------------------------------------------------------------------------
# User overview
# ---------------------------------------------------------------------------
user_follows = follows_df.loc[follows_df["UserID"] == selected_user]
follow_list = user_follows["Followed_Account"].tolist()

col1, col2, col3 = st.columns(3)
col1.metric("Accounts Followed", len(follow_list))
cat_counts = user_follows["Category"].value_counts()
col2.metric("Primary Interest", cat_counts.index[0] if len(cat_counts) > 0 else "—")
col3.metric("Categories Spanned", user_follows["Category"].nunique())

st.subheader(f"{selected_user}'s followed accounts")
tags_html = ""
for _, row in user_follows.iterrows():
    col = CAT_COLORS.get(row["Category"], "#888")
    tags_html += (
        f'<span class="tag" style="background:{col}22;color:{col};border:1px solid {col}55;">'
        f'{row["Followed_Account"]}</span> '
    )
st.markdown(tags_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Recommendations")

all_recs = engine.recommend_all_methods(selected_user, top_n=top_n)
tab_hybrid, tab_cb, tab_cf = st.tabs(["Hybrid", "Content-Based", "Collaborative"])


def render_recs(recs):
    if not recs:
        st.info("No recommendations available.")
        return
    for rank, (acct, score, cat) in enumerate(recs, 1):
        col = CAT_COLORS.get(cat, "#888")
        st.markdown(
            f'<div class="rec-card cat-{cat}">'
            f'<strong style="color:#fafafa;font-size:1.05em;">{rank}. @{acct}</strong>'
            f'&nbsp;&nbsp;<span class="tag" style="background:{col}22;color:{col};'
            f'border:1px solid {col}55;">{cat}</span>'
            f'<br/><span style="color:#999;font-size:0.85em;">score: {score:.4f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


with tab_hybrid:
    render_recs(all_recs["hybrid"])
with tab_cb:
    render_recs(all_recs["content_based"])
with tab_cf:
    render_recs(all_recs["collaborative"])

# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Similarity Heatmaps")
heat_tab1, heat_tab2 = st.tabs(["Account Similarity (Content)", "User Similarity (Collab)"])

with heat_tab1:
    sim_df = engine.cb.get_similarity_matrix()
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    sns.heatmap(sim_df, cmap="magma", linewidths=0.2, linecolor="#1a1a2e",
                square=True, cbar_kws={"shrink": 0.55}, ax=ax)
    ax.set_title("Account–Account Cosine Similarity", color="#fafafa", fontsize=13)
    ax.tick_params(colors="#aaa", labelsize=6)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with heat_tab2:
    user_sim = engine.cf.get_user_similarity_matrix()
    sub = user_sim.iloc[:25, :25]
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    fig2.patch.set_facecolor("#0e1117")
    ax2.set_facecolor("#0e1117")
    sns.heatmap(sub, cmap="viridis", linewidths=0.3, linecolor="#1a1a2e",
                square=True, cbar_kws={"shrink": 0.55}, ax=ax2)
    ax2.set_title("User–User Cosine Similarity (first 25)", color="#fafafa", fontsize=13)
    ax2.tick_params(colors="#aaa", labelsize=7)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

# ---------------------------------------------------------------------------
# Method comparison chart
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Method Comparison")
methods = list(all_recs.keys())
fig3, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig3.patch.set_facecolor("#0e1117")
palette = {"content_based": "#00d4ff", "collaborative": "#a855f7", "hybrid": "#22c55e"}

for ax, method in zip(axes, methods):
    ax.set_facecolor("#0e1117")
    recs = all_recs[method]
    names = [a for a, _, _ in recs]
    scores = [s for _, s, _ in recs]
    cats = [c for _, _, c in recs]
    colors = [CAT_COLORS.get(c, "#888") for c in cats]
    ax.barh(names[::-1], scores[::-1], color=colors[::-1], edgecolor="#222", height=0.55)
    ax.set_title(method.replace("_", " ").title(), fontsize=10,
                 color=palette.get(method, "#ccc"))
    ax.tick_params(colors="#aaa", labelsize=7)
    if scores:
        ax.set_xlim(0, max(scores) * 1.3 + 0.01)

plt.tight_layout()
st.pyplot(fig3, use_container_width=True)
plt.close(fig3)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Evaluation  (Leave-One-Out)")
with st.spinner("Running evaluation…"):
    eval_df = evaluate_recommender(follows_df, accounts_df, alpha=alpha, k=top_n)
e1, e2, e3 = st.columns(3)
e1.metric("Mean Precision@k", f"{eval_df['precision@k'].mean():.4f}")
e2.metric("Mean Recall@k", f"{eval_df['recall@k'].mean():.4f}")
e3.metric("Hit Rate", f"{eval_df['hit'].mean():.4f}")

fig4, ax4 = plt.subplots(figsize=(6, 4))
fig4.patch.set_facecolor("#0e1117")
ax4.set_facecolor("#0e1117")
means = eval_df[["precision@k", "recall@k", "hit"]].mean()
bars = ax4.bar(means.index, means.values, color=["#00d4ff", "#a855f7", "#22c55e"],
               edgecolor="#222", linewidth=0.8)
for b in bars:
    ax4.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
             f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=10, color="#fafafa")
ax4.set_ylim(0, max(means.values) * 1.35 + 0.05)
ax4.set_title("Evaluation Metrics", color="#fafafa", fontsize=12)
ax4.tick_params(colors="#aaa")
st.pyplot(fig4, use_container_width=True)
plt.close(fig4)

# ---------------------------------------------------------------------------
# Theory section
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("How It Works")

with st.expander("Content-Based Filtering"):
    st.markdown("""
**Idea**: Recommend accounts *similar to* the ones you already follow.

1. Each account has a text description (keywords about its niche).
2. We convert descriptions into **TF-IDF vectors** — numerical representations
   that capture the importance of each word relative to the corpus.
3. A **user profile** is built as the centroid (mean) of their followed accounts' vectors.
4. We compute **cosine similarity** between the user profile and every other account.
5. The highest-scoring accounts (that the user hasn't followed yet) become recommendations.

**Strengths**: No cold-start for items; explains *why* something is recommended.
**Weakness**: Limited to textual features; can't discover novel interests.
""")

with st.expander("Collaborative Filtering"):
    st.markdown("""
**Idea**: "Users who follow similar accounts to you also follow these other accounts."

1. Build a **binary interaction matrix** (users × accounts).
2. Compute **user-user cosine similarity** from this matrix.
3. For a target user, find the **K nearest neighbours** (most similar users).
4. Aggregate accounts followed by neighbours, weighted by similarity, excluding
   accounts the user already follows.
5. Top-scoring candidates become recommendations.

**Strengths**: Discovers unexpected interests; no need for item features.
**Weakness**: Cold-start for new users; popularity bias.
""")

with st.expander("Hybrid Approach"):
    st.markdown("""
We combine both methods with a **weighted sum**:

```
hybrid_score = α × CB_normalised + (1 − α) × CF_normalised
```

Both sub-scores are min-max normalised to [0, 1] before blending.
Use the slider on the left to adjust α and see how it shifts recommendations.
""")

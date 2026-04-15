"""
Hybrid Recommendation Engine + Evaluation
==========================================
Combines content-based and collaborative scores, and evaluates
with precision@k and recall@k using a leave-one-out protocol.
"""

import numpy as np
import pandas as pd
from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender
from dataset import generate_user_follows, get_account_descriptions


# ======================================================================
# Hybrid Recommender
# ======================================================================
class HybridRecommender:
    """
    Weighted-sum hybrid of content-based and collaborative filtering.

    final_score(account) = α · CB_score + (1 − α) · CF_score

    Both sub-scores are min-max normalised to [0, 1] before mixing.
    """

    def __init__(
        self,
        follows_df: pd.DataFrame,
        accounts_df: pd.DataFrame,
        alpha: float = 0.5,
        k_neighbors: int = 10,
    ):
        self.follows_df = follows_df
        self.accounts_df = accounts_df
        self.alpha = alpha
        self.cat_map = dict(zip(accounts_df["account"], accounts_df["category"]))

        # Sub-systems
        self.cb = ContentBasedRecommender(accounts_df)
        self.cf = CollaborativeRecommender(follows_df, k_neighbors=k_neighbors)

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        lo, hi = min(scores.values()), max(scores.values())
        rng = hi - lo if hi != lo else 1.0
        return {k: (v - lo) / rng for k, v in scores.items()}

    # ------------------------------------------------------------------
    def recommend_accounts(
        self,
        user_id: str,
        top_n: int = 5,
    ) -> list[tuple[str, float, str]]:
        """
        Main recommendation function.

        Returns list of (account, hybrid_score, category).
        """
        # Followed list
        user_follows = self.follows_df.loc[
            self.follows_df["UserID"] == user_id, "Followed_Account"
        ].tolist()

        if not user_follows:
            return []

        # --- Content-based scores (request more so we have a wide pool) ---
        cb_recs = self.cb.recommend(user_follows, top_n=30)
        cb_scores = {acct: sc for acct, sc, _ in cb_recs}

        # --- Collaborative scores ---
        cf_recs = self.cf.recommend(user_id, top_n=30, category_map=self.cat_map)
        cf_scores = {acct: sc for acct, sc, _ in cf_recs}

        # --- Normalise & merge ---
        cb_norm = self._normalise(cb_scores)
        cf_norm = self._normalise(cf_scores)

        all_candidates = set(cb_norm) | set(cf_norm)
        hybrid: dict[str, float] = {}
        for acct in all_candidates:
            cb_val = cb_norm.get(acct, 0.0)
            cf_val = cf_norm.get(acct, 0.0)
            hybrid[acct] = self.alpha * cb_val + (1 - self.alpha) * cf_val

        ranked = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [(acct, sc, self.cat_map.get(acct, "unknown")) for acct, sc in ranked]

    # ------------------------------------------------------------------
    def recommend_all_methods(
        self, user_id: str, top_n: int = 5
    ) -> dict[str, list[tuple[str, float, str]]]:
        """Return recommendations from all three methods for comparison."""
        user_follows = self.follows_df.loc[
            self.follows_df["UserID"] == user_id, "Followed_Account"
        ].tolist()
        return {
            "content_based": self.cb.recommend(user_follows, top_n=top_n),
            "collaborative": self.cf.recommend(user_id, top_n=top_n, category_map=self.cat_map),
            "hybrid": self.recommend_accounts(user_id, top_n=top_n),
        }


# ======================================================================
# Evaluation  (leave-one-out)
# ======================================================================
def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k recommendations that are relevant."""
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for r in top_k if r in relevant)
    return hits / len(top_k)


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items that appear in top-k."""
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / len(relevant)


def evaluate_recommender(
    follows_df: pd.DataFrame,
    accounts_df: pd.DataFrame,
    alpha: float = 0.5,
    k: int = 5,
    n_trials: int | None = None,
    seed: int = 99,
) -> pd.DataFrame:
    """
    Leave-one-out evaluation.

    For each user, hide one followed account, rebuild the recommender
    on the remaining data, and check whether the hidden account appears
    in the top-k recommendations.

    Parameters
    ----------
    n_trials : if set, sample this many user-account pairs to speed up evaluation.

    Returns
    -------
    DataFrame with per-trial precision@k, recall@k and aggregated means.
    """
    rng = np.random.default_rng(seed)
    users = follows_df["UserID"].unique()
    trials: list[dict] = []

    for user in users:
        user_accts = follows_df.loc[follows_df["UserID"] == user, "Followed_Account"].tolist()
        if len(user_accts) < 2:
            continue  # need at least 2 follows to hide one

        # Pick one to hide
        hidden = rng.choice(user_accts)
        remaining_df = follows_df[
            ~((follows_df["UserID"] == user) & (follows_df["Followed_Account"] == hidden))
        ].copy()

        rec = HybridRecommender(remaining_df, accounts_df, alpha=alpha)
        recs = rec.recommend_accounts(user, top_n=k)
        rec_list = [a for a, _, _ in recs]

        trials.append(
            {
                "UserID": user,
                "hidden": hidden,
                "precision@k": precision_at_k(rec_list, {hidden}, k),
                "recall@k": recall_at_k(rec_list, {hidden}, k),
                "hit": int(hidden in rec_list),
            }
        )
        if n_trials and len(trials) >= n_trials:
            break

    results = pd.DataFrame(trials)
    return results


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    follows_df = generate_user_follows(n_users=50, seed=42)
    accounts_df = get_account_descriptions()

    print("=" * 60)
    print("HYBRID RECOMMENDATION DEMO")
    print("=" * 60)

    engine = HybridRecommender(follows_df, accounts_df, alpha=0.5)

    for user in ["User1", "User5", "User10"]:
        follows = follows_df.loc[follows_df["UserID"] == user, "Followed_Account"].tolist()
        print(f"\n{user} follows ({len(follows)}): {follows}")
        all_recs = engine.recommend_all_methods(user, top_n=5)
        for method, recs in all_recs.items():
            print(f"  [{method}]")
            for rank, (acct, sc, cat) in enumerate(recs, 1):
                print(f"    {rank}. {acct:<25s} score={sc:.3f}  ({cat})")

    print("\n" + "=" * 60)
    print("EVALUATION  (leave-one-out, k=5)")
    print("=" * 60)
    eval_df = evaluate_recommender(follows_df, accounts_df, alpha=0.5, k=5)
    print(f"Trials: {len(eval_df)}")
    print(f"Mean precision@5 : {eval_df['precision@k'].mean():.4f}")
    print(f"Mean recall@5    : {eval_df['recall@k'].mean():.4f}")
    print(f"Hit rate         : {eval_df['hit'].mean():.4f}")

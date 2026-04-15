"""
Collaborative Filtering
========================
Recommends accounts that similar *users* follow, using a
user-item interaction matrix and user-user cosine similarity (KNN).
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


class CollaborativeRecommender:
    """
    User-user collaborative filtering.

    1. Build a binary user × account interaction matrix.
    2. Compute user-user cosine similarity.
    3. For a target user, find K nearest neighbours and aggregate
       the accounts they follow (weighted by similarity), excluding
       accounts the target already follows.
    """

    def __init__(self, follows_df: pd.DataFrame, k_neighbors: int = 10):
        """
        Parameters
        ----------
        follows_df : DataFrame with columns [UserID, Followed_Account]
        k_neighbors : number of nearest neighbours to consider
        """
        self.follows_df = follows_df.copy()
        self.k = k_neighbors

        # Build binary interaction matrix
        self.interaction = (
            follows_df
            .assign(val=1)
            .pivot_table(index="UserID", columns="Followed_Account", values="val", fill_value=0)
        )
        self.users = list(self.interaction.index)
        self.accounts = list(self.interaction.columns)

        # User-user cosine similarity
        self.user_sim_matrix = cosine_similarity(self.interaction.values)
        self.user_sim_df = pd.DataFrame(
            self.user_sim_matrix, index=self.users, columns=self.users,
        )

        # Fit KNN model
        self.knn = NearestNeighbors(
            n_neighbors=min(k_neighbors + 1, len(self.users)),
            metric="cosine",
            algorithm="brute",
        )
        self.knn.fit(self.interaction.values)

    # ------------------------------------------------------------------
    def get_user_similarity_matrix(self) -> pd.DataFrame:
        """Return labelled user-user similarity DataFrame."""
        return self.user_sim_df

    # ------------------------------------------------------------------
    def recommend(
        self,
        user_id: str,
        top_n: int = 5,
        category_map: dict[str, str] | None = None,
    ) -> list[tuple[str, float, str]]:
        """
        Recommend accounts for *user_id*.

        Steps
        -----
        1. Find K nearest users via KNN.
        2. For each neighbour, weight accounts they follow by similarity score.
        3. Sum weighted scores per candidate account.
        4. Exclude already-followed accounts.
        5. Return top-N sorted by aggregated score.

        Returns
        -------
        List of (account, score, category) tuples.
        """
        if user_id not in self.users:
            return []

        user_idx = self.users.index(user_id)
        user_vec = self.interaction.values[user_idx].reshape(1, -1)

        distances, indices = self.knn.kneighbors(user_vec)
        # distances are cosine *distances* → similarity = 1 − distance
        neighbor_sims = 1 - distances.flatten()
        neighbor_idxs = indices.flatten()

        # Weighted aggregation
        already_followed = set(
            self.follows_df.loc[self.follows_df["UserID"] == user_id, "Followed_Account"]
        )
        scores: dict[str, float] = {}
        for ni, sim in zip(neighbor_idxs, neighbor_sims):
            if self.users[ni] == user_id:
                continue  # skip self
            for j, acct in enumerate(self.accounts):
                if self.interaction.values[ni, j] == 1 and acct not in already_followed:
                    scores[acct] = scores.get(acct, 0.0) + sim

        # Resolve categories
        cat_map = category_map or {}
        candidates = [(acct, sc, cat_map.get(acct, "unknown")) for acct, sc in scores.items()]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_n]


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dataset import generate_user_follows, get_account_descriptions

    follows_df = generate_user_follows()
    accounts_df = get_account_descriptions()
    cat_map = dict(zip(accounts_df["account"], accounts_df["category"]))

    cf = CollaborativeRecommender(follows_df, k_neighbors=10)

    user = "User1"
    recs = cf.recommend(user, top_n=5, category_map=cat_map)
    follows = follows_df.loc[follows_df["UserID"] == user, "Followed_Account"].tolist()
    print(f"{user} follows: {follows}")
    print(f"\nCollaborative recommendations for {user}:")
    for rank, (acct, score, cat) in enumerate(recs, 1):
        print(f"  {rank}. {acct:<25s} (score={score:.3f}, cat={cat})")

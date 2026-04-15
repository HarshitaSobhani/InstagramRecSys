"""
Content-Based Filtering
========================
Recommends accounts similar to ones a user already follows,
based on textual descriptions (TF-IDF → Cosine Similarity).
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """
    Builds a TF-IDF profile for every Instagram account from its description,
    then recommends accounts whose profiles are most similar to the centroid
    of a user's already-followed accounts.
    """

    def __init__(self, accounts_df: pd.DataFrame):
        """
        Parameters
        ----------
        accounts_df : DataFrame with columns [account, desc, category]
        """
        self.accounts_df = accounts_df.copy().reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.accounts_df["desc"])
        # Pre-compute pairwise similarity (accounts × accounts)
        self.sim_matrix = cosine_similarity(self.tfidf_matrix)

    # ------------------------------------------------------------------
    def get_similarity_matrix(self) -> pd.DataFrame:
        """Return the account-account similarity as a labelled DataFrame."""
        labels = self.accounts_df["account"].tolist()
        return pd.DataFrame(self.sim_matrix, index=labels, columns=labels)

    # ------------------------------------------------------------------
    def recommend(
        self,
        followed_accounts: list[str],
        top_n: int = 5,
    ) -> list[tuple[str, float, str]]:
        """
        Recommend accounts for a user given their followed list.

        1. Build a *user profile vector* = mean of TF-IDF vectors of followed accounts.
        2. Compute cosine similarity between the profile and every account.
        3. Exclude already-followed accounts and return top-N.

        Returns
        -------
        List of (account_name, score, category) tuples, sorted by score descending.
        """
        idx_map = {
            name: i
            for i, name in enumerate(self.accounts_df["account"])
        }
        followed_idxs = [idx_map[a] for a in followed_accounts if a in idx_map]

        if not followed_idxs:
            return []

        # User profile = centroid of followed account vectors
        user_vec = np.asarray(self.tfidf_matrix[followed_idxs].mean(axis=0))
        # Similarity of this profile to every account
        scores = cosine_similarity(user_vec, self.tfidf_matrix).flatten()

        # Build candidate list (exclude followed)
        candidates = []
        for i, score in enumerate(scores):
            acct = self.accounts_df.iloc[i]["account"]
            cat = self.accounts_df.iloc[i]["category"]
            if acct not in followed_accounts:
                candidates.append((acct, float(score), cat))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_n]


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dataset import generate_user_follows, get_account_descriptions

    follows_df = generate_user_follows()
    accounts_df = get_account_descriptions()
    cb = ContentBasedRecommender(accounts_df)

    # Demo: recommend for User1
    user = "User1"
    user_follows = follows_df.loc[follows_df["UserID"] == user, "Followed_Account"].tolist()
    print(f"{user} follows: {user_follows}")
    recs = cb.recommend(user_follows, top_n=5)
    print(f"\nContent-Based recommendations for {user}:")
    for rank, (acct, score, cat) in enumerate(recs, 1):
        print(f"  {rank}. {acct:<25s} (score={score:.3f}, cat={cat})")

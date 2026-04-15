"""
Dataset Generation Module
=========================
Simulates a realistic Instagram follow-graph with users, accounts, and categories.
Each account has a text description used for content-based features.
"""

import pandas as pd
import numpy as np
import random

# ---------------------------------------------------------------------------
# Catalog of Instagram accounts grouped by category
# ---------------------------------------------------------------------------
ACCOUNT_CATALOG: dict[str, list[dict]] = {
    "tech": [
        {"account": "techcrunch", "desc": "startup funding tech innovation silicon valley venture capital"},
        {"account": "theverge", "desc": "consumer electronics gadgets reviews technology culture"},
        {"account": "wired", "desc": "technology science culture future digital trends"},
        {"account": "androidauthority", "desc": "android smartphones apps mobile reviews google pixel"},
        {"account": "techinsider", "desc": "technology explainer innovation science futuristic gadgets"},
        {"account": "mkbhd", "desc": "tech reviews smartphones gadgets youtube unboxing"},
        {"account": "theinformation", "desc": "tech business journalism startups silicon valley deals"},
        {"account": "mashable", "desc": "digital culture social media tech entertainment viral"},
        {"account": "engadget", "desc": "consumer electronics reviews gadgets technology news"},
        {"account": "aaboronkov_tech", "desc": "AI artificial intelligence machine learning deep learning research"},
    ],
    "travel": [
        {"account": "natgeo", "desc": "nature photography wildlife exploration planet earth conservation"},
        {"account": "beautifuldestinations", "desc": "travel luxury destinations hotels beaches resorts aerial"},
        {"account": "lonelyplanet", "desc": "travel guides backpacking adventure destinations culture tips"},
        {"account": "travelandleisure", "desc": "luxury travel hotels resorts vacation getaway spa"},
        {"account": "earthpix", "desc": "landscape photography nature scenic mountains waterfalls earth"},
        {"account": "voyaged", "desc": "travel wanderlust explore adventure destinations road trip"},
        {"account": "cntraveler", "desc": "conde nast travel luxury hotels world destinations reviews"},
        {"account": "travelawesome", "desc": "travel photography adventure explore dream destinations"},
        {"account": "globetrotters", "desc": "backpacking budget travel world tour hostels culture"},
        {"account": "passionpassport", "desc": "travel storytelling community culture photography wanderlust"},
    ],
    "fitness": [
        {"account": "fitnessworld", "desc": "workout gym exercise bodybuilding strength training health"},
        {"account": "nike", "desc": "running shoes athletics sports training performance apparel"},
        {"account": "crossfit", "desc": "crossfit wod functional fitness strength conditioning workout"},
        {"account": "yogajournal", "desc": "yoga meditation mindfulness flexibility poses wellness"},
        {"account": "menshealth", "desc": "fitness nutrition muscle health workout tips men lifestyle"},
        {"account": "womenshealthmag", "desc": "fitness nutrition wellness women health workout self care"},
        {"account": "therock", "desc": "gym workout motivation bodybuilding strength celebrity fitness"},
        {"account": "kaikifit", "desc": "personal training home workout fitness tips exercises routines"},
        {"account": "gymshark", "desc": "gym apparel fitness bodybuilding workout clothing brand"},
        {"account": "blogilates", "desc": "pilates workout fitness pop dance exercise fun youtube"},
    ],
    "food": [
        {"account": "foodnetwork", "desc": "cooking recipes chefs kitchen food television shows"},
        {"account": "bonappetit", "desc": "recipes gourmet cooking restaurants food culture magazine"},
        {"account": "tasty", "desc": "quick recipes cooking hacks food videos viral easy meals"},
        {"account": "gordonramsay", "desc": "chef cooking fine dining recipes restaurant kitchen masterchef"},
        {"account": "minimalistbaker", "desc": "vegan plant based recipes easy cooking healthy desserts"},
        {"account": "halfbakedharvest", "desc": "comfort food recipes seasonal cooking baking photography"},
        {"account": "eatingwell", "desc": "healthy eating nutrition recipes diet wellness meal prep"},
        {"account": "buzzfeedtasty", "desc": "food videos quick recipes hacks cooking viral kitchen"},
        {"account": "deliciouslyella", "desc": "plant based vegan recipes wellness healthy cooking lifestyle"},
        {"account": "thefeedfeed", "desc": "food photography recipes community cooking baking inspiration"},
    ],
    "fashion": [
        {"account": "voguemagazine", "desc": "high fashion runway designer luxury style trends couture"},
        {"account": "hm", "desc": "fast fashion affordable clothing sustainable trends outfits"},
        {"account": "zara", "desc": "fashion clothing style trends affordable designer inspired"},
        {"account": "dior", "desc": "luxury fashion couture designer paris haute accessories"},
        {"account": "asos", "desc": "online fashion streetwear youth trends clothing accessories"},
        {"account": "chiaraferragni", "desc": "fashion influencer style luxury outfits designer lifestyle"},
        {"account": "manrepeller", "desc": "fashion humor style culture women trends editorial"},
        {"account": "hypebeast", "desc": "streetwear sneakers urban fashion culture hype drops"},
        {"account": "everlane", "desc": "ethical fashion sustainable basics clothing transparency"},
        {"account": "highsnobiety", "desc": "streetwear culture fashion sneakers lifestyle editorial"},
    ],
    "music": [
        {"account": "spotify", "desc": "music streaming playlists artists songs discover new releases"},
        {"account": "rollingstone", "desc": "rock music reviews culture artists interviews magazine"},
        {"account": "billboard", "desc": "music charts top hits artists pop hip hop country"},
        {"account": "pitchfork", "desc": "indie music reviews albums artists alternative experimental"},
        {"account": "genius", "desc": "lyrics music meaning behind songs hip hop rap artists"},
        {"account": "npr_music", "desc": "public radio music tiny desk concert jazz folk world"},
        {"account": "complexmusic", "desc": "hip hop rap music culture interviews artists releases"},
        {"account": "loudwire", "desc": "rock metal music news reviews interviews heavy bands"},
        {"account": "songexploder", "desc": "podcast music creative process songwriting artists stories"},
        {"account": "stereogum", "desc": "indie rock alternative music reviews news albums artists"},
    ],
}

ALL_CATEGORIES = list(ACCOUNT_CATALOG.keys())
ALL_ACCOUNTS = []
for cat, accts in ACCOUNT_CATALOG.items():
    for a in accts:
        ALL_ACCOUNTS.append({**a, "category": cat})


def generate_user_follows(
    n_users: int = 50,
    min_follows: int = 3,
    max_follows: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic dataset of user → account follows.

    Each user has 1-2 *primary* interest categories (higher follow probability)
    and may occasionally follow accounts from other categories.

    Returns
    -------
    pd.DataFrame with columns: UserID, Followed_Account, Category, Description
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    rows: list[dict] = []
    for uid in range(1, n_users + 1):
        user_id = f"User{uid}"

        # Pick 1-2 primary interest categories
        n_primary = rng.choice([1, 2], p=[0.4, 0.6])
        primary_cats = list(rng.choice(ALL_CATEGORIES, size=n_primary, replace=False))

        n_follow = rng.integers(min_follows, max_follows + 1)
        followed: set[str] = set()

        for _ in range(n_follow):
            # 75 % chance to pick from primary categories, 25 % from others
            if rng.random() < 0.75:
                cat = rng.choice(primary_cats)
            else:
                cat = rng.choice(ALL_CATEGORIES)

            pool = [a for a in ACCOUNT_CATALOG[cat] if a["account"] not in followed]
            if not pool:
                continue
            chosen = rng.choice(pool)
            followed.add(chosen["account"])
            rows.append(
                {
                    "UserID": user_id,
                    "Followed_Account": chosen["account"],
                    "Category": cat,
                    "Description": chosen["desc"],
                }
            )

    df = pd.DataFrame(rows)
    return df


def get_account_descriptions() -> pd.DataFrame:
    """Return a DataFrame of all accounts with their descriptions and categories."""
    return pd.DataFrame(ALL_ACCOUNTS)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = generate_user_follows(n_users=50, seed=42)
    print(f"Generated {len(df)} follow-edges for {df['UserID'].nunique()} users")
    print(f"Unique accounts followed: {df['Followed_Account'].nunique()}")
    print("\nSample rows:")
    print(df.head(15).to_string(index=False))
    print("\nFollows per category:")
    print(df["Category"].value_counts().to_string())

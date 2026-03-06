from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__, template_folder='.', static_folder='static', static_url_path='/static')
CORS(app)

# -------------------------------
# SERVE INDEX.HTML
# -------------------------------

@app.route("/")
def serve_index():
    return render_template('index.html')

# -------------------------------
# Dummy Database
# -------------------------------

users = {}
interactions = []

# Load products
products = pd.DataFrame([
    {"product_id":"P1","product_name":"Laptop","brand":"Apple","category":"Electronics","price":79999,"image_url":"/static/images/laptop.jpg"},
    {"product_id":"P2","product_name":"Galaxy S24","brand":"Samsung","category":"Electronics","price":4999,"image_url":"/static/images/smartwatch.jpg"},
    {"product_id":"P3","product_name":"iPad Pro","brand":"Apple","category":"Electronics","price":89999,"image_url":"/static/images/ipad.jpg"},
    {"product_id":"P4","product_name":"Laptop","brand":"Dell","category":"Electronics","price":39999,"image_url":"/static/images/laptop1.jpg"},
    {"product_id":"P5","product_name":"Monitor","brand":"LG","category":"Electronics","price":14999,"image_url":"/static/images/monitor.jpg"},
    {"product_id":"P6","product_name":"Headphones","brand":"Sony","category":"Electronics","price":10000,"image_url":"/static/images/headphones.jpg"},
    {"product_id":"P7","product_name":"AirPods Pro","brand":"Apple","category":"Electronics","price":20000,"image_url":"/static/images/earpods.jpg"},
    {"product_id":"P8","product_name":"Smart Watch","brand":"Samsung","category":"Electronics","price":25000,"image_url":"/static/images/smartwatch.jpg"},

    {"product_id":"P9","product_name":"Running Sneakers","brand":"Nike","category":"Fashion","price":4999,"image_url":"/static/images/sneakers.jpg"},
    {"product_id":"P10","product_name":"Classic T-Shirt","brand":"H&M","category":"Fashion","price":799,"image_url":"/static/images/tshirt.jpg"},
    {"product_id":"P11","product_name":"Blue Jeans","brand":"Levi's","category":"Fashion","price":2499,"image_url":"/static/images/zeans.jpg"},
    {"product_id":"P12","product_name":"Leather Jacket","brand":"Zara","category":"Fashion","price":6999,"image_url":"/static/images/jacket.jpg"},
    {"product_id":"P13","product_name":"Kurthi","brand":"Local","category":"Fashion","price":1099,"image_url":"/static/images/Kurthi.jpg"},
    {"product_id":"P14","product_name":"Casual Pants","brand":"Uniqlo","category":"Fashion","price":1299,"image_url":"/static/images/pant.jpg"},
    {"product_id":"P15","product_name":"Formal Shirt","brand":"Arrow","category":"Fashion","price":1899,"image_url":"/static/images/shirt.jpg"},
    {"product_id":"P16","product_name":"Designer Shoes","brand":"Clarks","category":"Fashion","price":3599,"image_url":"/static/images/shoes.jpg"},

    {"product_id":"P17","product_name":"Face Serum","brand":"GlowCo","category":"Beauty","price":799,"image_url":"/static/images/bfaceserum.jpg"},
    {"product_id":"P18","product_name":"Lipstick","brand":"Beauté","category":"Beauty","price":599,"image_url":"/static/images/blipstick.jpg"},
    {"product_id":"P19","product_name":"Concealer","brand":"Tarte","category":"Beauty","price":599,"image_url":"/static/images/bconcealer.jpg"},
    {"product_id":"P20","product_name":"Compact Powder","brand":"Lakeme","category":"Beauty","price":499,"image_url":"/static/images/bcompactpowder.jpg"},
    {"product_id":"P21","product_name":"Moisturizer Cream","brand":"Nivea","category":"Beauty","price":999,"image_url":"/static/images/bcream.jpg"},
    {"product_id":"P22","product_name":"Eye Liner","brand":"SkinCare","category":"Beauty","price":199,"image_url":"/static/images/beyeliner.jpg"},
    {"product_id":"P23","product_name":"Mascara","brand":"Lash Sensational","category":"Beauty","price":309,"image_url":"/static/images/bmascara.jpg"},
    {"product_id":"P24","product_name":"Primer","brand":"SkinCare","category":"Beauty","price":399,"image_url":"/static/images/bprimer.jpg"},

    {"product_id":"P25","product_name":"Desk Lamp","brand":"Philips","category":"Home","price":1299,"image_url":"/static/images/hlamp.jpg"},
    {"product_id":"P26","product_name":"Chair","brand":"Ikea","category":"Home","price":8999,"image_url":"/static/images/hchair.jpg"},
    {"product_id":"P27","product_name":"Dining Table","brand":"BrewIt","category":"Home","price":8499,"image_url":"/static/images/hdiningtable.jpg"},
    {"product_id":"P28","product_name":"Hanging Chair","brand":"Wood","category":"Home","price":4499,"image_url":"/static/images/hangingchair.jpg"},
    {"product_id":"P29","product_name":"Mirror","brand":"BrewIt","category":"Home","price":2499,"image_url":"/static/images/hmirror.jpg"},
    {"product_id":"P30","product_name":"Sofa","brand":"BrewIt","category":"Home","price":4999,"image_url":"/static/images/hsofa.jpg"},
    {"product_id":"P31","product_name":"Wooden Bed","brand":"Ikea","category":"Home","price":10499,"image_url":"/static/images/hwoodbed.jpg"},
    {"product_id":"P27b","product_name":"Study Table","brand":"Ikea","category":"Home","price":6499,"image_url":"/static/images/studytable.jpg"},

    {"product_id":"P32","product_name":"Yoga Mat","brand":"Decathlon","category":"Sports","price":2999,"image_url":"/static/images/syogamat.jpg"},
    {"product_id":"P33","product_name":"Dumbbells Set","brand":"FitGear","category":"Sports","price":1999,"image_url":"/static/images/sdumbells.jpg"},
    {"product_id":"P34","product_name":"Shuttle Bat","brand":"Adidas","category":"Sports","price":999,"image_url":"/static/images/shuttlebat.jpg"},
    {"product_id":"P35","product_name":"Sports Bag","brand":"Adidas","category":"Sports","price":5999,"image_url":"/static/images/sbag.jpg"},
    {"product_id":"P36","product_name":"Ball","brand":"Adidas","category":"Sports","price":999,"image_url":"/static/images/sball.jpg"},
    {"product_id":"P37","product_name":"Gloves","brand":"Adidas","category":"Sports","price":2999,"image_url":"/static/images/sgloves.jpg"},
    {"product_id":"P38","product_name":"Headband","brand":"Adidas","category":"Sports","price":1999,"image_url":"/static/images/sheadband.jpg"},
    {"product_id":"P39","product_name":"Skipping Rope","brand":"Adidas","category":"Sports","price":4999,"image_url":"/static/images/skippingrope.jpg"},
])

# Remove duplicate product_ids (P27 appeared twice)
products = products.drop_duplicates(subset="product_id", keep="first").reset_index(drop=True)

# Pre-create 1000 users
for i in range(1, 1001):
    uid = f"U{i:04d}"
    users[uid] = {"password": "1234"}


# ================================================
# AUTH
# ================================================

@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    uid  = data["user_id"]
    if uid in users:
        return jsonify({"user_id": uid, "is_new": False})
    return jsonify({"error": "User not found"}), 400


@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.json
    uid  = data["user_id"]
    pwd  = data["password"]
    if uid in users:
        return jsonify({"error": "User already exists"}), 400
    users[uid] = {"password": pwd}
    return jsonify({"user_id": uid, "is_new": True})


# ================================================
# PRODUCTS
# ================================================

@app.route("/api/products")
def get_products():
    category = request.args.get("category")
    per_page = int(request.args.get("per_page", 20))
    df = products.copy()
    if category:
        df = df[df["category"] == category]
    return jsonify({"products": df.head(per_page).to_dict(orient="records")})


# ================================================
# INTERACTION
# ================================================

@app.route("/api/interact", methods=["POST"])
def interact():
    data = request.json
    interactions.append({
        "user_id":    data["user_id"],
        "product_id": data["product_id"],
        "rating":     data["rating"]
    })
    return jsonify({"status": "ok"})


@app.route("/api/interactions")
def list_interactions():
    return jsonify({"count": len(interactions), "interactions": interactions})


# ================================================
# HELPER: apply category filter + exclude seen
# ================================================

def _apply_filters(rec_df, category_filter, exclude_ids):
    if category_filter:
        rec_df = rec_df[rec_df["category"] == category_filter]
    if exclude_ids:
        rec_df = rec_df[~rec_df["product_id"].isin(exclude_ids)]
    return rec_df


def _popular_fallback(top_n, category_filter, exclude_ids=None):
    df = products.copy()
    df = _apply_filters(df, category_filter, exclude_ids or [])
    sample_n = min(top_n, len(df))
    if sample_n == 0:
        return df.to_dict(orient="records")
    return df.sample(sample_n).to_dict(orient="records")


# ================================================
# 1. SVD COLLABORATIVE FILTERING
# ================================================

def train_svd():
    if len(interactions) < 5:
        return None, None, None
    df = pd.DataFrame(interactions)
    u_enc = LabelEncoder()
    i_enc = LabelEncoder()
    df["user"] = u_enc.fit_transform(df["user_id"])
    df["item"] = i_enc.fit_transform(df["product_id"])
    matrix = df.pivot_table(index="user", columns="item", values="rating").fillna(0)
    n_components = min(5, matrix.shape[1] - 1)
    if n_components < 1:
        return None, None, None
    svd = TruncatedSVD(n_components=n_components)
    latent = svd.fit_transform(matrix)
    reconstructed = np.dot(latent, svd.components_)
    return reconstructed, u_enc, i_enc


# ================================================
# 2. ITEM-BASED COLLABORATIVE FILTERING
# ================================================

def train_item_similarity():
    if len(interactions) < 3:
        return None, None, None, None
    df = pd.DataFrame(interactions)
    u_enc = LabelEncoder()
    i_enc = LabelEncoder()
    df["user"] = u_enc.fit_transform(df["user_id"])
    df["item"] = i_enc.fit_transform(df["product_id"])
    matrix = df.pivot_table(index="user", columns="item", values="rating").fillna(0)
    item_matrix = matrix.T.values
    if item_matrix.shape[0] < 2:
        return None, None, None, None
    sim = cosine_similarity(item_matrix)
    item_ids = i_enc.inverse_transform(range(item_matrix.shape[0]))
    return sim, u_enc, i_enc, item_ids


# ================================================
# 3. USER-BASED COLLABORATIVE FILTERING  ← NEW
#    "Users who bought what you bought also bought…"
# ================================================

def recommend_user_based(user_id, top_n, category_filter):
    """
    Build a user-user cosine similarity matrix from the interaction data.
    For the target user, find the K most similar users and recommend
    products they rated highly that the target user hasn't seen yet.
    """
    if len(interactions) < 3:
        return None, None

    df = pd.DataFrame(interactions)
    u_enc = LabelEncoder()
    i_enc = LabelEncoder()
    df["user"] = u_enc.fit_transform(df["user_id"])
    df["item"] = i_enc.fit_transform(df["product_id"])

    if user_id not in u_enc.classes_:
        return None, None

    matrix = df.pivot_table(index="user", columns="item", values="rating").fillna(0)

    # User-user cosine similarity
    user_sim = cosine_similarity(matrix.values)   # shape: (n_users, n_users)

    user_idx  = u_enc.transform([user_id])[0]
    sim_scores = user_sim[user_idx]               # similarity to every other user

    # Items the current user has already interacted with
    seen_items = df[df["user"] == user_idx]["item"].unique().tolist()
    seen_pids  = i_enc.inverse_transform(seen_items).tolist()

    # Weighted sum of ratings from similar users for unseen items
    n_items  = matrix.shape[1]
    scores   = np.zeros(n_items)
    sim_sums = np.zeros(n_items)

    for other_idx, sim in enumerate(sim_scores):
        if other_idx == user_idx or sim <= 0:
            continue
        other_ratings = matrix.iloc[other_idx].values
        scores   += sim * other_ratings
        sim_sums += sim * (other_ratings > 0).astype(float)

    # Avoid division by zero
    with np.errstate(invalid="ignore", divide="ignore"):
        weighted_scores = np.where(sim_sums > 0, scores / sim_sums, 0)

    # Zero out already-seen items
    for idx in seen_items:
        weighted_scores[idx] = 0

    # Map back to product_ids
    all_item_ids = i_enc.inverse_transform(range(n_items))
    scored_df = pd.DataFrame({"product_id": all_item_ids, "score": weighted_scores})
    scored_df = scored_df.sort_values("score", ascending=False)

    rec_pids = scored_df[scored_df["score"] > 0]["product_id"].tolist()

    rec_products = products[products["product_id"].isin(rec_pids)].copy()
    rec_products = _apply_filters(rec_products, category_filter, seen_pids)
    rec_products = rec_products.head(top_n)

    return rec_products, seen_pids


# ================================================
# 4. CATEGORY-BASED FILTERING  ← NEW
#    "You bought Electronics → here's more Electronics
#     you haven't seen, sorted by price similarity"
# ================================================

def recommend_category_based(user_id, top_n, category_filter):
    """
    Look at the categories the user has purchased from.
    Within each category they've engaged with, recommend the
    most price-similar unseen products (same category affinity).

    If category_filter is set, restrict to that category only.
    """
    user_interactions = [i for i in interactions if i["user_id"] == user_id]

    if not user_interactions:
        return None, None

    seen_pids = list({i["product_id"] for i in user_interactions})

    # Categories the user has bought from
    seen_products_df = products[products["product_id"].isin(seen_pids)]
    user_cats = seen_products_df["category"].unique().tolist()

    # If a specific category_filter is requested, use only that
    target_cats = [category_filter] if category_filter else user_cats

    # Average price per category the user bought in
    avg_price_per_cat = (
        seen_products_df.groupby("category")["price"].mean().to_dict()
    )

    candidates = products[
        products["category"].isin(target_cats) &
        ~products["product_id"].isin(seen_pids)
    ].copy()

    if candidates.empty:
        return candidates, seen_pids

    # Score each candidate: lower price distance from user's avg in that category = higher score
    def price_score(row):
        avg = avg_price_per_cat.get(row["category"], row["price"])
        max_price = products[products["category"] == row["category"]]["price"].max()
        # normalise distance to [0,1], invert so closer = higher score
        dist = abs(row["price"] - avg) / (max_price + 1)
        return 1 - dist

    candidates["score"] = candidates.apply(price_score, axis=1)

    # Sort: primary = category (preserve diversity), secondary = score
    candidates = candidates.sort_values(
        ["category", "score"], ascending=[True, False]
    )

    # Round-robin across categories for diversity
    result_rows = []
    cat_groups = {cat: grp.to_dict(orient="records")
                  for cat, grp in candidates.groupby("category")}
    cat_cycle = list(cat_groups.keys())
    pointers = {cat: 0 for cat in cat_cycle}
    i = 0
    while len(result_rows) < top_n:
        cat = cat_cycle[i % len(cat_cycle)]
        ptr = pointers[cat]
        group = cat_groups[cat]
        if ptr < len(group):
            result_rows.append(group[ptr])
            pointers[cat] = ptr + 1
        i += 1
        # break if we've exhausted all groups
        if all(pointers[c] >= len(cat_groups[c]) for c in cat_cycle):
            break

    return pd.DataFrame(result_rows), seen_pids


# ================================================
# 5. CONTENT-BASED FILTERING (existing, cleaned up)
# ================================================

def train_content_based():
    products_copy = products.copy()
    products_copy["text_features"] = (
        products_copy["product_name"] + " " + products_copy["brand"]
    )
    tfidf = TfidfVectorizer(stop_words="english", max_features=100)
    text_matrix = tfidf.fit_transform(products_copy["text_features"])

    ohe = OneHotEncoder(sparse_output=False)
    category_matrix = ohe.fit_transform(products_copy[["category"]])

    price_norm = (
        (products_copy["price"] - products_copy["price"].min())
        / (products_copy["price"].max() - products_copy["price"].min())
    )
    price_matrix = price_norm.values.reshape(-1, 1)

    feature_matrix = np.hstack([
        text_matrix.toarray(), category_matrix, price_matrix
    ])
    sim = cosine_similarity(feature_matrix)
    return sim, products_copy["product_id"].values


# ================================================
# RECOMMENDATION ENDPOINT
# ================================================

@app.route("/api/recommendations")
def recommend():
    user_id         = request.args.get("user_id")
    top_n           = int(request.args.get("top_n", 10))
    method          = request.args.get("method", "auto")
    category_filter = request.args.get("category")

    # ── Cold start (no user) ──────────────────────────────────────────────────
    if not user_id:
        return jsonify({
            "method": "popular",
            "recommendations": _popular_fallback(top_n, category_filter)
        })

    seen_pids = list({
        i["product_id"] for i in interactions if i["user_id"] == user_id
    })

    # ── USER-BASED COLLABORATIVE FILTERING ───────────────────────────────────
    if method == "user_based":
        rec_df, s_pids = recommend_user_based(user_id, top_n, category_filter)
        if rec_df is not None and not rec_df.empty:
            return jsonify({
                "method": "user_based",
                "recommendations": rec_df.to_dict(orient="records")
            })
        # fallback
        return jsonify({
            "method": "popular",
            "recommendations": _popular_fallback(top_n, category_filter, seen_pids)
        })

    # ── CATEGORY-BASED FILTERING ──────────────────────────────────────────────
    if method == "category_based":
        rec_df, s_pids = recommend_category_based(user_id, top_n, category_filter)
        if rec_df is not None and not rec_df.empty:
            return jsonify({
                "method": "category_based",
                "recommendations": rec_df.to_dict(orient="records")
            })
        # fallback: popular within those categories
        return jsonify({
            "method": "popular",
            "recommendations": _popular_fallback(top_n, category_filter, seen_pids)
        })

    # ── CONTENT-BASED ─────────────────────────────────────────────────────────
    if method == "content_based":
        user_interactions = [i for i in interactions if i["user_id"] == user_id]
        if not user_interactions:
            return jsonify({
                "method": "popular",
                "recommendations": _popular_fallback(top_n, category_filter, seen_pids)
            })
        sim, product_ids = train_content_based()
        scores = np.zeros(len(product_ids))
        interacted_ids = [i["product_id"] for i in user_interactions]
        for pid in interacted_ids:
            if pid in product_ids:
                idx = np.where(product_ids == pid)[0][0]
                scores += sim[idx]
        scores /= max(len(interacted_ids), 1)

        scored_df = pd.DataFrame({"product_id": product_ids, "score": scores})
        scored_df = scored_df.sort_values("score", ascending=False)
        rec_products = products[products["product_id"].isin(scored_df["product_id"])]
        rec_products = _apply_filters(rec_products, category_filter, interacted_ids)
        return jsonify({
            "method": "content_based",
            "recommendations": rec_products.head(top_n).to_dict(orient="records")
        })

    # ── ITEM-BASED COLLABORATIVE FILTERING ───────────────────────────────────
    if method == "item_based":
        sim, u_enc, i_enc, item_ids = train_item_similarity()
        if sim is None or user_id not in u_enc.classes_:
            return jsonify({
                "method": "popular",
                "recommendations": _popular_fallback(top_n, category_filter, seen_pids)
            })
        df = pd.DataFrame(interactions)
        df["user"] = u_enc.transform(df["user_id"])
        df["item"] = i_enc.transform(df["product_id"])
        matrix    = df.pivot_table(index="user", columns="item", values="rating").fillna(0)
        user_idx  = u_enc.transform([user_id])[0]
        user_vec  = (
            matrix.loc[user_idx].reindex(range(sim.shape[0]), fill_value=0).values
            if user_idx in matrix.index else np.zeros(sim.shape[0])
        )
        agg_scores    = sim.dot(user_vec)
        scored_df     = pd.DataFrame({"product_id": item_ids, "score": agg_scores})
        scored_df     = scored_df.sort_values("score", ascending=False)
        rec_products  = products[products["product_id"].isin(scored_df["product_id"])]
        rec_products  = _apply_filters(rec_products, category_filter, seen_pids)
        return jsonify({
            "method": "item_based",
            "recommendations": rec_products.head(top_n).to_dict(orient="records")
        })

    # ── SVD ───────────────────────────────────────────────────────────────────
    if method == "svd":
        model, u_enc, i_enc = train_svd()
        if model is None or user_id not in u_enc.classes_:
            return jsonify({
                "method": "popular",
                "recommendations": _popular_fallback(top_n, category_filter, seen_pids)
            })
        user_index    = u_enc.transform([user_id])[0]
        scores        = model[user_index]
        item_ids      = i_enc.inverse_transform(range(len(scores)))
        scored_df     = pd.DataFrame({"product_id": item_ids, "score": scores})
        scored_df     = scored_df.sort_values("score", ascending=False)
        rec_products  = products[products["product_id"].isin(scored_df.head(top_n)["product_id"])]
        rec_products  = _apply_filters(rec_products, category_filter, seen_pids)
        return jsonify({
            "method": "svd",
            "recommendations": rec_products.to_dict(orient="records")
        })

    # ── AUTO MODE ─────────────────────────────────────────────────────────────
    # Priority: user_based (if enough data) → category_based → item_based → popular
    if method == "auto":
        # Try user-based first
        rec_df, s_pids = recommend_user_based(user_id, top_n, category_filter)
        if rec_df is not None and not rec_df.empty:
            return jsonify({
                "method": "user_based",
                "recommendations": rec_df.to_dict(orient="records")
            })

        # Try category-based next
        rec_df, s_pids = recommend_category_based(user_id, top_n, category_filter)
        if rec_df is not None and not rec_df.empty:
            return jsonify({
                "method": "category_based",
                "recommendations": rec_df.to_dict(orient="records")
            })

        # Try item-based
        sim, u_enc, i_enc, item_ids = train_item_similarity()
        if sim is not None and user_id in u_enc.classes_:
            df       = pd.DataFrame(interactions)
            df["user"] = u_enc.transform(df["user_id"])
            df["item"] = i_enc.transform(df["product_id"])
            matrix   = df.pivot_table(index="user", columns="item", values="rating").fillna(0)
            user_idx = u_enc.transform([user_id])[0]
            user_vec = np.zeros(sim.shape[0])
            item_list = list(item_ids)
            for pid in seen_pids:
                try:
                    user_vec[item_list.index(pid)] = 1
                except ValueError:
                    pass
            agg_scores   = sim.dot(user_vec)
            scored_df    = pd.DataFrame({"product_id": item_ids, "score": agg_scores})
            scored_df    = scored_df.sort_values("score", ascending=False)
            rec_products = products[products["product_id"].isin(scored_df["product_id"])]
            rec_products = _apply_filters(rec_products, category_filter, seen_pids)
            if not rec_products.empty:
                return jsonify({
                    "method": "item_based",
                    "recommendations": rec_products.head(top_n).to_dict(orient="records")
                })

        # Final fallback
        return jsonify({
            "method": "popular",
            "recommendations": _popular_fallback(top_n, category_filter, seen_pids)
        })

    # Unknown method → popular
    return jsonify({
        "method": "popular",
        "recommendations": _popular_fallback(top_n, category_filter, seen_pids)
    })


if __name__ == "__main__":
    app.run(debug=True)

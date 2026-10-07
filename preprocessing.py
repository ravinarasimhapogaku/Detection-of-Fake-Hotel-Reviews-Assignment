# %%
import re
import hashlib
import unicodedata
import numpy as np
import pandas as pd

SEED = 42
RAW_PATH = "all_data.csv"
CLEAN_PATH = "english_reviews.csv"

# %%
df = pd.read_csv(RAW_PATH)

print(df.shape)
print(df.dtypes)
df.head()

# %% [markdown]
# # Review per language and class

# %%
print(df["Review_Language"].value_counts())

print(pd.crosstab(df["Review_Language"], df["source"], margins=True))

print(pd.crosstab(df["Review_Language"], df["Sentiment"]))

# %% [markdown]
# #  missing values and meta data columns.

# %%
print(df.isna().sum())

print(pd.crosstab(df["Prompt_Language"].isna(), df["source"]))

# Missingness flags differ by class
print(pd.crosstab(df["source"], [df["na_up_review"], df["na_down_review"]]))

# %%
eng_preview = df[df["Review_Language"] == "English"]
print(eng_preview.shape)
print(pd.crosstab(eng_preview["source"], eng_preview["Sentiment"]))
print(eng_preview["Prompt_Language"].value_counts())

# %% [markdown]
# # keeping english

# %%
eng = df[df["Review_Language"] == "English"].copy()

eng = eng.reset_index().rename(columns={"index": "id"})
eng = eng[["id", "Upside_Review", "Downside_Review", "Hotel Name", "City Name", "source", "Sentiment"]]
eng = eng.rename(columns={"source": "label", "Sentiment": "sentiment"})

# sanity checks
print(eng.shape)
print(eng["label"].value_counts())
print(pd.crosstab(eng["label"], eng["sentiment"]))
print(eng["id"].is_unique)

# %% [markdown]
# # building review text

# %%
eng["Upside_Review"]   = eng["Upside_Review"].fillna("")
eng["Downside_Review"] = eng["Downside_Review"].fillna("")
eng["text"] = (eng["Upside_Review"] + " " + eng["Downside_Review"]).str.strip()

eng["text"] = (eng["text"]
              .apply(lambda t: unicodedata.normalize("NFKC", t))
              .str.replace("’", "'", regex=False))

def mask_entities(row):
    t, name, city = row["text"], str(row["Hotel Name"]), str(row["City Name"])
    variants = {name,
                re.sub(r"^(the )?hotel ", "", name, flags=re.I),
                re.sub(r" hotel$", "", name, flags=re.I)}
    for v in sorted(variants, key=len, reverse=True):
        if len(v) > 3:
            t = re.sub(re.escape(v), " hotelname ", t, flags=re.I)
    t = re.sub(r"\b" + re.escape(city) + r"\b", " cityname ", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()

eng["text"] = eng.apply(mask_entities, axis=1)

print(eng.groupby("label")["text"].apply(lambda s: s.str.contains(r"\bhotelname\b").mean()).round(3))
print(eng.groupby("label")["text"].apply(lambda s: s.str.contains(r"\bcityname\b").mean()).round(3))

eng = eng[["id", "text", "label", "sentiment"]]

# %% [markdown]
# # Data quality check

# %%
norm = eng["text"].str.lower().str.replace(r"[^a-z ]", "", regex=True).str.split().str.join(" ")
print("exact duplicates:", eng["text"].duplicated().sum())
print("duplicates after normalisation:", norm.duplicated().sum())
print(eng.loc[norm.duplicated(keep=False), ["id", "label", "text"]])

print("mojibake:", eng["text"].str.contains("â€|Ã").sum())
print("HTML entities:", eng["text"].str.contains(r"&[a-z]+;|&#").sum())

nonlat = eng["text"].apply(lambda t: sum(c.isalpha() and ord(c) > 591 for c in t)
                                    / max(1, sum(c.isalpha() for c in t)))
print("reviews with >10% non-Latin letters:", (nonlat > 0.1).sum())

# %% [markdown]
# # Sanity check for text and review lengths

# %%
print((eng["text"] == "").sum())
print(eng["text"].str.contains(r"\bnan\b").sum())
print(eng["text"].str.split().str.len().describe())
print(eng.groupby("label")["text"].apply(lambda s: s.str.split().str.len().mean()))
print(eng.sample(3, random_state=1)["text"].tolist())

# %% [markdown]
# # stratified split

# %%
from sklearn.model_selection import train_test_split

strat_key = eng["label"].astype(str) + "_" + eng["sentiment"]

train_df, test_df = train_test_split(
    eng,
    test_size=0.25,
    stratify=strat_key,
    random_state=SEED,
)

# %% [markdown]
# # checksum and saving

# %%
eng["split"] = "train"
eng.loc[test_df.index, "split"] = "test"

eng.to_csv(CLEAN_PATH, index=False)
print("data checksum (MD5):", hashlib.md5(open(CLEAN_PATH, "rb").read()).hexdigest())

# %% [markdown]
# # split checking 

# %%
print(eng["split"].value_counts())
print(pd.crosstab([eng["split"], eng["label"]], eng["sentiment"]))
print(set(train_df["id"]) & set(test_df["id"]))

check = pd.read_csv(CLEAN_PATH)
print(check.shape, check.columns.tolist())

# %% [markdown]
# Vectorizer

# %%
from sklearn.feature_extraction.text import CountVectorizer

TOKEN_PATTERN = r"(?u)\b[^\W\d_]{2,}\b"

def make_vectorizer(min_df=0.01):
    return CountVectorizer(
        lowercase=True,
        token_pattern=TOKEN_PATTERN,
        stop_words=None,
        ngram_range=(1, 2),
        min_df=min_df
    )

# %% [markdown]
# 

# %%
data  = pd.read_csv(CLEAN_PATH)
train = data[data["split"] == "train"]
test  = data[data["split"] == "test"]

vec = make_vectorizer()
X_train = vec.fit_transform(train["text"])
X_test  = vec.transform(test["text"])
y_train, y_test = train["label"].values, test["label"].values

vocab = vec.get_feature_names_out()
pd.Series(vocab, name="term").to_csv("vocabulary.csv", index=False)

def load_and_preprocess():
    data = pd.read_csv(CLEAN_PATH)

    train = data[data["split"] == "train"]
    test = data[data["split"] == "test"]

    vec = make_vectorizer()

    X_train = vec.fit_transform(train["text"])
    X_test = vec.transform(test["text"])

    y_train = train["label"].values
    y_test = test["label"].values

    vocab = vec.get_feature_names_out()

    pd.Series(vocab, name="term").to_csv(
        "vocabulary.csv",
        index=False
    )

    return X_train, X_test, y_train, y_test, vec

# load_and_preprocess()

# %% [markdown]
# # vocab and sparsity threshold check.

# %%
print(X_train.shape, X_test.shape)
print(sum(" " in t for t in vocab), "bigrams")
print(vocab[:15])

print((X_test.sum(axis=1) == 0).sum(), "test reviews with no known terms")

for md in [1, 0.005, 0.01, 0.02, 0.05]:
    v = CountVectorizer(lowercase=True, token_pattern=TOKEN_PATTERN,
                        ngram_range=(1, 2), min_df=md).fit(train["text"])
    print(md, len(v.vocabulary_))

# %%




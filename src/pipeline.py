import pandas as pd, numpy as np
import numpy as np, pandas as pd, os
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
SEED = 42


def build_features(train, test, blocks):
    full = pd.concat([train.drop(columns="Transported"), test], ignore_index=True)
    grp = full["PassengerId"].str.split("_").str[0].astype(int)
    df = full[["HomePlanet", "CryoSleep", "Destination", "VIP", "Age"] + SPEND].copy()
    for c in ["CryoSleep", "VIP"]:
        df[c] = full[c].map({True: "True", False: "False"})

    if "groupfill" in blocks:
        # "first" returns the first non-missing value in each group.
        df["HomePlanet"] = df["HomePlanet"].fillna(df.groupby(grp)["HomePlanet"].transform("first"))

    if "rules" in blocks:
        # Order matters: repair CryoSleep first (anyone who spent money was not in cryosleep),
        # then use it to zero the missing spend of cryosleepers.
        df.loc[df["CryoSleep"].isna() & (df[SPEND].sum(axis=1) > 0), "CryoSleep"] = "False"
        cryo = df["CryoSleep"] == "True"
        for c in SPEND:
            df.loc[cryo & df[c].isna(), c] = 0.0
    
        # Put this right AFTER the existing `rules` block.
    if "rules2" in blocks:
        # Why: among passengers with all five spends observed and zero, 85.1% were in
        # cryosleep, yet the mode imputer fills a missing CryoSleep with False for them.
        # Uses only other features, never Transported, so there is no label leakage.
        obs_zero = df[SPEND].notna().all(axis=1) & (df[SPEND].sum(axis=1) == 0)
        df.loc[df["CryoSleep"].isna() & obs_zero, "CryoSleep"] = "True"

        # Split the old 'spend' block into three single-idea blocks to find which part hurt Logistic.
    if {"nospend", "totspend"} & set(blocks):
        total = df[SPEND].sum(axis=1)          # computed BEFORE any log transform
        if "totspend" in blocks:
            df["TotalSpend"] = total           # raw total: tests the heavy-tail hypothesis
        if "nospend" in blocks:
            df["NoSpend"] = (total == 0).astype(int)
    if "logspend" in blocks:
        for c in SPEND:                        # log of each column: tests the reshaping hypothesis
            df[c] = np.log1p(df[c])

    if "age" in blocks:
        df["IsChild"] = (df["Age"] <= 12).astype(int)

    if "cabin" in blocks:
        cab = full["Cabin"].str.split("/", expand=True)
        df["Deck"], df["CabinNum"], df["Side"] = cab[0], pd.to_numeric(cab[1]), cab[2]

    # Put this right AFTER the existing `cabin` block (it needs CabinNum).
    if "cabinbin" in blocks:
        # Why: the transported rate is a wave across cabin numbers, not a slope. One-hot
        # bins of fixed width 100 (chosen up front, not tuned to the labels) let Logistic
        # fit a separate level per range. CabinNum itself stays as a numeric feature.
        if "CabinNum" not in df:
            cab = full["Cabin"].str.split("/", expand=True)
            df["Deck"], df["CabinNum"], df["Side"] = cab[0], pd.to_numeric(cab[1]), cab[2]
        b = df["CabinNum"] // 100
        df["CabinBin"] = b.map(lambda v: f"bin{int(v)}" if pd.notna(v) else np.nan)

    if "group" in blocks:
        df["GroupSize"] = grp.map(grp.value_counts())
        df["IsAlone"] = (df["GroupSize"] == 1).astype(int)

    if "family" in blocks:
        sur = full["Name"].str.split().str[-1]
        df["FamilySize"] = sur.map(sur.value_counts()).fillna(1)   # missing name -> assume alone

    n = len(train)
    return df.iloc[:n].reset_index(drop=True), df.iloc[n:].reset_index(drop=True)

def make_preprocessor(scale):
    # Numeric: median is robust to the heavy skew in spending. Scaling only for logistic
    # regression, because its penalty depends on feature scale and trees don't care.
    num = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num), make_column_selector(dtype_include="number")),
        # Categorical: most frequent value; handle_unknown="ignore" avoids crashes on rare
        # levels that appear in only one fold or only in the test set.
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
         make_column_selector(dtype_exclude="number")),
    ])

def make_models():
    return {
        "Logistic": Pipeline([("prep", make_preprocessor(True)),
                              ("m", LogisticRegression(max_iter=3000))]),
        "RF": Pipeline([("prep", make_preprocessor(False)),
                        ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                     random_state=SEED, n_jobs=1))]),
    }

def cv_accuracy(model, X, grouped=True, n_repeats=3):
    scores = []
    for r in range(n_repeats):
        if grouped:
            splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED + r)
                          .split(X, y, groups))
        else:
            splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(X, y))
        scores.extend(cross_val_score(model, X, y, cv=splits, scoring="accuracy", n_jobs=-1))
    return np.mean(scores), np.std(scores)
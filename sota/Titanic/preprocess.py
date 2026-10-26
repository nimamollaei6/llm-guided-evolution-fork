# Note: Code from this file is taken from the Titanic Tutorial found at https://www.kaggle.com/code/soham1024/titanic-data-science-eda-with-meme-solution

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

if __name__ == '__main__':
    # =========================
    # 1. LOAD DATA
    # =========================

    df = pd.read_csv("data/train.csv")


    # =========================
    # 2. FEATURE ENGINEERING
    # =========================

    # Extract titles such as Mr, Mrs, Miss, Master
    df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.")

    common_titles = ["Mr", "Miss", "Mrs", "Master"]
    df["Title"] = df["Title"].where(
        df["Title"].isin(common_titles),
        "Rare"
    )

    # Family-related features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)


    # =========================
    # 3. SELECT FEATURES
    # =========================

    features = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "Title",
        "FamilySize",
        "IsAlone"
    ]

    X = df[features]
    y = df["Survived"]


    # =========================
    # 5. SHARED PREPROCESSING
    # =========================

    numeric_features = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "FamilySize",
        "IsAlone"
    ]

    categorical_features = [
        "Sex",
        "Embarked",
        "Title"
    ]


    numeric_processor = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])


    categorical_processor = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])


    preprocessor = ColumnTransformer([
        ("num", numeric_processor, numeric_features),
        ("cat", categorical_processor, categorical_features)
    ])
    
    X_processed = preprocessor.fit_transform(X)

    processed_df = pd.DataFrame(
        X_processed,
        columns=preprocessor.get_feature_names_out()
    )
    processed_df["Survived"] = y.values

    print(processed_df.head())
    processed_df.to_csv("data/processed_train.csv", index=False)
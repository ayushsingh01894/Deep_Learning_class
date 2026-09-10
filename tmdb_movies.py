import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("tmdb_5000_movies.csv")
print(df.head())
print("\nNull Values:")
print(df.isnull().sum())

df["runtime"] = df["runtime"].fillna(df["runtime"].median())
df["release_date"] = pd.to_datetime(
    df["release_date"],
    errors="coerce"
)
df["release_date"] = df["release_date"].fillna(
    df["release_date"].mode()[0]
)
df["release_year"] = df["release_date"].dt.year
df = df.drop_duplicates()
df = df.drop(columns=[
    "homepage",
    "tagline",
    "overview",
    "release_date",
    "genres",
    "keywords",
    "production_companies",
    "production_countries",
    "spoken_languages",
    "title",
    "original_title"
])
df = pd.get_dummies(
    df,
    columns=["original_language", "status"],
    drop_first=True
)
X = df.drop("revenue", axis=1)
y = df["revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nR2 Score:", r2_score(y_test, y_pred))
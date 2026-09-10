import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("tmdb_5000_movies.csv")
print(df.head())

# 2. Dataset Shape
print(df.shape)

# 3. Dataset Information
print(df.info())


# 4. Null Values
print(df.isnull().sum())


# 5. Handle Null Values
df["runtime"] = df["runtime"].fillna(df["runtime"].median())
df["release_date"] = pd.to_datetime(
    df["release_date"],
    errors="coerce"
)

df["release_date"] = df["release_date"].fillna(
    df["release_date"].mode()[0]
)
print(df.isnull().sum())


# 6. Duplicate Values
print(df.duplicated().sum())

# 7. Remove Duplicates
df = df.drop_duplicates()
print(df.shape)


# 8. Convert Release Date

df["release_year"] = df["release_date"].dt.year
print(df[["release_date", "release_year"]].head())


# 9. Create Target Variable

df["success"] = np.where(
    df["revenue"] > 100000000,
    1,
    0
)

print(df["success"].value_counts())


# 11. Remove Unnecessary Columns

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
    "original_title",
    "revenue"
])

print(df.columns)


# 12. Encoding
df = pd.get_dummies(
    df,
    columns=[
        "original_language",
        "status"
    ],
    drop_first=True
)

print(df.head())


# 13. Check Data Types

print(df.dtypes)


# 14. Create X and y

X = df.drop(
    "success",
    axis=1
)
y = df["success"]

print(X.head())
print(y.head())


# 15. Check Missing Values

print(X.isnull().sum().sum())


# 16. Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 18. Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()

# 20. Train Model

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)

print(y_pred[:20])

print(y_test.values[:20])


accuracy = accuracy_score(
    y_test,
    y_pred
)

print(accuracy)

# grip search cv
# randomized seach cv

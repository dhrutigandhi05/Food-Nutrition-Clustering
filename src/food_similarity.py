import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df = pd.read_csv("data/processed/foods_cleaned.csv")
features = ["Protein", "Fat", "Carbs", "Fibre"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

def search_foods(query):
    matches = df[
        df["Food_Description_EN"].str.contains(query, case=False, regex=False, na=False)
    ]

    return matches[["Food_Code", "Food_Description_EN"]]

def find_similar_foods(food_code, n=5):
    matches = df.index[df["Food_Code"] == food_code]

    if len(matches) == 0:
        return pd.DataFrame()

    food_index = matches[0]
    cluster = df.iloc[food_index]["Cluster"]

    candidates = df[
        (df["Cluster"] == cluster) &
        (df.index != food_index)
    ].copy()

    selected_point = X_pca[food_index] # euclidean distance calculation

    candidates["Distance"] = np.sqrt(
        (candidates["PC1"] - selected_point[0]) ** 2
        + (candidates["PC2"] - selected_point[1]) ** 2
    )

    results = candidates.nsmallest(n, "Distance").copy()
    results["Distance"] = results["Distance"].round(3)

    return results[
        [
            "Food_Description_EN",
            "Protein",
            "Fat",
            "Carbs",
            "Fibre",
            "Calories",
            "Cluster",
            "Distance"
        ]
    ].rename(columns={"Food_Description_EN": "Food"})

if __name__ == "__main__":
    print(
        f"PCA variance explained: "
        f"{pca.explained_variance_ratio_.sum():.2%}"
    )

    query = input("Search for a food: ")
    matches = search_foods(query)

    if matches.empty:
        print("No matching foods found.")
    else:
        print("\nMatching foods:")
        print(matches.head(20).to_string(index=False))

        try:
            food_code = int(input("\nEnter Food Code: "))
            results = find_similar_foods(food_code)

            if results.empty:
                print("Invalid food code.")
            else:
                print("Most similar foods:")
                print(results.to_string(index=False))

        except ValueError:
            print("Please enter a valid food code.")
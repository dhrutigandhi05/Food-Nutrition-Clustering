# Food Nutrition Clustering & Recommendation System
An unsupervised machine learning project that groups foods based on their nutritional profiles and recommends similar foods using **K-Means clustering, Principal Component Analysis (PCA), and Euclidean distance**.

## Overview

- **K-Means Clustering:** Groups foods into 5 clusters based on standardized nutritional features. The optimal number of clusters was selected using the elbow method and silhouette scores.
- **PCA:** Reduces the nutritional features to two principal components for cluster visualization.
- **Food Recommendations:** Finds the 5 most nutritionally similar foods within the selected food's cluster using Euclidean distance.
- **Model Persistence:** Saves the trained K-Means model and StandardScaler using Joblib to avoid retraining during recommendations.

## Technologies

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib, Jupyter Notebook

## How It Works

1. Nutritional features are standardized using `StandardScaler`.
2. K-Means assigns each food to one of 5 nutritional clusters.
3. The user searches for and selects a food.
4. Euclidean distances are calculated between the selected food and other foods in the same cluster.
5. The 5 closest foods are returned along with their nutritional information and similarity distances.
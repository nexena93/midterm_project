import sys

import pandas as pd
import requests

COLUMNS = [
    "id", "diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean",
    "area_mean", "smoothness_mean", "compactness_mean",
    "concavity_mean", "concave_points_mean", "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se",
    "area_se", "smoothness_se", "compactness_se",
    "concavity_se", "concave_points_se", "symmetry_se",
    "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst",
    "area_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave_points_worst", "symmetry_worst",
    "fractal_dimension_worst",
]

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9696"

df = pd.read_csv("/workspaces/midterm_project/wdbc.data", header=None, names=COLUMNS)
for label in ["M", "B"]:
    row = df[df["diagnosis"] == label].drop(columns=["id", "diagnosis"]).iloc[0]
    response = requests.post(f"{url}/predict", json=row.to_dict())
    print(f"True: {label} ->", response.status_code, response.json())
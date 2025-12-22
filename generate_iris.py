import pandas as pd
from sklearn.datasets import load_iris

# Load Iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['species'] = data.target

# Map target integers to names for better visualization
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

output_path = "datasets/iris.csv"
df.to_csv(output_path, index=False)
print(f"Created {output_path} with shape {df.shape}")

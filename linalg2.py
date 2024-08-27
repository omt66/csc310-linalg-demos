import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
species = iris.target_names

# Convert to a Pandas DataFrame for easier plotting with Seaborn
import pandas as pd
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = [species[i] for i in y]

# Pairplot to visualize the relationships between features
sns.pairplot(df, hue='species', markers=["o", "s", "D"])
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
plt.show()

# Apply PCA to reduce the dataset to 2 dimensions for a scatter plot
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Convert the PCA result into a DataFrame
df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
df_pca['species'] = df['species']

# Scatter plot of the first two principal components
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='species', data=df_pca, palette='Set2', s=100)
plt.title('PCA of Iris Dataset')
plt.show()
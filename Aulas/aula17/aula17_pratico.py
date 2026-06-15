import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

DATASETS = {
    "Aggregation": ("http://cs.joensuu.fi/sipu/datasets/Aggregation.txt", 7),
    "D31":         ("http://cs.joensuu.fi/sipu/datasets/D31.txt",         31),
    "Path-based":  ("http://cs.joensuu.fi/sipu/datasets/pathbased.txt",   3),
    "Flame":       ("http://cs.joensuu.fi/sipu/datasets/flame.txt",       2),
}


def load_dataset(url):
    with urllib.request.urlopen(url) as response:
        data = response.read().decode("utf-8")
    rows = [line.split() for line in data.strip().splitlines() if line.strip()]
    arr = np.array([[float(r[0]), float(r[1])] for r in rows])
    return arr


def plot_clusters(ax, X, labels, title):
    unique = np.unique(labels)
    cmap = plt.colormaps["tab20"].resampled(len(unique))
    for i, lbl in enumerate(unique):
        mask = labels == lbl
        ax.scatter(X[mask, 0], X[mask, 1], s=8, color=cmap(i), linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


fig, axes = plt.subplots(len(DATASETS), 2, figsize=(10, 16))
fig.suptitle("K-means vs Single Linkage", fontsize=13, fontweight="bold")
axes[0, 0].set_title("K-means", fontsize=11, fontweight="bold")
axes[0, 1].set_title("Single Linkage", fontsize=11, fontweight="bold")

for row, (name, (url, k)) in enumerate(DATASETS.items()):
    print(f"Processando {name} (k={k})...")
    X = load_dataset(url)
    Xs = StandardScaler().fit_transform(X)

    km_labels = KMeans(n_clusters=k, n_init="auto", random_state=42).fit_predict(Xs)
    sl_labels = AgglomerativeClustering(n_clusters=k, linkage="single").fit_predict(Xs)

    plot_clusters(axes[row, 0], X, km_labels, f"{name} - K-means (k={k})")
    plot_clusters(axes[row, 1], X, sl_labels, f"{name} - Single Linkage (k={k})")

plt.tight_layout()
plt.savefig("clusters.png", dpi=150, bbox_inches="tight")
plt.show()
print("Salvo em clusters.png")

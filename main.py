import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import dataset_generator

# 1. k-means clustering (vectorized)
class KMeansModel:
    def __init__(self, k=2, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.cost_history = []
        
    def fit(self, x):
        n_samples, n_features = x.shape
        self.centroids = x[np.random.choice(n_samples, self.k, replace=False)]
        
        for i in range(self.max_iters):
            # euclidean distance computation
            distances = np.sum(x**2, axis=1)[:, np.newaxis] + np.sum(self.centroids**2, axis=1) - 2 * np.dot(x, self.centroids.T)
            labels = np.argmin(distances, axis=1)
            
            self.cost_history.append(np.sum(np.min(distances, axis=1)))
            new_centroids = np.zeros((self.k, n_features))
            
            for j in range(self.k):
                cluster_points = x[labels == j]
                if len(cluster_points) > 0:
                    new_centroids[j] = np.mean(cluster_points, axis=0)
                else:
                    new_centroids[j] = x[np.random.choice(n_samples)]
                    
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            self.centroids = new_centroids
        return labels

# 2. gaussian naive bayes
class GaussianNBModel:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}
        
    def fit(self, x, y):
        self.classes = np.unique(y)
        for c in self.classes:
            x_c = x[y == c]
            self.priors[c] = x_c.shape[0] / x.shape[0]
            self.mean[c] = np.mean(x_c, axis=0)
            self.var[c] = np.var(x_c, axis=0) + 1e-9 
            
    def predict(self, x):
        log_probs = np.zeros((x.shape[0], len(self.classes)))
        for idx, c in enumerate(self.classes):
            mean, var = self.mean[c], self.var[c]
            num = -((x - mean) ** 2) / (2 * var)
            den = 0.5 * np.log(2 * np.pi * var)
            log_probs[:, idx] = np.log(self.priors[c]) + np.sum(num - den, axis=1)
        return self.classes[np.argmax(log_probs, axis=1)]

# 3. c4.5 decision tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
class C45Tree:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        
    def _entropy(self, y):
        probs = np.bincount(y) / len(y)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))
        
    def _gain_ratio(self, y, y_left, y_right):
        p = len(y_left) / len(y)
        info_gain = self._entropy(y) - (p * self._entropy(y_left) + (1 - p) * self._entropy(y_right))
        split_info = -p * np.log2(p + 1e-9) - (1 - p) * np.log2((1 - p) + 1e-9)
        return info_gain / (split_info + 1e-9)
        
    def _best_split(self, x, y):
        best_gain = -1
        split_idx, split_thresh = None, None
        
        for feat_idx in range(x.shape[1]):
            for thresh in np.unique(x[:, feat_idx]):
                left_mask = x[:, feat_idx] <= thresh
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                gain = self._gain_ratio(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain, split_idx, split_thresh = gain, feat_idx, thresh
        return split_idx, split_thresh
        
    def _build_tree(self, x, y, depth=0):
        if len(np.unique(y)) == 1 or len(y) < self.min_samples_split or depth >= self.max_depth:
            return Node(value=np.bincount(y).argmax())
            
        feat_idx, thresh = self._best_split(x, y)
        if feat_idx is None:
            return Node(value=np.bincount(y).argmax())
            
        left_mask = x[:, feat_idx] <= thresh
        left = self._build_tree(x[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(x[~left_mask], y[~left_mask], depth + 1)
        return Node(feature=feat_idx, threshold=thresh, left=left, right=right)
        
    def fit(self, x, y):
        self.root = self._build_tree(x, y)
        
    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)
        
    def predict(self, x):
        return np.array([self._traverse(sample, self.root) for sample in x])

# execution and chart generation
def run_experiments():
    datasets = dataset_generator.save_datasets()
    
    # kmeans initialization sensitivity plot
    x_kmeans, _ = datasets["low_noise"]
    plt.figure(figsize=(10, 5))
    for i in range(20):
        kmeans = KMeansModel(k=2, max_iters=50)
        kmeans.fit(x_kmeans)
        plt.plot(kmeans.cost_history, alpha=0.5)
    plt.title("k-means convergence over 20 random initializations")
    plt.xlabel("iterations")
    plt.ylabel("clustering cost")
    plt.grid(True)
    plt.show()
    
    # decision tree depth vs accuracy plot
    x_dt, y_dt = datasets["high_dim"]
    x_train, x_val, y_train, y_val = train_test_split(x_dt, y_dt, test_size=0.2, random_state=dataset_generator.SEED)
    
    depths, train_acc, val_acc = [1, 3, 5, 10, 15, 20], [], []
    for d in depths:
        dt = C45Tree(max_depth=d)
        dt.fit(x_train, y_train)
        train_acc.append(accuracy_score(y_train, dt.predict(x_train)))
        val_acc.append(accuracy_score(y_val, dt.predict(x_val)))
        print(f"depth {d} calculated.")
        
    plt.figure(figsize=(10, 5))
    plt.plot(depths, train_acc, label="train acc", marker='o')
    plt.plot(depths, val_acc, label="val acc", marker='x')
    plt.title("c4.5 decision tree: depth vs accuracy")
    plt.xlabel("maximum tree depth")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_experiments()

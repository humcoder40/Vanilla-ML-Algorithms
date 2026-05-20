import numpy as np
import pandas as pd

# last 3 digits of my roll number  (python is not allowing me to write 002,
#  it gives error on that.So I am writing 2 instead of 002)
SEED = 2
np.random.seed(SEED)

def generate_base_features(n_samples, n_informative, n_noisy):
    info = np.random.normal(0, 1, (n_samples, n_informative))
    noise = np.random.normal(0, 1, (n_samples, n_noisy))
    return info, noise

def generate_low_noise_dataset():
    # n>=1000, d>=15 constraints
    n, d_info, d_noise = 1500, 5, 10
    info, noise = generate_base_features(n, d_info, d_noise)
    y = (np.sum(info, axis=1) > 0).astype(int)
    x = np.hstack((info, noise))
    return x, y

def generate_high_noise_dataset():
    n, d_info, d_noise = 1500, 5, 10
    info = np.random.normal(0, 1, (n, d_info))
    noise = np.random.normal(0, 5, (n, d_noise))
    y = (np.sum(info, axis=1) > 0).astype(int)
    
    # flip 30% of labels for noise
    flip_indices = np.random.choice(n, size=int(0.3 * n), replace=False)
    y[flip_indices] = 1 - y[flip_indices]
    x = np.hstack((info, noise))
    return x, y

def generate_high_dimensional_dataset():
    # d>=50, n>=5000 requirement
    n, d_info, d_noise = 5000, 10, 45 
    info, noise = generate_base_features(n, d_info, d_noise)
    y = (np.sin(info[:, 0]) + np.cos(info[:, 1]) > 0).astype(int)
    x = np.hstack((info, noise))
    return x, y

def generate_kmeans_adversarial():
    # concentric circles where kmeans fails
    n = 1500
    theta = np.random.uniform(0, 2*np.pi, n)
    r = np.where(np.random.rand(n) > 0.5, np.random.normal(2, 0.2, n), np.random.normal(6, 0.2, n))
    
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    info = np.column_stack((x1, x2, np.random.normal(0, 1, (n, 3))))
    noise = np.random.normal(0, 1, (n, 10))
    
    x = np.hstack((info, noise))
    y = (r > 4).astype(int)
    return x, y

def generate_gnb_correlated():
    # correlated features violating naive bayes assumption
    n = 1500
    x1 = np.random.normal(0, 1, n)
    x2 = x1 + np.random.normal(0, 0.1, n) 
    
    info_rest = np.random.normal(0, 1, (n, 3))
    noise = np.random.normal(0, 1, (n, 10))
    
    x = np.column_stack((x1, x2, info_rest, noise))
    y = (x1 + x2 > 0).astype(int)
    return x, y

def save_datasets():
    return {
        "low_noise": generate_low_noise_dataset(),
        "high_noise": generate_high_noise_dataset(),
        "high_dim": generate_high_dimensional_dataset(),
        "kmeans_adv": generate_kmeans_adversarial(),
        "gnb_corr": generate_gnb_correlated()
    }

if __name__ == "__main__":
    data = save_datasets()
    print("datasets generated successfully.")

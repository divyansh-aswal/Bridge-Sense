import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Load the features data
features_df = pd.read_csv("damage_sensitive_features.csv")

# Assuming we have columns 'Spectral_Centroid' and 'Low_Frequency_Energy'
# Also assuming we have a 'Time' column or we can create one based on the index
features_df['Time'] = features_df.index

# Function to perform DBSCAN and plot
def perform_dbscan_and_plot(feature_column):
    # Extract the feature
    X = features_df[[feature_column]].values
    
    # Perform DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(X)
    
    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.scatter(features_df['Time'], X, c=clusters, cmap='viridis', s=50)
    plt.xlabel('Time')
    plt.ylabel(feature_column)
    plt.title(f'DBSCAN Clustering on {feature_column}')
    plt.colorbar(label='Cluster ID')
    plt.show()

# Perform and plot for Spectral Centroid
perform_dbscan_and_plot('Spectral_Centroid')

# Perform and plot for Low Frequency Energy
perform_dbscan_and_plot('Low_Frequency_Energy')


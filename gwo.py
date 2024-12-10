import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler

class GreyWolfOptimizer:
    def __init__(self, image: np.ndarray, num_clusters: int = 3, 
                 max_iterations: int = 30, population_size: int = 20):
        """
        Initialize Grey Wolf Optimizer for image clustering
        
        :param image: Input image as numpy array
        :param num_clusters: Number of clusters to create
        :param max_iterations: Maximum number of iterations
        :param population_size: Number of wolf agents
        """
        # Resize image if too large
        max_pixel_count = 50000
        if image.size > max_pixel_count:
            scale_factor = np.sqrt(max_pixel_count / image.size)
            image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, 
                               interpolation=cv2.INTER_AREA)
        
        self.original_image = image
        self.height, self.width = image.shape[:2]
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.population_size = population_size
        
        # Flatten and normalize image
        self.flat_image = image.reshape(-1, 3)
        self.scaler = StandardScaler()
        self.normalized_image = self.scaler.fit_transform(self.flat_image)
        
    def fitness_function(self, centroids):
        """
        Calculate fitness based on within-cluster sum of squares
        """
        # Compute distances 
        distances = np.sqrt(np.sum((self.normalized_image[:, np.newaxis] - centroids)**2, axis=2))
        labels = np.argmin(distances, axis=1)
        
        # Compute within-cluster sum of squares
        wcss = 0
        for k in range(self.num_clusters):
            cluster_points = self.normalized_image[labels == k]
            if len(cluster_points) > 0:
                wcss += np.sum((cluster_points - centroids[k])**2)
        
        return wcss
    
    def optimize(self):
        """
        Perform Grey Wolf Optimization clustering
        """
        # Initialize wolf positions (centroids)
        min_vals = self.normalized_image.min(axis=0)
        max_vals = self.normalized_image.max(axis=0)
        
        wolves = np.random.uniform(
            low=min_vals, 
            high=max_vals, 
            size=(self.population_size, self.num_clusters, 3)
        )
        
        # Best wolf tracking
        alpha_score = float('inf')
        alpha_pos = None
        
        # Progress tracking
        print("Optimization Progress:")
        
        # GWO Algorithm
        for iteration in range(self.max_iterations):
            best_fitness = float('inf')
            
            # Evaluate each wolf's fitness
            for i in range(self.population_size):
                current_fitness = self.fitness_function(wolves[i])
                
                # Update best fitness
                if current_fitness < best_fitness:
                    best_fitness = current_fitness
                
                # Update alpha wolf
                if current_fitness < alpha_score:
                    alpha_score = current_fitness
                    alpha_pos = wolves[i].copy()
            
            # Print progress
            print(f"Iteration {iteration+1}: Best Fitness = {alpha_score:.4f}")
            
            # Early stopping if converged
            if alpha_score < 0.01:
                print("Converged early")
                break
            
            # Decay parameter
            a = 2 - iteration * (2 / self.max_iterations)
            
            # Update wolf positions
            for i in range(self.population_size):
                for j in range(self.num_clusters):
                    A1 = 2 * a * np.random.random() - a
                    C1 = 2 * np.random.random()
                    
                    # Distance from alpha wolf
                    D_alpha = abs(C1 * alpha_pos[j] - wolves[i][j])
                    X1 = alpha_pos[j] - A1 * D_alpha
                    
                    wolves[i][j] = X1
        
        # Final clustering
        centroids = alpha_pos
        distances = np.sqrt(np.sum((self.normalized_image[:, np.newaxis] - centroids)**2, axis=2))
        labels = np.argmin(distances, axis=1)
        
        # Denormalize centroids
        denormalized_centroids = self.scaler.inverse_transform(centroids)
        
        # Create clustered image
        clustered_image = denormalized_centroids[labels].reshape(
            self.height, self.width, 3
        ).astype(np.uint8)
        
        return denormalized_centroids, clustered_image

def main():
    # Read the PNG image
    image_path = 'equalized_image.png'  # Replace with your image path
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read the image at {image_path}")
        return
    
    # Convert to RGB if needed
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Perform GWO clustering
    gwo = GreyWolfOptimizer(image, num_clusters=5, max_iterations=30)
    centroids, clustered_image = gwo.optimize()
    
    # Save results
    cv2.imwrite('gwo_clustered_image.png', clustered_image)
    
    # Print centroids
    print("\nCluster Centroids:")
    for i, centroid in enumerate(centroids):
        print(f"Cluster {i+1}: {centroid}")

if __name__ == "__main__":
    main()
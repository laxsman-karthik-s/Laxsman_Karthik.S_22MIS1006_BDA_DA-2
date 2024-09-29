import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Define the SOM class
class SelfOrganizingMap:
    def __init__(self, m, n, dim, learning_rate=0.5, radius=None, num_iterations=1000):
        self.m = m  # Number of rows in the SOM grid
        self.n = n  # Number of columns in the SOM grid
        self.dim = dim  # Dimension of input data
        self.learning_rate = learning_rate  # Initial learning rate
        self.radius = radius if radius else max(m, n) / 2  # Initial neighborhood radius
        self.num_iterations = num_iterations  # Number of iterations
        self.time_constant = num_iterations / np.log(self.radius)  # Time constant for decay
        self.weights = np.random.rand(m, n, dim)  # Initialize random weights for each neuron

    def train(self, data):
        for iteration in range(self.num_iterations):
            input_vector = data[np.random.randint(0, data.shape[0])]
            bmu_index = self.find_bmu(input_vector)
            current_learning_rate = self.learning_rate * np.exp(-iteration / self.num_iterations)
            current_radius = self.radius * np.exp(-iteration / self.time_constant)
            self.update_weights(input_vector, bmu_index, current_learning_rate, current_radius)

    def find_bmu(self, input_vector):
        distances = np.linalg.norm(self.weights - input_vector, axis=2)
        bmu_index = np.unravel_index(np.argmin(distances), (self.m, self.n))
        return bmu_index

    def update_weights(self, input_vector, bmu_index, learning_rate, radius):
        for i in range(self.m):
            for j in range(self.n):
                distance_to_bmu = np.linalg.norm(np.array([i, j]) - np.array(bmu_index))
                if distance_to_bmu < radius:
                    influence = np.exp(-distance_to_bmu**2 / (2 * (radius**2)))
                    self.weights[i, j] += influence * learning_rate * (input_vector - self.weights[i, j])

# Function to preprocess the Minimal Dataset
def preprocess_minimal_data(file_path):
    # Load the dataset
    df = pd.read_excel(file_path)

    # Check if the dataframe is empty
    if df.empty:
        raise ValueError("The Excel file does not contain any data.")

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Remove rows with negative or zero quantities
    df = df[df['Quantity'] > 0]

    # Group data by 'CustomerID' and aggregate features
    customer_df = df.groupby('CustomerID').agg({
        'Quantity': 'sum',         # Total quantity purchased
        'TotalPrice': 'sum'        # Total amount spent
    }).rename(columns={'Quantity': 'TotalQuantity'})

    # Normalize the features
    scaler = MinMaxScaler()
    customer_data_scaled = scaler.fit_transform(customer_df)

    return customer_data_scaled, customer_df

# Function to visualize SOM clusters with enhanced graphs
def visualize_som(som, customer_df):
    plt.figure(figsize=(14, 8))
    
    # Create a color palette
    cmap = sns.color_palette("viridis", as_cmap=True)
    
    # Create a grid to visualize the customer distribution
    grid = np.zeros((som.m, som.n))
    count_grid = np.zeros((som.m, som.n))  # To count how many data points fall in each cell
    
    for input_vector in customer_df.values:
        bmu_index = som.find_bmu(input_vector)
        grid[bmu_index] += 1  # Increment count for the BMU
        count_grid[bmu_index] += 1
    
    # Normalize the grid for better visualization
    grid_normalized = grid / (count_grid + 1e-5)  # Avoid division by zero
    
    # Create a heatmap for the SOM grid
    plt.subplot(1, 1, 1)
    sns.heatmap(grid_normalized, cmap=cmap, annot=True, fmt='.0f', cbar=True)
    
    plt.title('Customer Distribution on SOM Grid')
    plt.xlabel('SOM Columns')
    plt.ylabel('SOM Rows')
    plt.xticks(np.arange(som.n) + 0.5, np.arange(1, som.n + 1))
    plt.yticks(np.arange(som.m) + 0.5, np.arange(1, som.m + 1))

    plt.tight_layout()
    plt.show()

# Main function for customer segmentation using SOM
if __name__ == '__main__':
    # Step 1: Preprocess the online retail dataset
    file_path = r'C:\Users\Pc\AppData\Local\Programs\Python\Python38\Online Retail.xlsx'  # Provide the path to the dataset
    customer_data_scaled, customer_df = preprocess_minimal_data(file_path)

    # Step 2: Initialize SOM with a 10x10 grid and input dimension matching the data
    som = SelfOrganizingMap(10, 10, customer_data_scaled.shape[1], learning_rate=0.5, num_iterations=1000)

    # Step 3: Train the SOM on the customer data
    som.train(customer_data_scaled)

    # Step 4: Visualize the customer segments using SOM
    visualize_som(som, customer_df)

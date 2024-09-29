import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage, dendrogram

class SelfOrganizingMap:
    def __init__(self, m, n, dim, learning_rate=0.5, radius=None, num_iterations=1000):
        self.m = m
        self.n = n
        self.dim = dim
        self.learning_rate = learning_rate
        self.radius = radius if radius else max(m, n) / 2
        self.num_iterations = num_iterations
        self.time_constant = num_iterations / np.log(self.radius)
        self.weights = np.random.rand(m, n, dim)

    def train(self, data):
        for iteration in range(self.num_iterations):
            input_vector = data[np.random.randint(0, data.shape[0])]
            bmu_index = self.find_bmu(input_vector)
            current_learning_rate = self.learning_rate * np.exp(-iteration / self.num_iterations)
            current_radius = self.radius * np.exp(-iteration / self.time_constant)
            self.update_weights(input_vector, bmu_index, current_learning_rate, current_radius)

            if iteration % 200 == 0:
                self.visualize_weights(iteration)

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

    def visualize_weights(self, iteration):
        plt.figure(figsize=(10, 6))
        for i in range(self.m):
            for j in range(self.n):
                color = plt.cm.coolwarm(self.weights[i, j].mean())
                plt.scatter(i, j, s=100, c=[color], marker='o', edgecolor='black')
        plt.title(f"SOM Weights at Iteration {iteration}")
        plt.xlabel("SOM Grid Rows")
        plt.ylabel("SOM Grid Columns")
        plt.colorbar(label='Weight Value')
        plt.show()

    def plot_dendrogram(self):
        reshaped_weights = self.weights.reshape(self.m * self.n, self.dim)
        Z = linkage(reshaped_weights, method='ward')
        plt.figure(figsize=(12, 6))
        dendrogram(Z, orientation='top', labels=[f'Neuron ({i},{j})' for i in range(self.m) for j in range(self.n)])
        plt.title('Dendrogram of SOM Weights')
        plt.xlabel('Neurons')
        plt.ylabel('Distance')
        plt.show()

def preprocess_telco_customer_churn_data(file_path):
    df = pd.read_csv(file_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    df = pd.get_dummies(df, columns=['gender', 'Partner', 'Dependents', 'PhoneService', 
                                     'MultipleLines', 'InternetService', 'OnlineSecurity', 
                                     'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                                     'StreamingTV', 'StreamingMovies', 'Contract', 
                                     'PaperlessBilling', 'PaymentMethod'], drop_first=True)
    df.drop(columns=['customerID'], inplace=True)
    scaler = MinMaxScaler()
    customer_data_scaled = scaler.fit_transform(df.drop(columns=['Churn']))
    return customer_data_scaled, df

def visualize_som(som, customer_df):
    plt.figure(figsize=(10, 6))
    for i in range(som.m):
        for j in range(som.n):
            plt.scatter(i, j, s=100, c='gray', marker='o', edgecolor='black')
    for input_vector in customer_df.values:
        if input_vector.shape[0] == som.dim:
            bmu_index = som.find_bmu(input_vector)
            plt.scatter(bmu_index[0], bmu_index[1], s=100, marker='x', color='blue')
        else:
            print(f"Warning: Input vector shape {input_vector.shape} does not match SOM input dimension {som.dim}.")
    plt.title("SOM Customer Segmentation")
    plt.show()

if __name__ == '__main__':
    file_path = 'Telco-Customer.csv'
    customer_data_scaled, customer_df = preprocess_telco_customer_churn_data(file_path)
    num_features = customer_data_scaled.shape[1]
    print(f"Number of features: {num_features}")
    som = SelfOrganizingMap(10, 10, num_features, learning_rate=0.5, num_iterations=1000)
    som.train(customer_data_scaled)
    visualize_som(som, customer_df)
    som.plot_dendrogram()

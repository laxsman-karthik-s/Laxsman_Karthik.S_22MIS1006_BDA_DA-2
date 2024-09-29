# Customer Churn prediction - SOM
## About the Dataset:
The Telco Customer Churn dataset contains information about customers of a telecommunications company and their services usage.Here’s a brief description of its features:

- CustomerID: A unique identifier for each customer.
- Gender: The gender of the customer (Male, Female).
- SeniorCitizen: Indicates whether the customer is a senior citizen (1 for yes, 0 for no).
- Tenure: The number of months the customer has been with the company.
- PhoneService: Indicates whether the customer subscribes to a phone service (Yes, No).
- MultipleLines: Indicates whether the customer has multiple lines (Yes, No, No phone service).
- InternetService: Type of internet service the customer subscribes to (DSL, Fiber optic, No).
- TechSupport: Indicates whether the customer has tech support service (Yes, No, No internet service).
- Contract: The type of contract the customer has (Month-to-month, One year, Two year).
- PaperlessBilling: Indicates whether the customer has paperless billing (Yes, No).
- PaymentMethod: The method of payment (Electronic check, Mailed check, Bank transfer, Credit card).
- MonthlyCharges: The amount charged to the customer monthly.
- TotalCharges: The total amount charged to the customer over their tenure.
- Churn: Indicates whether the customer has churned (Yes, No).

## Requirements
- Python Version:Python 3.6 or higher (recommended: Python 3.8 or higher).
### Required Libraries:
You need to install the following Python libraries:
- numpy: For numerical operations.
- pandas: For data manipulation and analysis.
- matplotlib: For data visualization.
- scikit-learn: For data preprocessing (specifically for MinMaxScaler).
- openpyxl: For reading Excel files.

You can install these libraires using the following commands:
``` bash
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install openpyxl
```
## Working
1. SelfOrganizingMap Class:
- Initialization: Sets up the SOM grid dimensions, learning rate, and weights.
- Training: Updates weights based on input data by finding the Best Matching Unit (BMU) and adjusting weights within the neighborhood.
- Visualization: Plots SOM weights and creates a dendrogram to show hierarchical relationships between neurons.
2. Data Preprocessing: The preprocess_telco_customer_churn_data function loads the dataset, handles missing values, converts categorical variables to dummy variables, and normalizes the features.
3. SOM Visualization: The visualize_som function plots customer segments in the SOM grid.
4. Main Functionality: In the __main__ block, the code loads the dataset, preprocesses the data, initializes and trains the SOM, and visualizes customer segments and the dendrogram.

### Additional notes
- Ensure that the dataset file is in the same directory as your script or adjust the file_path variable to point to the correct location.

## Ouputs
- The SOM grid after 0 iterations
![SOM_i0](https://github.com/user-attachments/assets/ae96f0df-5f07-489b-94c0-abafbed6eb1d)
- The SOM grid after 200 iterations
![SOM_i200](https://github.com/user-attachments/assets/2693795a-8dd5-4715-8d5f-89bb00d4e64a)
- The SOM grid after 400 iterations
![SOM_i400](https://github.com/user-attachments/assets/0acf26a5-83c9-42bc-93b2-116c61ebb354)
- The SOM grid after 600 iterations
![SOM_i600](https://github.com/user-attachments/assets/84fbb816-8db0-4d6f-8c10-dd9cad650711)
- The SOM grid after 800 iterations
![SOM_i800](https://github.com/user-attachments/assets/f0ff44b7-0cee-4693-9280-1d64fe38cab6)
- SOM Dendrogram for the telecom churn dataset
![SOM_dg](https://github.com/user-attachments/assets/afd5bcfa-2d97-47d9-83d5-70f2c0a4dc5a)

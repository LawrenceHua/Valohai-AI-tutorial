import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle

# Load the dataset from Valohai's input path
data = pd.read_csv('/valohai/inputs/data.csv/data.csv')

# Train a simple collaborative filtering model using Nearest Neighbors
model = NearestNeighbors(n_neighbors=5, algorithm='auto')
model.fit(data[['user_id', 'movie_id']].values)

# Save the trained model in the writable outputs directory
with open('/valohai/outputs/model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

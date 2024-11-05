import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle

print("Loading data...")
# Load the dataset from Valohai's input path
data = pd.read_csv('/valohai/inputs/data.csv/data.csv')
print("Data loaded successfully.")

print("Training the recommendation model...")
# Train a simple collaborative filtering model using Nearest Neighbors
model = NearestNeighbors(n_neighbors=5, algorithm='auto')
model.fit(data[['user_id', 'movie_id']].values)
print("Model training complete.")

print("Saving the model...")
# Save the trained model in the writable outputs directory
with open('/valohai/outputs/model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
print("Model saved as model.pkl.")

# Perform a quick prediction for demonstration purposes
print("Running a quick prediction...")
# Example: Predict 5 nearest neighbors for a sample user-movie pair
sample_user_movie = [[1, 10]]  # Replace with actual user_id and movie_id values
distances, indices = model.kneighbors(sample_user_movie)
print("Prediction complete. Nearest neighbors (movie_id) for user 1 and movie 10 are:\n", indices)

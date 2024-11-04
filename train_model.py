import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import pickle

# Load the data
data = pd.read_csv('data.csv')

# Split data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Train a collaborative filtering model using Nearest Neighbors
model = NearestNeighbors(n_neighbors=5, algorithm='auto')
model.fit(train_data[['user_id', 'movie_id']].values)

# Save the trained model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model training complete and saved as model.pkl")

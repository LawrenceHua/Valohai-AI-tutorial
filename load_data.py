import pandas as pd

# Load the dataset from the sample CSV
data_url = 'https://sandbox:/mnt/data/sample_movie_ratings.csv'  # Replace with actual link if needed
data = pd.read_csv(data_url)

# Save locally as "data.csv" for the training step
data.to_csv('data.csv', index=False)
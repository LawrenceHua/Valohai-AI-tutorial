import pandas as pd

# Load the dataset from the current directory (already bundled with the project)
data = pd.read_csv('sample_movie_ratings.csv')

# Save the output in the designated writable outputs directory
data.to_csv('/valohai/outputs/data.csv', index=False)
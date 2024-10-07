import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Flatten, Dense, concatenate
from tensorflow.keras.optimizers import Adam

# Load dataset

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Encode user IDs and movie IDs to integer values
user_ids = ratings['userId'].unique().tolist()
movie_ids = ratings['movieId'].unique().tolist()

user_to_encoded = {x: i for i, x in enumerate(user_ids)}
movie_to_encoded = {x: i for i, x in enumerate(movie_ids)}

ratings['user'] = ratings['userId'].map(user_to_encoded)
ratings['movie'] = ratings['movieId'].map(movie_to_encoded)

# Prepare the data
X = ratings[['user', 'movie']].values
y = ratings['rating'].values

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the embedding size (latent features)
n_factors = 50

# User input
user_input = Input(shape=(1,))
user_embedding = Embedding(input_dim=len(user_ids), output_dim=n_factors)(user_input)
user_vector = Flatten()(user_embedding)

# Movie input
movie_input = Input(shape=(1,))
movie_embedding = Embedding(input_dim=len(movie_ids), output_dim=n_factors)(movie_input)
movie_vector = Flatten()(movie_embedding)

# Concatenate user and movie vectors
concatenated = concatenate([user_vector, movie_vector])

# Fully connected layers
dense1 = Dense(128, activation='relu')(concatenated)
dense2 = Dense(64, activation='relu')(dense1)
output = Dense(1)(dense2)  # Predict a single number (rating)

# Compile the model
model = Model([user_input, movie_input], output)
model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')

# Train the model
history = model.fit([X_train[:, 0], X_train[:, 1]], y_train, 
                    validation_data=([X_test[:, 0], X_test[:, 1]], y_test), 
                    epochs=10, 
                    batch_size=64)

# Predict ratings for the test set
y_pred = model.predict([X_test[:, 0], X_test[:, 1]])

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")

def recommend_movies(user_id, num_recommendations=10):
    # Get the user’s encoded ID
    user_encoded = user_to_encoded[user_id]
    
    # Predict ratings for all movies
    all_movies = np.array(movie_ids)
    predicted_ratings = model.predict([np.array([user_encoded] * len(movie_ids)), np.array(movie_to_encoded.values())])
    
    # Sort movies based on predicted rating
    top_indices = predicted_ratings.flatten().argsort()[-num_recommendations:][::-1]
    top_movie_ids = [all_movies[i] for i in top_indices]
    
    recommended_movies = movies[movies['movieId'].isin(top_movie_ids)]
    return recommended_movies

# Example usage
user_id = 5  # Provide the user ID you want recommendations for
recommendations = recommend_movies(user_id)
print(recommendations)  


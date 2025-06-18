import pandas as pd

# Load the CSV file
df = pd.read_csv("/Users/rea/Documents/GitHub/LLMFromScratch/test.csv", low_memory=False)

print(df.columns.tolist())

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random


# how am i going to do this?

# --- 1. Prepare the Data ---
# For a very simple model, we'll use a small corpus of text.
# In a real scenario, you would load a much larger text file.
text = """
The quick brown fox jumps over the lazy dog.
A simple language model learns patterns in text.
This example uses Keras and TensorFlow.
"""

text = " ".join(["dog" for i in range(20)] + ["cat" for i in range(20)])

# Convert text to lowercase to simplify vocabulary
text = text.lower()

# Create a vocabulary
cards = sorted(list(set(text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(text)
n_vocab = len(chars)

print(f"Total Characters in corpus: {n_chars}")
print(f"Total Unique Characters (Vocabulary Size): {n_vocab}")

# Prepare training data: input sequences and corresponding output characters
seq_length = 10 # Number of characters to use as input to predict the next character
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = text[i:i + seq_length]
    seq_out = text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)
print(f"Total Patterns: {n_patterns}")

# Reshape X to be [samples, time steps, features]
# For a recurrent neural network, input should be 3D.
X = np.reshape(dataX, (n_patterns, seq_length))
print(X.shape)
# Normalize input values to range (0-1)
# This helps the neural network learn more effectively.

# One-hot encode the output variable (y)
# Each output character is converted into a binary vector with a 1 at its index.
y = to_categorical(dataY)

# --- 2. Define the Keras Model ---
model = Sequential()
# Embedding layer to convert integer inputs into dense vectors
# input_dim: size of the vocabulary
# output_dim: dimension of the dense embedding
# input_length: length of input sequences
model.add(Embedding(input_dim=n_vocab, output_dim=256, input_length=seq_length))
# LSTM (Long Short-Term Memory) layer is good for sequential data.
# It helps the model remember long-term dependencies.
model.add(LSTM(256))
# Dense output layer with softmax activation for multi-class classification.
# n_vocab is the number of possible output characters.
model.add(Dense(n_vocab, activation='softmax'))

# Compile the model
# loss='categorical_crossentropy' is suitable for multi-class classification
# optimizer='adam' is a popular and effective optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam')

# --- 3. Train the Model ---
# epochs: number of times the model will go through the entire dataset
# batch_size: number of samples per gradient update
# You might need more epochs for a more complex dataset or model.
print("\nStarting model training...")
model.fit(X, y, epochs=50, batch_size=64, verbose=2) # verbose=2 shows one line per epoch
print("Model training complete.")

# --- 4. Generate Text ---
# Pick a random seed sequence from the training data to start generation
start_index = np.random.randint(0, n_patterns-1)
pattern = dataX[start_index]
print(f"\nSeed for generation: \"{''.join([int_to_char[value] for value in pattern])}\"")




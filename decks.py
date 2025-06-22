import pandas as pd
import numpy as np
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random


df = pd.read_csv("/Users/rea/Documents/GitHub/LLMFromScratch/test.csv", low_memory=False)

print(df)

for column in df.columns:
    print(column)

cards = [column[len("pack_card_"):] for column in df.columns if column.startswith("pack_card")]
print(cards)

n_vocab = len(cards) + 1


card_to_int = dict((c, i) for i, c in enumerate(cards))
int_to_card = dict((i, c) for i, c in enumerate(cards))
int_to_card[len(cards)] = "_"

print(cards)
print(card_to_int)
print(int_to_card)
print(card_to_int[int_to_card[22]])

seq_length = 42
dataX = []
dataY = []
for index, row in df.iterrows():
    if index > 60:
        break

    X = [card_to_int[column[len("pool_"):]]  for column in df.columns if column.startswith("pool_") and row[column]==1]
    X = []
    for column in df.columns:
        if column.startswith("pool_"):
              X.extend([card_to_int[column[len("pool_"):]]]*row[column])
    y = card_to_int[row["pick"]]
    X = [len(cards)]*(seq_length-len(X))+X
    dataX.append(X)
    dataY.append(y)
    print(row["pick"])
    print(X, y)
#     print(row.pick)
#     print(row.)

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
y = to_categorical(dataY, n_vocab)

# --- 2. Define the Keras Model ---
model = Sequential()
# Embedding layer to convert integer inputs into dense vectors
# input_dim: size of the vocabulary
# output_dim: dimension of the dense embedding
# input_length: length of input sequences
model.add(Embedding(input_dim=n_vocab, output_dim=256, input_length=seq_length))
# LSTM (Long Short-Term Memory) layer is good for sequential data.
# It helps the model remember long-term dependencies.
model.add(LSTM(512))
# Dense output layer with softmax activation for multi-class classification.
# n_vocab is the number of possible output characters.
model.add(Dense(n_vocab, activation='softmax'))

# Compile the model
# loss='categorical_crossentropy' is suitable for multi-class classification
# optimizer='adam' is a popular and effective optimizer
fast_adam = optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=fast_adam)

# --- 3. Train the Model ---
# epochs: number of times the model will go through the entire dataset
# batch_size: number of samples per gradient update
# You might need more epochs for a more complex dataset or model.
print("\nStarting model training...")
model.fit(X, y, epochs=500, batch_size=64, verbose=2) # verbose=2 shows one line per epoch
print("Model training complete.")

# --- 4. Generate Text ---
#this is an empty sequence
pattern = [len(cards)]*seq_length

# generate the cards
generated_text = ""
for i in range(seq_length):
    # Reshape the input pattern for prediction
    x = np.reshape(pattern, (1, len(pattern), 1))
    prediction = model.predict(x, verbose=0) # Predict the next character probabilities
    index = np.argmax(prediction) # Get the index of the character with the highest probability
    result = int_to_card[index]
    generated_text += result

    # Shift the pattern to include the new character and drop the oldest
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

print(f"\nGenerated Text:\n{generated_text}")
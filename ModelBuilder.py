from enum import Enum
import numpy as np
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.utils import to_categorical


class ModelType(Enum):
    LSTM = 1

#number of cards to be picked
SEQUENCE_LENGTH = 42

class ModelBuilder:


    def __init__(self, draft_data):
        self._draft_data = draft_data
        self._model = None

    def train_model(self, epochs = 50 ):
        data = self._draft_data.draft_data
        dataX = []
        dataY = []
        for index, row in data.iterrows():
            if index > 60:
                break

            X = [self._draft_data.cards_to_int[column[len("pool_"):]] for column in data.columns if
                 column.startswith("pool_") and row[column] == 1]
            X = []
            for column in data.columns:
                if column.startswith("pool_"):
                    X.extend([self._draft_data.cards_to_int[column[len("pool_"):]]] * row[column])
            y = self._draft_data.cards_to_int[row["pick"]]
            X = [len(self._draft_data.cards)] * (SEQUENCE_LENGTH - len(X)) + X
            dataX.append(X)
            dataY.append(y)

        n_patterns = len(dataX)

        # Reshape X to be [samples, time steps, features]
        # For a recurrent neural network, input should be 3D.
        X = np.reshape(dataX, (n_patterns, SEQUENCE_LENGTH))

        # One-hot encode the output variable (y)
        # Each output character is converted into a binary vector with a 1 at its index.
        y = to_categorical(dataY, self._draft_data.n_vocab)

        # --- 2. Define the Keras Model ---
        model = Sequential()
        # Embedding layer to convert integer inputs into dense vectors
        # input_dim: size of the vocabulary
        # output_dim: dimension of the dense embedding
        # input_length: length of input sequences
        model.add(Embedding(input_dim=self._draft_data.n_vocab, output_dim=256, input_length=SEQUENCE_LENGTH))
        # LSTM (Long Short-Term Memory) layer is good for sequential data.
        # It helps the model remember long-term dependencies.
        model.add(LSTM(512))
        # Dense output layer with softmax activation for multi-class classification.
        # n_vocab is the number of possible output characters.
        model.add(Dense(self._draft_data.n_vocab, activation='softmax'))

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
        model.fit(X, y, epochs=epochs, batch_size=64, verbose=2)  # verbose=2 shows one line per epoch
        print("Model training complete.")
        self._model = model

    def predict(self, deck, pack):
        #Fill with empty cars and then add what we have at the end
        deck = [self._draft_data.empty_card]*(SEQUENCE_LENGTH-len(deck)) + deck
        x = np.reshape(deck, (1, len(deck), 1))
        prediction = self._model.predict(x, verbose=0)

        card_probability = []
        pack_probability = sum([prediction[0][card] for card in pack])

        for card in pack:
            card_probability.append(prediction[0][card]/pack_probability)
            print(card)
            print(prediction[0][card]/pack_probability)
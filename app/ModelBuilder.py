from enum import Enum
import numpy as np
import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LayerNormalization, Dropout, MultiHeadAttention, Dense
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Layer, Embedding
from tensorflow.keras.models import Sequential
import pandas as pd
import random
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# --- Custom Layers ---
SEQUENCE_LENGTH = 64

def create_padding_mask(seq):
    """Creates a padding mask."""
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

class TransformerBlock(Model):
    """Transformer block with multi-head attention and feed-forward network."""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, inputs, training=False, mask=None):
        attn_output = self.att(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config

class PositionalEmbedding(Layer):
    """Adds positional embeddings to token embeddings."""
    def __init__(self, sequence_length, output_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        self.position_embedding = Embedding(
            input_dim=sequence_length,
            output_dim=output_dim
        )
    
    def call(self, inputs):
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        position_vectors = self.position_embedding(positions)
        return inputs + position_vectors
    
    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "output_dim": self.output_dim,
        })
        return config

class ModelType(Enum):
    TRANSFORMER_PICK = 1

# --- ModelBuilder Class ---

class ModelBuilder:
    """
    Builds and trains a transformer model to predict the next card pick 
    based on the current pool and pack. FIXED VERSION.
    """
    
    def __init__(self, draft_data):
        self._draft_data = draft_data
        self._model = None
    
    def train_model(self, epochs=50, min_player_wr=0.6, validation_split=0.2):
        
        # --- 1. DATA FILTERING AND PREPARATION ---
        data = self._draft_data.draft_data.copy()
        
        # Calculate player win rate
        total_matches = data['event_match_wins'] + data['event_match_losses']
        data['player_win_rate'] = np.where(
            total_matches > 0, 
            data['event_match_wins'] / total_matches, 
            0.5
        )
        
        initial_count = len(data)
        filtered_data = data[data['player_win_rate'] >= min_player_wr].copy()
        
        print(f"Calculated 'player_win_rate' based on event match results.")
        print(f"Filtered picks from {initial_count} down to {len(filtered_data)}")
        print(f"Training on 'good player' picks (WR >= {min_player_wr}) only.")

        dataX = []
        dataY = []
        PAD_TOKEN = self._draft_data.pad_token
        
        print("Building sequences for pick prediction...")
        
        for index, row in filtered_data.iterrows():
            # 1. Get the Pool (cards already picked)
            pool = []
            for column in data.columns:
                if column.startswith("pool_"):
                    card_name = column[len("pool_"):]
                    if card_name in self._draft_data.cards_to_int:
                        pool.extend([self._draft_data.cards_to_int[card_name]] * int(row[column]))
            
            # 2. Get the Pack (cards currently available)
            pack_cards = []
            for column in data.columns:
                if column.startswith("pack_card_") and row[column] == 1:
                    card_name = column[len("pack_card_"):]
                    if card_name in self._draft_data.cards_to_int:
                         pack_cards.append(self._draft_data.cards_to_int[card_name])

            # 3. Get the Target Pick (y)
            target_card_name = row["pick"]
            if target_card_name not in self._draft_data.cards_to_int:
                continue
            target_pick = self._draft_data.cards_to_int[target_card_name]
            
            # CRITICAL FIX: Do NOT include the target pick in the input sequence
            # The input is: [SEP_POOL] + pool + [SEP_PACK] + pack (without the pick)
            current_sequence = (
                [self._draft_data._sep_pool_token] + 
                pool + 
                [self._draft_data._sep_pack_token] + 
                pack_cards  # All available cards, including what will be picked
            )

            # Truncate/Pad
            if len(current_sequence) > SEQUENCE_LENGTH:
                current_sequence = current_sequence[len(current_sequence) - SEQUENCE_LENGTH:]

            padded_sequence = [PAD_TOKEN] * (SEQUENCE_LENGTH - len(current_sequence)) + current_sequence

            dataX.append(padded_sequence)
            dataY.append(target_pick)

        X = np.array(dataX) 
        y = to_categorical(dataY, self._draft_data.n_vocab)
        
        print(f"Total training examples created: {len(X)}")

        # --- 2. DEFINE THE TRANSFORMER MODEL ---
        EMBED_DIM = 256
        NUM_HEADS = 8
        FF_DIM = 512
        DROPOUT_RATE = 0.2  # Increased dropout to combat overfitting

        print("\nDefining Transformer Model...")
        
        inputs = Input(shape=(SEQUENCE_LENGTH,), dtype='int32')
        
        # Embedding with mask_zero to handle padding
        x = Embedding(
            input_dim=self._draft_data.n_vocab, 
            output_dim=EMBED_DIM, 
            input_length=SEQUENCE_LENGTH,
            mask_zero=True  # Enable masking for padding tokens
        )(inputs)

        x = PositionalEmbedding(SEQUENCE_LENGTH, EMBED_DIM)(x)
        x = Dropout(DROPOUT_RATE)(x)  # Add dropout after embeddings

        # Transformer blocks
        x = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM, dropout_rate=DROPOUT_RATE)(x)
        x = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM, dropout_rate=DROPOUT_RATE)(x)
        x = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM, dropout_rate=DROPOUT_RATE)(x)

        # Global average pooling instead of just last token
        # This aggregates information from the entire sequence
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Additional dense layer for better representation
        x = Dense(256, activation='relu')(x)
        x = Dropout(DROPOUT_RATE)(x)

        outputs = Dense(self._draft_data.n_vocab, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        
        # Lower initial learning rate
        optimizer = optimizers.Adam(learning_rate=0.0001)
        model.compile(
            loss='categorical_crossentropy', 
            optimizer=optimizer,
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        print(model.summary())
        
        # --- 3. TRAIN THE MODEL WITH CALLBACKS ---
        print("\nStarting model training with Early Stopping...")
        
        # Early stopping with more patience
        early_stopper = EarlyStopping(
            monitor='val_loss', 
            patience=10,  
            verbose=1,
            mode='min',
            restore_best_weights=True
        )
        
        # Reduce LR on plateau
        lr_reducer = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-7, 
            verbose=1
        )
        
        # Save best model
        checkpoint = ModelCheckpoint(
            'app/model/best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        callbacks_list = [early_stopper, lr_reducer, checkpoint]
        
        history = model.fit(
            X, y, 
            epochs=epochs, 
            batch_size=32,  # Smaller batch size for better generalization
            validation_split=validation_split,
            callbacks=callbacks_list,
            verbose=2
        )
        
        self._model = model
        return history
        
    def predict(self, deck: list[int], pack: list[int]):
        """
        Predicts the probability of picking each card in the pack given the current pool.
        """
        if self._model is None:
            raise ValueError("Model not trained! Call train_model() first.")
            
        PAD_TOKEN = self._draft_data.pad_token
        
        # FIXED: Match training - don't add PAD_TOKEN at the end
        current_sequence = (
            [self._draft_data._sep_pool_token] + 
            deck + 
            [self._draft_data._sep_pack_token] + 
            pack
        )

        if len(current_sequence) > SEQUENCE_LENGTH:
            current_sequence = current_sequence[len(current_sequence) - SEQUENCE_LENGTH:]

        padded_sequence = [PAD_TOKEN] * (SEQUENCE_LENGTH - len(current_sequence)) + current_sequence
        x = np.array([padded_sequence]) 

        prediction = self._model.predict(x, verbose=0)
        
        card_probability = []
        pack_probability_sum = sum(prediction[0][card] for card in pack)

        for card_id in pack:
            prob = prediction[0][card_id]
            normalized_prob = prob / pack_probability_sum if pack_probability_sum != 0 else 0
            
            card_name = self._draft_data.int_to_card[card_id]
            card_probability.append({
                "card_id": card_id,
                "card_name": card_name,
                "probability": float(normalized_prob)
            })

        card_probability.sort(key=lambda x: x["probability"], reverse=True)
        return card_probability
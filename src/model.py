# LSTM model architecture for text generation.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

class LSTMModel:
    def __init__(self, vocab_size, max_sequence_length, embedding_dim=256, lstm_units=512):
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        
    def build_model(self, bidirectional=True, dropout_rate=0.2):
        model = Sequential([Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length
            ),
            self._add_lstm_layer(bidirectional, dropout_rate),
            
            
            Dense(self.vocab_size, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("Model built successfully")
        print(f"Model summary:\n{model.summary()}")
        return model
    
    def _add_lstm_layer(self, bidirectional, dropout_rate):
        if bidirectional:
            return Bidirectional(
                LSTM(self.lstm_units, return_sequences=False, dropout=dropout_rate),
                name='bidirectional_lstm'
            )
        else:
            return LSTM(self.lstm_units, return_sequences=False, dropout=dropout_rate)
    
    def build_deeper_model(self, num_layers=2):
        model = Sequential([
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length
            )
        ])
        
        for i in range(num_layers):
            return_sequences = i < num_layers - 1  # Only last layer returns sequences=False
            model.add(
                LSTM(
                    self.lstm_units // (2**i), 
                    return_sequences=return_sequences,
                    dropout=0.2,
                    name=f'lstm_{i+1}'
                )
            )
            if i < num_layers - 1:
                model.add(Dropout(0.2))
        
        # Dense output layer
        model.add(Dense(self.vocab_size, activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print(f"Deeper model built with {num_layers} LSTM layers")
        print(f"Model summary:\n{model.summary()}")
        return model
    
    def get_callbacks(self, model_path="models/best_model.h5"):
        """Get training callbacks."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=128):
        if self.model is None:
            self.build_model()
        
        callbacks = self.get_callbacks()
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def save_model(self, filepath="models/lstm_text_generator.h5"):
        
        if self.model is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save")
    
    def load_model(self, filepath="models/lstm_text_generator.h5"):
        
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model
    
    def predict_next_token(self, input_sequence):
        if self.model is None:
            raise ValueError("Model not loaded or built. Please load or build the model first.")
        
        
        if len(input_sequence.shape) == 1:
            input_sequence = input_sequence.reshape(1, -1)
        
        predictions = self.model.predict(input_sequence, verbose=0)
        return predictions[0]

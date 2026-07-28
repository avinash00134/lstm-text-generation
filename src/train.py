# training scripts for LSTM text generation model.

import numpy as np
from sklearn.model_selection import train_test_split
from data_preprocessing import TextPreprocessor
from model import LSTMModel
import os
import pickle

def train_character_level():

    print("=" * 50)
    print("CHARACTER-LEVEL TEXT GENERATION")
    print("=" * 50)
    
   
    preprocessor = TextPreprocessor(max_sequence_length=40)
    
    text = preprocessor.load_text_from_url()
    clean_text = preprocessor.preprocess_text(text)
    
    preprocessor.tokenize_text(clean_text, character_level=True)
    
  
    X, y = preprocessor.create_sequence_pairs()
   
    char_to_idx, idx_to_char = preprocessor.tokenizer
    X_encoded = []
    y_encoded = []
    
    for seq, target in preprocessor.sequences:
        encoded_seq = [char_to_idx[char] for char in seq]
        X_encoded.append(encoded_seq)
        y_encoded.append(char_to_idx[target])
    
    X = np.array(X_encoded)
    y = np.array(y_encoded)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Build train model
    model = LSTMModel(
        vocab_size=preprocessor.vocab_size,
        max_sequence_length=preprocessor.max_sequence_length,
        embedding_dim=128,
        lstm_units=256
    )
    
    # Build model
    model.build_model(bidirectional=True, dropout_rate=0.2)
    
    # Train
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=64
    )
   
    model.save_model("models/char_lstm_model.h5")
    preprocessor.save_tokenizer("models/char_tokenizer.pkl")
    
    return model, preprocessor

def train_word_level():
    print("=" * 50)
    print("WORD-LEVEL TEXT GENERATION")
    print("=" * 50)
    
    preprocessor = TextPreprocessor(max_sequence_length=10)
    
    text = preprocessor.load_text_from_url()
    clean_text = preprocessor.preprocess_text(text)
    
    preprocessor.tokenize_text(clean_text, character_level=False)
    
    # Prepare sequences
    X, y = preprocessor.create_sequence_pairs()
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Build train model
    model = LSTMModel(
        vocab_size=preprocessor.vocab_size,
        max_sequence_length=preprocessor.max_sequence_length,
        embedding_dim=256,
        lstm_units=512
    )
    
    # Build model
    model.build_model(bidirectional=True, dropout_rate=0.2)
    
    # Train
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=30,
        batch_size=128
    )
    
    # Save model and tokenizer
    model.save_model("models/word_lstm_model.h5")
    preprocessor.save_tokenizer("models/word_tokenizer.pkl")
    
    return model, preprocessor

def experiment_with_architectures():
    print("\n" + "=" * 50)
    print("EXPERIMENT: DIFFERENT ARCHITECTURES")
    print("=" * 50)
    
    
    preprocessor = TextPreprocessor(max_sequence_length=20)
    text = preprocessor.load_text_from_url()
    clean_text = preprocessor.preprocess_text(text)[:5000]  # Use smaller sample
    
    # Tokenize
    preprocessor.tokenize_text(clean_text, character_level=True)
    X, y = preprocessor.create_sequence_pairs()
    
    # Convert to indices
    char_to_idx, idx_to_char = preprocessor.tokenizer
    X_encoded = []
    y_encoded = []
    
    for seq, target in preprocessor.sequences[:1000]:  # Use subset for experiments
        encoded_seq = [char_to_idx[char] for char in seq]
        X_encoded.append(encoded_seq)
        y_encoded.append(char_to_idx[target])
    
    X = np.array(X_encoded)
    y = np.array(y_encoded)
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    architectures = [
        {"lstm_units": 128, "bidirectional": False, "dropout": 0.2, "name": "Small Unidirectional"},
        {"lstm_units": 256, "bidirectional": True, "dropout": 0.2, "name": "Medium Bidirectional"},
        {"lstm_units": 512, "bidirectional": True, "dropout": 0.3, "name": "Large Bidirectional"},
    ]
    
    results = []
    
    for arch in architectures:
        print(f"\nTesting architecture: {arch['name']}")
        print("-" * 30)
        
        model = LSTMModel(
            vocab_size=preprocessor.vocab_size,
            max_sequence_length=preprocessor.max_sequence_length,
            embedding_dim=128,
            lstm_units=arch['lstm_units']
        )
        
        model.build_model(
            bidirectional=arch['bidirectional'],
            dropout_rate=arch['dropout']
        )
        
        history = model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=5,
            batch_size=32
        )
      
        final_accuracy = history.history['accuracy'][-1]
        final_val_accuracy = history.history['val_accuracy'][-1]
        
        results.append({
            'architecture': arch['name'],
            'lstm_units': arch['lstm_units'],
            'bidirectional': arch['bidirectional'],
            'accuracy': final_accuracy,
            'val_accuracy': final_val_accuracy
        })
        
        print(f"Final accuracy: {final_accuracy:.4f}")
        print(f"Final validation accuracy: {final_val_accuracy:.4f}")
    
    # Print comparison
    print("\n" + "=" * 50)
    print("ARCHITECTURE COMPARISON RESULTS")
    print("=" * 50)
    
    for result in results:
        print(f"\n{result['architecture']}:")
        print(f"  LSTM Units: {result['lstm_units']}")
        print(f"  Bidirectional: {result['bidirectional']}")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Validation Accuracy: {result['val_accuracy']:.4f}")
    
    return results

if __name__ == "__main__":
   
    os.makedirs("models", exist_ok=True)
    
    print("Starting LSTM Text Generation Training")
    print("-" * 50)
    
    try:
        
        char_model, char_preprocessor = train_character_level()
        print("\n✓ Character-level model trained successfully!")
        
        word_model, word_preprocessor = train_word_level()
        print("\n✓ Word-level model trained successfully!")
        
        experiment_results = experiment_with_architectures()
        print("\n✓ Architecture experiments completed!")
        
        print("\n" + "=" * 50)
        print("TRAINING COMPLETE!")
        print("=" * 50)
        print("Models saved in 'models/' directory")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

  

# text generation trained LSTM model.


import numpy as np
import pickle
from tensorflow.keras.models import load_model

class TextGenerator:
    
    def __init__(self, model_path, tokenizer_path, is_character_level=False):
        self.model = load_model(model_path)
        self.is_character_level = is_character_level
        
        # Load tokenizer
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        if is_character_level:
            self.char_to_idx, self.idx_to_char = self.tokenizer
            self.vocab_size = len(self.char_to_idx) + 1
        else:
            self.vocab_size = len(self.tokenizer.word_index) + 1
        
        self.max_sequence_length = self.model.input_shape[1]
        print(f"Generator initialized with max_sequence_length: {self.max_sequence_length}")
    
    def generate_text(self, seed_text, num_tokens=100, temperature=1.0, diversity=0.5):
        if self.is_character_level:
            return self._generate_character_level(seed_text, num_tokens, temperature, diversity)
        else:
            return self._generate_word_level(seed_text, num_tokens, temperature, diversity)
    
    def _generate_character_level(self, seed_text, num_tokens, temperature, diversity):
       
        generated_text = seed_text.lower()
        char_to_idx, idx_to_char = self.tokenizer
        
        for _ in range(num_tokens):
            # Prepare input sequence
            if len(generated_text) < self.max_sequence_length:
                # Pad sequence if needed
                input_seq = generated_text.rjust(self.max_sequence_length, ' ')
            else:
                input_seq = generated_text[-self.max_sequence_length:]
            
            # Encode input
            input_encoded = []
            for char in input_seq:
                if char in char_to_idx:
                    input_encoded.append(char_to_idx[char])
                else:
                    input_encoded.append(0)  
            
            
            input_array = np.array([input_encoded])
            predictions = self.model.predict(input_array, verbose=0)[0]
          
            predictions = predictions ** (1/temperature)
            predictions = predictions / predictions.sum()
            
            next_char_idx = np.random.choice(range(len(predictions)), p=predictions)
            next_char = idx_to_char.get(next_char_idx, '')
            
            generated_text += next_char
        
        return generated_text
    
    def _generate_word_level(self, seed_text, num_tokens, temperature, diversity):
        generated_text = seed_text.lower()
        tokenizer = self.tokenizer
        
        for _ in range(num_tokens):
            # Tokenize seed text
            seed_words = generated_text.split()
            
            
            if len(seed_words) < self.max_sequence_length:
                # Pad with empty strings
                seed_words_padded = [''] * (self.max_sequence_length - len(seed_words)) + seed_words
            else:
                seed_words_padded = seed_words[-self.max_sequence_length:]
            
            input_sequence = []
            for word in seed_words_padded:
                if word in tokenizer.word_index:
                    input_sequence.append(tokenizer.word_index[word])
                else:
                    input_sequence.append(0)  # Unknown word
            
            # Predict next word
            input_array = np.array([input_sequence])
            predictions = self.model.predict(input_array, verbose=0)[0]
            
            # Apply temperature and diversity
            predictions = predictions ** (1/temperature)
            predictions = predictions / predictions.sum()
            
            # Sample from predicted distribution
            next_word_idx = np.random.choice(range(len(predictions)), p=predictions)
            
            # Convert index back to word
            next_word = ''
            for word, idx in tokenizer.word_index.items():
                if idx == next_word_idx:
                    next_word = word
                    break
            
            generated_text += ' ' + next_word if next_word else ''
        
        return generated_text
    
    def generate_multiple_samples(self, seed_texts, num_tokens=100, temperature=0.8):
        results = {}
        
        for seed in seed_texts:
            generated = self.generate_text(seed, num_tokens, temperature)
            results[seed] = generated
            print(f"\n{'='*60}")
            print(f"SEED: '{seed[:50]}...'")
            print(f"GENERATED: '{generated[:200]}...'")
            print(f"{'='*60}")
        
        return results

def main():
    print("="*60)
    print("LSTM TEXT GENERATION DEMO")
    print("="*60)
    
    # Seed texts for generation
    seed_texts = [
        "to be or not to be",
        "the king shall",
        "i love thee",
        "shall i compare thee to",
        "all the world's a stage"
    ]
    
    try:
        # Character-level generation
        print("\n" + "="*60)
        print("CHARACTER-LEVEL GENERATION")
        print("="*60)
        
        char_generator = TextGenerator(
            model_path="models/char_lstm_model.h5",
            tokenizer_path="models/char_tokenizer.pkl",
            is_character_level=True
        )
        
        char_generator.generate_multiple_samples(
            seed_texts, 
            num_tokens=100, 
            temperature=0.8
        )
        
        # Word-level generation
        print("\n" + "="*60)
        print("WORD-LEVEL GENERATION")
        print("="*60)
        
        word_generator = TextGenerator(
            model_path="models/word_lstm_model.h5",
            tokenizer_path="models/word_tokenizer.pkl",
            is_character_level=False
        )
        
        word_generator.generate_multiple_samples(
            seed_texts, 
            num_tokens=30, 
            temperature=0.7
        )
        
        # Temperature experiment
        print("\n" + "="*60)
        print("TEMPERATURE EXPERIMENT")
        print("="*60)
        
        temperatures = [0.5, 0.8, 1.2]
        test_seed = "to be or not to be"
        
        for temp in temperatures:
            print(f"\nTemperature: {temp}")
            print("-"*40)
            generated = char_generator.generate_text(
                test_seed, 
                num_tokens=100, 
                temperature=temp
            )
            print(f"Generated: {generated[:150]}...")
        
    except Exception as e:
        print(f"\nError during generation: {e}")
        print("Make sure you have trained models first!")
        print("Run train.py to train the models.")

if __name__ == "__main__":
    main()

# data processing module
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
import os
import pickle

class TextPreprocessor:
    
    def __init__(self, max_sequence_length=50):
       
        self.max_sequence_length = max_sequence_length
        self.tokenizer = None
        self.vocab_size = 0
        self.sequences = None
        self.total_tokens = 0
        self.X = None
        self.y = None
    
    def load_text_from_url(self, url="https://www.gutenberg.org/cache/epub/100/pg100.txt"):
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            self.text = response.text
            print(f"Text loaded successfully. Length: {len(self.text)} characters")
            return self.text
        except Exception as e:
            print(f"Error loading text: {e}")
            # Fallback to local file
            return self.load_local_text()
    
    def load_local_text(self, file_path="data/shakespeare.txt"):
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                self.text = file.read()
            print(f"Local text loaded. Length: {len(self.text)} characters")
            return self.text
        else:
            
            self.text = self._create_sample_text()
            return self.text
    
    def _create_sample_text(self):
        sample = """
        To be, or not to be: that is the question:
        Whether 'tis nobler in the mind to suffer
        The slings and arrows of outrageous fortune,
        Or to take arms against a sea of troubles,
        And by opposing end them? To die: to sleep;
        No more; and by a sleep to say we end
        The heart-ache and the thousand natural shocks
        That flesh is heir to, 'tis a consummation
        Devoutly to be wish'd. To die, to sleep;
        To sleep: perchance to dream: ay, there's the rub;
        For in that sleep of death what dreams may come
        When we have shuffled off this mortal coil,
        Must give us pause: there's the respect
        That makes calamity of so long life;
        """
        return sample * 100  # Repeat to create larger dataset
    
    def preprocess_text(self, text=None):
        if text is None:
            text = self.text
        
       
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        self.clean_text = text
        print(f"Text preprocessed. Length: {len(text)} characters")
        return text
    
    def tokenize_text(self, text=None, character_level=False):
        if text is None:
            text = self.clean_text
        
        if character_level:
            # Character-level tokenization
            chars = sorted(list(set(text)))
            char_to_idx = {c: i+1 for i, c in enumerate(chars)} 
            idx_to_char = {i+1: c for i, c in enumerate(chars)}
            self.tokenizer = (char_to_idx, idx_to_char)
            self.vocab_size = len(chars) + 1  
            sequences = []
            for i in range(len(text) - self.max_sequence_length):
                seq = text[i:i + self.max_sequence_length]
                next_char = text[i + self.max_sequence_length]
                sequences.append((seq, next_char))
            
            self.sequences = sequences
            self.total_tokens = len(text)
            
        else:
            # Word-level tokenization
            self.tokenizer = Tokenizer(num_words=None, filters='', lower=True)
            self.tokenizer.fit_on_texts([text])
            self.vocab_size = len(self.tokenizer.word_index) + 1  
            
            token_sequences = self.tokenizer.texts_to_sequences([text])[0]
            self.total_tokens = len(token_sequences)
            
            self.sequences = []
            for i in range(self.max_sequence_length, len(token_sequences)):
                seq = token_sequences[i-self.max_sequence_length:i]
                next_token = token_sequences[i]
                self.sequences.append((seq, next_token))
        
        print(f"Tokenization complete. Vocabulary size: {self.vocab_size}")
        print(f"Number of sequences: {len(self.sequences)}")
        return self.sequences
    
    def create_sequence_pairs(self, sequences=None):
        if sequences is None:
            sequences = self.sequences
        
        self.X = []
        self.y = []
        
        for seq, target in sequences:
            self.X.append(seq)
            self.y.append(target)
        
        # Convert to numpy arrays
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        
        # Convert target to categorical
        self.y = np.array(self.y)
        
        print(f"Created {len(self.X)} training pairs")
        return self.X, self.y
    
    def save_tokenizer(self, filepath="models/tokenizer.pkl"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"Tokenizer saved to {filepath}")
    
    def load_tokenizer(self, filepath="models/tokenizer.pkl"):
        with open(filepath, 'rb') as f:
            self.tokenizer = pickle.load(f)
        print(f"Tokenizer loaded from {filepath}")

    def get_vocab_size(self):
        """Return vocabulary size."""
        return self.vocab_size

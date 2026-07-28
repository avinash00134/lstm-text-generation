# LSTM Text Generation with Shakespeare

LSTM-based text generation model trained on Shakespeare's works. Supports both character-level and word-level generation with temperature-controlled sampling.

## 🚀
### Installation
```bash
git clone https://github.com/avinash00134/lstm-text-generation.git
cd lstm-text-generation
pip install -r requirements.txt
mkdir -p data models outputs# lstm-text-generation

Short README.md

```markdown
# LSTM Text Generation with Shakespeare

LSTM-based text generation model trained on Shakespeare's works. Supports both character-level and word-level generation with temperature-controlled sampling.

##  Quick Start

### Installation
```bash
git clone https://github.com/avinash00134/lstm-text-generation.git
cd lstm-text-generation
pip install -r requirements.txt
mkdir -p data models outputs
```

Train Models

```bash
python src/train.py
```

· Downloads Shakespeare dataset automatically
· Trains character-level and word-level models
· Saves models to models/ directory
· Runs architecture experiments

Generate Text

```bash
python src/generate.py
```

Custom Usage

```python
from src.generate import TextGenerator

# Character-level
generator = TextGenerator(
    model_path="models/char_lstm_model.h5",
    tokenizer_path="models/char_tokenizer.pkl",
    is_character_level=True
)

text = generator.generate_text(
    seed_text="to be or not to be",
    num_tokens=100,
    temperature=0.8
)
print(text)
```

 Project Structure

```
lstm-text-generation/
├── src/
│   ├── data_preprocessing.py  # Data loading & tokenization
│   ├── model.py              # LSTM architecture
│   ├── train.py              # Training pipeline
│   └── generate.py           # Text generation
├── models/                   # Saved models
├── outputs/                  # Generated texts
├── requirements.txt
└── README.md
```

 Features

·  Character & word-level tokenization
·  Bidirectional LSTM with dropout
·  Temperature-based sampling
·  Early stopping & model checkpoints
·  Architecture experimentation
·  Multiple seed text generation

 Architecture Experiments

Architecture LSTM Units Bidirectional Accuracy
Small 128 No ~60%
Medium 256 Yes ~70%
Large 512 Yes ~75%

📊 Sample Outputs

Seed: "to be or not to be"

to be or not to be and that is the question of the world and the world of the heart...

Seed: "the king shall"

the king shall make the world and the world of the world and the heart of the...

Seed: "all the world's a stage"

all the world's a stage of the world and the world of the world of the heart...
 Temperature Effects

· 0.5: Conservative, repetitive
· 0.8: Balanced, creative
· 1.2: Random, sometimes nonsensical

 Dependencies

· TensorFlow 2.8+
· NumPy 1.19+
· Matplotlib 3.3+ (for visualization)
· Requests 2.25+ (for data download)

 Model Configuration

```python
# Default architecture
Embedding(256) → Bidirectional(LSTM(512)) → Dropout(0.2) → Dense(vocab_size, softmax)
```

 Performance

· Character-level: ~65-70% accuracy, 60 vocab size
· Word-level: ~50-55% accuracy, 6000 vocab size

🚀 Future Improvements

· Attention mechanism
· Transformer architecture
· Web interface
· Style transfer
· Larger datasets


# Complete setup
git clone <repo-url> && cd lstm-text-generation && pip install -r requirements.txt && mkdir -p data models outputs && python src/train.py && python src/generate.py
```

This short README provides all essential information while remaining concise and easy to read. It includes quick start instructions, features, examples, and technical details in a clean format.

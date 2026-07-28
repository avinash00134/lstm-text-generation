cat > run.sh << 'EOF'
#!/bin/bash

echo "========================================="
echo "LSTM Text Generation Setup and Run"
echo "========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run training
echo "Training models..."
python src/train.py

# Run generation
echo "Generating text..."
python src/generate.py

echo "========================================="
echo "Done! Check the models/ and outputs/ directories"
echo "========================================="
EOF

# Make the script executable
chmod +x run.sh

# Run everything with one command
./run.sh# Runner file

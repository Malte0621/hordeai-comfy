#!/bin/bash
echo "Installing AI Horde dependencies..."
echo

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not available. Please ensure Python and pip are installed."
    exit 1
fi

echo "Installing required packages..."
pip install -r requirements.txt

echo
echo "Installation complete!"
echo "You can now restart ComfyUI to use the AI Horde nodes."
echo

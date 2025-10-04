#! /bin/bash
pip install --ignore-installed Flask
pip install flask-cors PyMuPDF markdown-it-py ultralytics
pip install git+https://github.com/openai/whisper.git
pip insatll gradio pillow hf_transfer transformers soundfile "moviepy==1.0.3"

curl -fsSL https://ollama.com/install.sh | sh
pip install ollama
ollama serve &
sleep 100 && echo "Server Ready..."
ollama pull gemma3:4b

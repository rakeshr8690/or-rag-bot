```bash
# Create virtual environment
python -m venv venv

# Activate it:
venv\Scripts\activate

# Install all packages
pip install -r requirements.txt

# Use one of these
#1 mistralai/Mistral-7B-Instruct-v0.2
$env:HF_API_TOKEN=""
$env:USE_LOCAL_LLM="false"
$env:LLM_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
.\venv\Scripts\python.exe -m src.app

#2 TinyLlama/TinyLlama-1.1B-Chat-v1.0
$env:HF_API_TOKEN=""
$env:USE_LOCAL_LLM="false"
$env:LLM_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
.\venv\Scripts\python.exe -m src.app

# Open the following
http://localhost:5000/

# To exit 
use "Ctrl+C"

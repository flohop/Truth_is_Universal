#!/usr/bin/env bash
set -e

echo "=============================================="
echo "🚀 Setting up the Truth_is_Universal environment"
echo "=============================================="

# 1. Install Conda
echo "📦 Installing Miniconda..."
mkdir -p ~/miniconda3
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

# Initialize and enable conda
source ~/miniconda3/bin/activate
conda init bash || true
source ~/.bashrc || true

# Accept Anaconda Terms of Service automatically
echo "✅ Accepting Anaconda Terms of Service..."
conda config --set always_yes yes
conda config --set report_errors false
conda config --set auto_activate_base false

# Install Git LFS
echo "📥 Installing Git LFS..."
git lfs install
git config --global lfs.concurrenttransfers 10


# --- 2. Clone Llama3 Weights ---
echo "🧠 Cloning Llama3 weights to /quickpod..."
mkdir -p /quickpod/weights
cd /quickpod/weights || exit 1

REPO_URL="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"
TARGET_DIR="Meta-Llama-3-8B-Instruct"

if [ -d "$TARGET_DIR/.git" ]; then
    echo "✅ Repository '$TARGET_DIR' already exists — skipping clone."
else
    read -p "Please enter your Hugging Face token or password for cloning Meta-Llama-3-8B-Instruct: " HF_TOKEN

    echo "📦 Cloning repository from $REPO_URL..."
    # Temporary script to provide token to git
    GIT_ASKPASS_SCRIPT=$(mktemp)
    cat <<EOF > "$GIT_ASKPASS_SCRIPT"
#!/bin/bash
echo "$HF_TOKEN"
EOF
    chmod +x "$GIT_ASKPASS_SCRIPT"

    # Clone using Hugging Face authentication
    GIT_ASKPASS="$GIT_ASKPASS_SCRIPT" git clone "$REPO_URL" "$TARGET_DIR"

    # Clean up token helper
    rm -f "$GIT_ASKPASS_SCRIPT"
fi


# 3. Clone the Truth_is_Universal repo
echo "💾 Cloning the Truth_is_Universal repository..."
cd /workspace
if [ -d "Truth_is_Universal" ]; then
    echo "📂 Truth_is_Universal already exists, skipping clone."
else
    git clone https://github.com/flohop/Truth_is_Universal.git
fi


# 4. Update config.ini
CONFIG_PATH="/workspace/Truth_is_Universal/config.ini"
echo "🛠️  Updating config.ini to point to Llama3 weights..."

cat > "$CONFIG_PATH" <<EOL
[Llama3]
weights_directory = /quickpod/weights
8B_base_subdir = Meta-Llama-3-8B-Instruct
8B_chat_subdir = Meta-Llama-3-8B-Instruct
70B_base_subdir = llama3_70b_hf
70B_chat_subdir = llama3_70b_chat_hf

[Llama2]
weights_directory = ../../../../data/lbuerger/llama_hf
7B_base_subdir = llama2_7b_hf
7B_chat_subdir = llama2_7b_chat_hf
13B_base_subdir = llama2_13b_hf
13B_chat_subdir = llama2_13b_chat_hf
70B_base_subdir = llama2_70b_hf
70B_chat_subdir = llama2_70b_chat_hf

[Gemma]
weights_directory = ../../../../data/lbuerger/gemma
7B_base_subdir = gemma_7b_hf
7B_chat_subdir = gemma_7b_chat_hf

[Gemma2]
weights_directory = ../../../../data/lbuerger/gemma2_hf
27B_chat_subdir = gemma2_27b_it_hf
9B_base_subdir = gemma2_9b_hf

[Mistral]
weights_directory = ../../../../data/lbuerger/mistral_hf
7B_chat_subdir = mistral_7b_it_hf_v3
EOL

# 5. Install Python dependencies
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

echo "🐍 Creating conda environment and installing requirements..."
if ! conda env list | grep -q "truth_is_universal"; then
    conda create -y -n truth_is_universal python=3.11
fi
conda activate truth_is_universal
cd /workspace/Truth_is_Universal
pip install -r requirements.txt


# 6. Ask if user wants to generate acts
read -p "Do you want to generate weights for lie detection now? (y/n): " GENERATE

if [[ "$GENERATE" == "y" || "$GENERATE" == "Y" ]]; then
    echo "⚙️  Generating activations for lie detection..."
    python generate_acts.py --model_family Llama3 --model_size 8B --model_type chat --layers 12 --datasets all --device cuda:0
else
    echo "⏭️  Skipping generation step."
fi

# 7. Start Jupyter server
echo "🌐 Starting Jupyter server..."
nohup jupyter notebook --no-browser --ip=0.0.0.0 --port=8888 > jupyter.log 2>&1 &
sleep 5

TOKEN=$(jupyter notebook list | grep -o 'token=[a-z0-9]*' | head -n 1)

echo "=============================================="
echo "✅ Setup complete!"
echo "----------------------------------------------"
echo "📍 Jupyter Notebook is running."
echo "Connect using this command from your local machine:"
echo
echo "    ssh -L 8888:localhost:8888 <your_user>@<remote_host>"
echo
echo "Then open this in your browser:"
echo "    http://localhost:8888/?$TOKEN"
echo "----------------------------------------------"
echo "🎉 All is finished and ready!"
echo "=============================================="

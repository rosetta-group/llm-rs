#!/bin/bash
set -e  # Exit on error

# =========
# 1. SSH SETUP
# =========
echo "[*] Setting up SSH for GitHub..."
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Write deploy key from RunPod secret
echo "$GITHUB_DEPLOY_KEY" > ~/.ssh/my-github-deploy-key
chmod 600 ~/.ssh/my-github-deploy-key

# Add GitHub to known hosts
ssh-keyscan github.com >> ~/.ssh/known_hosts
chmod 644 ~/.ssh/known_hosts

# SSH config to always use this key
cat <<EOF > ~/.ssh/config
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/my-github-deploy-key
  IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config

# =========
# 2. INSTALL POETRY (if missing)
# =========
if ! command -v poetry &> /dev/null; then
    echo "[*] Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[*] Poetry already installed."
    export PATH="$HOME/.local/bin:$PATH"
fi

# =========
# 3. INSTALL PYTHON DEPENDENCIES
# =========
echo "[*] Installing dependencies with Poetry..."
poetry install

echo "[✓] Setup complete — ready to work."

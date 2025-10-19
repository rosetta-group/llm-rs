#!/bin/bash
set -e  # Exit on error

# =========
# 1. SSH SETUP
# =========
echo "[*] Setting up SSH for GitHub..."
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Write deploy key from RunPod secret
echo "$GITHUB_DEPLOY_KEY" | base64 -d > ~/.ssh/my-github-deploy-key
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
# 2. CLONE OR UPDATE REPO
# =========
REPO_SSH="git@github.com:rosetta-group/llm-rs.git"
APP_DIR=llm-rs

if [ ! -d "$APP_DIR" ]; then
    echo "[*] Cloning repository..."
    git clone "$REPO_SSH" "$APP_DIR"
fi

cd "$APP_DIR"

# =========
# 3. GIT CONFIG (local to this repo)
# =========
echo "[*] Checking Git identity..."

EXISTING_NAME=$(git config --local user.name || true)
EXISTING_EMAIL=$(git config --local user.email || true)

if [ -z "$EXISTING_NAME" ] || [ -z "$EXISTING_EMAIL" ]; then
    echo "No Git identity found for this repo."
    read -p "Enter your name for Git commits: " GIT_NAME
    read -p "Enter your email for Git commits: " GIT_EMAIL
    git config --local user.name "$GIT_NAME"
    git config --local user.email "$GIT_EMAIL"
    echo "[+] Git identity saved locally for this repo."
else
    echo "[*] Using existing Git identity:"
    echo "    Name:  $EXISTING_NAME"
    echo "    Email: $EXISTING_EMAIL"
fi
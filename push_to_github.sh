#!/bin/bash
# Complete script to push code to GitHub
# This will guide you through the process

set -e

echo "=========================================="
echo "Push to GitHub - CSE_590_03_VLM"
echo "=========================================="
echo ""
echo "Authors:"
echo "  - Ranjith Merugu (116842918)"
echo "  - Sambhav Srivastava (117525285)"
echo ""
echo "=========================================="
echo ""

# Check if remote exists
if git remote get-url origin &> /dev/null; then
    CURRENT_REMOTE=$(git remote get-url origin)
    echo "Remote already configured: $CURRENT_REMOTE"
    read -p "Push to existing remote? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Pushing to GitHub..."
        git branch -M main 2>/dev/null || true
        git push -u origin main
        echo ""
        echo "✓ Code pushed successfully!"
        if [[ $CURRENT_REMOTE == *"github.com"* ]]; then
            REPO_URL=$(echo $CURRENT_REMOTE | sed 's/\.git$//' | sed 's/^git@github.com:/https:\/\/github.com\//')
            echo ""
            echo "Repository URL: $REPO_URL"
        fi
        exit 0
    fi
fi

# Get GitHub username
echo "Step 1: Enter your GitHub username"
read -p "GitHub username: " GITHUB_USERNAME

if [ -z "$GITHUB_USERNAME" ]; then
    echo "Error: GitHub username is required"
    exit 1
fi

# Repository name
REPO_NAME="CSE_590_03_VLM"
REPO_URL="https://github.com/$GITHUB_USERNAME/$REPO_NAME"

echo ""
echo "Step 2: Create repository on GitHub"
echo "=========================================="
echo "Please create the repository on GitHub:"
echo ""
echo "1. Go to: https://github.com/new"
echo "2. Repository name: $REPO_NAME"
echo "3. Description: Frequency Prior for Autoregressive Image Generation"
echo "4. Choose Public or Private"
echo "5. DO NOT initialize with README, .gitignore, or license"
echo "6. Click 'Create repository'"
echo ""
read -p "Press Enter after you've created the repository on GitHub..."

# Add remote
echo ""
echo "Step 3: Adding remote and pushing..."
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL.git"

# Rename branch to main
git branch -M main 2>/dev/null || true

# Push
echo "Pushing code to GitHub..."
git push -u origin main

echo ""
echo "=========================================="
echo "✓ Successfully pushed to GitHub!"
echo "=========================================="
echo ""
echo "Repository URL: $REPO_URL"
echo ""
echo "You can view your code at:"
echo "  $REPO_URL"
echo ""


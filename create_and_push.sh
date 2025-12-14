#!/bin/bash
# Create GitHub repository and push code

set -e

GITHUB_USER="meruguranjith123"
REPO_NAME="CSE_590_03_VLM"
REPO_URL="https://github.com/$GITHUB_USER/$REPO_NAME"

echo "=========================================="
echo "Creating GitHub Repository and Pushing Code"
echo "=========================================="
echo "User: $GITHUB_USER"
echo "Repository: $REPO_NAME"
echo "=========================================="
echo ""

# Check if gh CLI is installed
if command -v gh &> /dev/null; then
    echo "✓ GitHub CLI found"
    
    # Check if authenticated
    if gh auth status &> /dev/null; then
        echo "✓ GitHub CLI authenticated"
        
        # Create repository
        echo "Creating repository on GitHub..."
        gh repo create "$REPO_NAME" \
            --public \
            --description "Frequency Prior for Autoregressive Image Generation - CSE 590-03" \
            --source=. \
            --remote=origin \
            --push || {
            
            # If repo already exists, just set remote
            echo "Repository might already exist, setting remote..."
            git remote remove origin 2>/dev/null || true
            git remote add origin "$REPO_URL.git"
        }
        
        # Push code
        echo "Pushing code..."
        git branch -M main
        git push -u origin main
        
        echo ""
        echo "=========================================="
        echo "✓ Successfully created and pushed!"
        echo "=========================================="
        echo "Repository URL: $REPO_URL"
        exit 0
    else
        echo "⚠ GitHub CLI not authenticated"
        echo "Run: gh auth login"
    fi
fi

# Fallback: Manual instructions
echo "GitHub CLI not available or not authenticated"
echo ""
echo "Please create the repository manually:"
echo ""
echo "Option 1: Using GitHub CLI (Recommended)"
echo "  1. Install: brew install gh"
echo "  2. Authenticate: gh auth login"
echo "  3. Run this script again"
echo ""
echo "Option 2: Manual Web Creation"
echo "  1. Go to: https://github.com/new"
echo "  2. Repository name: $REPO_NAME"
echo "  3. Description: Frequency Prior for Autoregressive Image Generation"
echo "  4. Choose Public or Private"
echo "  5. DO NOT initialize with README"
echo "  6. Click 'Create repository'"
echo "  7. Then run: git push -u origin main"
echo ""
echo "Repository URL will be:"
echo "  $REPO_URL"


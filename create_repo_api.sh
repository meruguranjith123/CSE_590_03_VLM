#!/bin/bash
# Create GitHub repository using GitHub API

GITHUB_USER="meruguranjith123"
REPO_NAME="CSE_590_03_VLM"
REPO_URL="https://github.com/$GITHUB_USER/$REPO_NAME"

echo "=========================================="
echo "Creating Repository via GitHub API"
echo "=========================================="

# Check if token is provided
if [ -z "$GITHUB_TOKEN" ]; then
    echo ""
    echo "GitHub Personal Access Token required!"
    echo ""
    echo "To create a token:"
    echo "1. Go to: https://github.com/settings/tokens"
    echo "2. Click 'Generate new token' -> 'Generate new token (classic)'"
    echo "3. Name: 'CSE_590_03_VLM'"
    echo "4. Select scope: 'repo' (full control of private repositories)"
    echo "5. Click 'Generate token'"
    echo "6. Copy the token"
    echo ""
    read -sp "Enter your GitHub Personal Access Token: " GITHUB_TOKEN
    echo ""
fi

if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: Token is required"
    exit 1
fi

# Create repository using GitHub API
echo "Creating repository..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github.v3+json" \
    https://api.github.com/user/repos \
    -d "{
        \"name\": \"$REPO_NAME\",
        \"description\": \"Frequency Prior for Autoregressive Image Generation - CSE 590-03\",
        \"private\": false
    }")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" -eq 201 ]; then
    echo "✓ Repository created successfully!"
    
    # Set remote and push
    echo "Setting up remote..."
    git remote remove origin 2>/dev/null || true
    git remote add origin "$REPO_URL.git"
    git branch -M main
    
    echo "Pushing code..."
    git push -u origin main
    
    echo ""
    echo "=========================================="
    echo "✓ Successfully created and pushed!"
    echo "=========================================="
    echo "Repository URL: $REPO_URL"
    
elif [ "$HTTP_CODE" -eq 422 ]; then
    echo "Repository already exists!"
    echo "Setting up remote and pushing..."
    git remote remove origin 2>/dev/null || true
    git remote add origin "$REPO_URL.git"
    git branch -M main
    git push -u origin main
    echo "✓ Code pushed successfully!"
    echo "Repository URL: $REPO_URL"
else
    echo "✗ Error creating repository (HTTP $HTTP_CODE)"
    echo "$BODY"
    exit 1
fi


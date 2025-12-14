#!/bin/bash
# Push to GitHub after creating repository

echo "Pushing to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Successfully pushed to GitHub!"
    echo ""
    echo "Repository URL:"
    echo "https://github.com/meruguranjith123/CSE_590_03_VLM"
else
    echo ""
    echo "✗ Push failed. Make sure:"
    echo "1. Repository exists at: https://github.com/meruguranjith123/CSE_590_03_VLM"
    echo "2. You're authenticated (git config or GitHub CLI)"
    echo ""
    echo "To authenticate, you can use:"
    echo "  gh auth login"
    echo "Or use a personal access token when prompted for password"
fi


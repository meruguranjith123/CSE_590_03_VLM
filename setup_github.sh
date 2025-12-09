#!/bin/bash
# Setup script to initialize git and prepare for GitHub push

set -e

echo "=========================================="
echo "GitHub Repository Setup"
echo "=========================================="
echo "Repository: CSE_590_03_VLM"
echo "Authors:"
echo "  - Ranjith Merugu (116842918)"
echo "  - Sambhav Srivastava (117525285)"
echo "=========================================="
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Error: Git is not installed!"
    echo "Please install Git first:"
    echo "  macOS: brew install git"
    echo "  Linux: sudo apt-get install git"
    exit 1
fi

# Initialize git if not already
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    echo "✓ Git repository initialized"
else
    echo "✓ Git repository already exists"
fi

# Check if remote exists
if git remote get-url origin &> /dev/null; then
    echo "Remote 'origin' already exists:"
    git remote get-url origin
    read -p "Do you want to change it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git remote remove origin
    else
        echo "Keeping existing remote"
        exit 0
    fi
fi

# Add all files
echo ""
echo "Adding files to git..."
git add .

# Create initial commit if needed
if ! git rev-parse --verify HEAD &> /dev/null; then
    echo "Creating initial commit..."
    git commit -m "Initial commit: Frequency Prior for Autoregressive Image Generation

- Implemented frequency prior autoregressive image generation (mid-term method)
- Added prompt-based conditional generation extension
- Implemented LLM-based text-to-image generation architecture
- Multi-GPU training support
- Support for CIFAR-100, ImageNet-10k, and COCO datasets
- Google Cloud deployment scripts with auto dataset download

Authors:
- Ranjith Merugu (116842918)
- Sambhav Srivastava (117525285)

Course: CSE 590-03 (VLM)"
    echo "✓ Initial commit created"
else
    echo "✓ Repository already has commits"
fi

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Create a new repository on GitHub:"
echo "   - Go to: https://github.com/new"
echo "   - Repository name: CSE_590_03_VLM"
echo "   - Description: Frequency Prior for Autoregressive Image Generation"
echo "   - Make it Public or Private (your choice)"
echo "   - DO NOT initialize with README, .gitignore, or license"
echo ""
echo "2. Add the remote and push:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/CSE_590_03_VLM.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "Or if you want to use SSH:"
echo "   git remote add origin git@github.com:YOUR_USERNAME/CSE_590_03_VLM.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "=========================================="


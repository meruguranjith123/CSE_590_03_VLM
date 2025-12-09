# Push to GitHub - Quick Guide

## Repository Name: CSE_590_03_VLM

## Authors
- Ranjith Merugu (116842918)
- Sambhav Srivastava (117525285)

## Step-by-Step Instructions

### Option 1: Automated Setup (Recommended)

```bash
# Run the setup script
bash setup_github.sh

# Then follow the instructions shown
```

### Option 2: Manual Setup

#### 1. Initialize Git (if not done)
```bash
cd /Users/meruguranjith/VLM
git init
```

#### 2. Create GitHub Repository
1. Go to https://github.com/new
2. Repository name: **CSE_590_03_VLM**
3. Description: "Frequency Prior for Autoregressive Image Generation"
4. Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license
6. Click "Create repository"

#### 3. Add All Files and Commit
```bash
# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Frequency Prior for Autoregressive Image Generation

Authors:
- Ranjith Merugu (116842918)
- Sambhav Srivastava (117525285)

Course: CSE 590-03 (VLM)"
```

#### 4. Add Remote and Push
```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/CSE_590_03_VLM.git

# Rename branch to main
git branch -M main

# Push to GitHub
git push -u origin main
```

### Using SSH (Alternative)

If you prefer SSH instead of HTTPS:

```bash
git remote add origin git@github.com:YOUR_USERNAME/CSE_590_03_VLM.git
git branch -M main
git push -u origin main
```

### Verify Push

After pushing, visit:
```
https://github.com/YOUR_USERNAME/CSE_590_03_VLM
```

You should see all files including:
- README.md (with author information)
- AUTHORS.md
- All model files
- Training scripts
- Dataset loaders
- Cloud setup scripts

## Files Included

The repository includes:
- ✅ Core model architectures
- ✅ Training scripts (base and LLM prompt)
- ✅ Generation scripts
- ✅ Dataset loaders (CIFAR-100, ImageNet, COCO)
- ✅ Text-image datasets for LLM training
- ✅ Multi-GPU training support
- ✅ Google Cloud deployment scripts
- ✅ Auto dataset download functionality
- ✅ Complete documentation
- ✅ Author information

## Repository Structure

```
CSE_590_03_VLM/
├── README.md                    # Main documentation
├── AUTHORS.md                   # Author information
├── models/                      # Model architectures
├── datasets/                    # Dataset loaders
├── cloud_setup/                 # Cloud deployment scripts
├── train.py                     # Base model training
├── train_llm_prompt.py         # LLM prompt training
├── generate.py                  # Image generation
├── generate_llm_prompt.py      # LLM prompt generation
├── run_gpu_training.py         # GPU training script
├── requirements.txt             # Dependencies
└── setup_github.sh             # GitHub setup script
```

## Troubleshooting

### Authentication Issues
If you get authentication errors:
```bash
# Use personal access token instead of password
# Or setup SSH keys: https://docs.github.com/en/authentication
```

### Large Files
If you have large files that shouldn't be committed:
- Add them to `.gitignore`
- Files in `data/`, `checkpoints/`, `logs/` are already ignored

### Push Conflicts
If you need to force push (be careful!):
```bash
git push -u origin main --force
```

## Next Steps After Push

1. **Add repository description** on GitHub
2. **Add topics/tags**: `vision-language-models`, `image-generation`, `autoregressive`, `pytorch`
3. **Add collaborators** if working as a team
4. **Set up GitHub Actions** for CI/CD (optional)
5. **Create releases** when ready to publish


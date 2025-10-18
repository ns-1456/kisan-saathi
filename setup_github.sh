#!/bin/bash

# GitHub Repository Setup Script for Kisan Saathi
# This script helps set up the GitHub repository with proper structure

echo "🚀 Setting up Kisan Saathi GitHub Repository"
echo "=============================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Git repository not initialized. Please run 'git init' first."
    exit 1
fi

# Add remote repository (user needs to create this on GitHub first)
echo "📝 To complete the setup:"
echo "1. Create a new repository on GitHub named 'kisan-saathi'"
echo "2. Run the following commands:"
echo ""
echo "git remote add origin https://github.com/YOUR_USERNAME/kisan-saathi.git"
echo "git branch -M main"
echo "git push -u origin main"
echo ""

# Create GitHub issue templates
echo "📋 Creating GitHub issue templates..."

mkdir -p .github/ISSUE_TEMPLATE

cat > .github/ISSUE_TEMPLATE/sprint-task.md << 'EOF'
---
name: Sprint Task
about: Create a task for the current sprint
title: '[SPRINT-X] Task Title'
labels: ['task', 'sprint']
assignees: ''
---

## Task Description
Brief description of the task

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Estimated Time
X days

## Dependencies
- Dependency 1
- Dependency 2

## Additional Notes
Any additional information or context
EOF

cat > .github/ISSUE_TEMPLATE/bug-report.md << 'EOF'
---
name: Bug Report
about: Report a bug in the Kisan Saathi project
title: '[BUG] Brief description'
labels: ['bug']
assignees: ''
---

## Bug Description
A clear and concise description of what the bug is.

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
A clear and concise description of what you expected to happen.

## Actual Behavior
A clear and concise description of what actually happened.

## Environment
- OS: [e.g. Ubuntu 22.04]
- Python Version: [e.g. 3.9]
- GPU: [e.g. NVIDIA T4]

## Additional Context
Add any other context about the problem here.
EOF

# Create pull request template
mkdir -p .github/pull_request_template

cat > .github/pull_request_template.md << 'EOF'
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced

## Related Issues
Closes #(issue number)
EOF

# Create GitHub Actions workflow
mkdir -p .github/workflows

cat > .github/workflows/ci.yml << 'EOF'
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Lint with flake8
      run: |
        pip install flake8
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    
    - name: Test with pytest
      run: |
        pip install pytest
        pytest tests/ -v
EOF

# Create project board configuration
cat > .github/project-board.md << 'EOF'
# Kisan Saathi - Project Board

## Sprint 1: Infrastructure Setup & Data Collection (Weeks 1-2)
- [ ] Set up AWS EC2 GPU instance with CUDA support
- [ ] Configure Python environment with ML dependencies
- [ ] Download and process AI4Bharat IndicCorp Gujarati dataset
- [ ] Scrape Gujarati agricultural resources from government websites
- [ ] Create initial Q&A dataset for instruction tuning

## Sprint 2: Data Processing & Knowledge Base Creation (Weeks 3-4)
- [ ] Clean and normalize Gujarati text data
- [ ] Structure agricultural documents into JSON format
- [ ] Build FAISS vector store with agricultural knowledge embeddings
- [ ] Create train/val/test splits for fine-tuning datasets

## Sprint 3: Base Model Selection & Baseline (Weeks 5-6)
- [ ] Download and evaluate candidate SLMs (Gemma-2B, Llama-3.2-1B, Bloomz-1B7)
- [ ] Prepare training datasets for LoRA fine-tuning
- [ ] Configure LoRA hyperparameters and training environment

## Sprint 4: LoRA Fine-tuning (Weeks 7-8)
- [ ] Fine-tune SLM with LoRA on domain adaptation corpus
- [ ] Instruction fine-tuning on Q&A dataset
- [ ] Evaluate fine-tuned model vs base model

## Sprint 5: RAG Pipeline Implementation (Weeks 9-10)
- [ ] Implement RAG pipeline with LangChain integration
- [ ] Optimize retrieval quality and prompt engineering
- [ ] Evaluate RAG system performance

## Sprint 6: Model Quantization & Mobile Optimization (Weeks 11-12)
- [ ] Quantize fine-tuned SLM to 4-bit using bitsandbytes
- [ ] Convert quantized model to mobile deployment formats
- [ ] Package deployment bundle with model and vector store

## Sprint 7: CNN for Crop Disease Detection (Weeks 13-14)
- [ ] Download and prepare PlantVillage crop disease dataset
- [ ] Train MobileNetV2 CNN for disease classification
- [ ] Evaluate CNN performance and create confusion matrix

## Sprint 8: Multi-modal Integration & Final Testing (Weeks 15-16)
- [ ] Integrate CNN and RAG+SLM into unified system
- [ ] Comprehensive system testing and performance optimization
- [ ] Create comprehensive documentation and deployment guide
EOF

echo "✅ GitHub templates and workflows created!"
echo ""
echo "📋 Next steps:"
echo "1. Create GitHub repository: https://github.com/new"
echo "2. Add remote origin: git remote add origin https://github.com/YOUR_USERNAME/kisan-saathi.git"
echo "3. Push initial commit: git push -u origin main"
echo "4. Create GitHub issues using the generated issue templates"
echo "5. Set up project board with sprint tasks"
echo ""
echo "🎯 Ready to start Sprint 1 implementation!"

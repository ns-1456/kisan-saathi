#!/bin/bash

# Assign Issues to Milestones and Create Sequence for Kisan Saathi
# This script assigns each issue to its corresponding milestone and creates a logical sequence

echo "🔗 Assigning Issues to Milestones and Creating Sequence"
echo "======================================================"

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed!"
    echo "Please install it from: https://cli.github.com/"
    exit 1
fi

# Check if user is authenticated
if ! gh auth status &> /dev/null; then
    echo "❌ Please authenticate with GitHub CLI first:"
    echo "Run: gh auth login"
    exit 1
fi

echo "✅ GitHub CLI authenticated"

# Get milestone numbers
echo "📋 Getting milestone numbers..."
MILESTONE_1=$(gh api repos/ns-1456/kisan-saathi/milestones --jq '.[] | select(.title | contains("Sprint 1")) | .number')
MILESTONE_2=$(gh api repos/ns-1456/kisan-saathi/milestones --jq '.[] | select(.title | contains("Sprint 2")) | .number')
MILESTONE_3=$(gh api repos/ns-1456/kisan-saathi/milestones --jq '.[] | select(.title | contains("Sprint 3")) | .number')
MILESTONE_4=$(gh api repos/ns-1456/kisan-saathi/milestones --jq '.[] | select(.title | contains("Sprint 4")) | .number')
MILESTONE_5=$(gh api repos/ns-1456/kisan-saathi/milestones --jq '.[] | select(.title | contains("Sprint 5")) | .number')
MILESTONE_6=$(gh api repos/ns-1456/kisan-saathi/milestones --jq '.[] | select(.title | contains("Sprint 6")) | .number')
MILESTONE_7=$(gh api repos/ns-1456/kisan-saathi/milestones --jq '.[] | select(.title | contains("Sprint 7")) | .number')
MILESTONE_8=$(gh api repos/ns-1456/kisan-saathi/milestones --jq '.[] | select(.title | contains("Sprint 8")) | .number')

echo "📅 Milestone numbers:"
echo "  Sprint 1: $MILESTONE_1"
echo "  Sprint 2: $MILESTONE_2"
echo "  Sprint 3: $MILESTONE_3"
echo "  Sprint 4: $MILESTONE_4"
echo "  Sprint 5: $MILESTONE_5"
echo "  Sprint 6: $MILESTONE_6"
echo "  Sprint 7: $MILESTONE_7"
echo "  Sprint 8: $MILESTONE_8"
echo ""

# Function to assign issue to milestone
assign_issue_to_milestone() {
    local issue_number=$1
    local milestone_number=$2
    local sprint_name=$3
    
    echo "Assigning Issue #$issue_number to $sprint_name (Milestone $milestone_number)"
    
    # Update issue with milestone
    gh api repos/ns-1456/kisan-saathi/issues/$issue_number -X PATCH -f milestone="$milestone_number" > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "  ✅ Successfully assigned Issue #$issue_number to $sprint_name"
    else
        echo "  ❌ Failed to assign Issue #$issue_number to $sprint_name"
    fi
}

# Function to update issue title with sequence
update_issue_title() {
    local issue_number=$1
    local new_title=$2
    
    echo "Updating Issue #$issue_number title: $new_title"
    
    # Update issue title
    gh api repos/ns-1456/kisan-saathi/issues/$issue_number -X PATCH -f title="$new_title" > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "  ✅ Successfully updated Issue #$issue_number title"
    else
        echo "  ❌ Failed to update Issue #$issue_number title"
    fi
}

echo "🎯 Assigning Sprint 1 Issues (Issues #1-5)..."

# Sprint 1: Infrastructure Setup & Data Collection
assign_issue_to_milestone 1 $MILESTONE_1 "Sprint 1"
update_issue_title 1 "1.1 Set up AWS EC2 GPU instance with CUDA support"

assign_issue_to_milestone 2 $MILESTONE_1 "Sprint 1"
update_issue_title 2 "1.2 Configure Python environment with ML dependencies"

assign_issue_to_milestone 3 $MILESTONE_1 "Sprint 1"
update_issue_title 3 "1.3 Download and process AI4Bharat IndicCorp Gujarati dataset"

assign_issue_to_milestone 4 $MILESTONE_1 "Sprint 1"
update_issue_title 4 "1.4 Scrape Gujarati agricultural resources from government websites"

assign_issue_to_milestone 5 $MILESTONE_1 "Sprint 1"
update_issue_title 5 "1.5 Create initial Q&A dataset for instruction tuning"

echo "✅ Sprint 1 issues assigned!"

echo ""
echo "🎯 Assigning Sprint 2 Issues (Issues #6-9)..."

# Sprint 2: Data Processing & Knowledge Base Creation
assign_issue_to_milestone 6 $MILESTONE_2 "Sprint 2"
update_issue_title 6 "2.1 Clean and normalize Gujarati text data"

assign_issue_to_milestone 7 $MILESTONE_2 "Sprint 2"
update_issue_title 7 "2.2 Structure agricultural documents into JSON format"

assign_issue_to_milestone 8 $MILESTONE_2 "Sprint 2"
update_issue_title 8 "2.3 Build FAISS vector store with agricultural knowledge embeddings"

assign_issue_to_milestone 9 $MILESTONE_2 "Sprint 2"
update_issue_title 9 "2.4 Create train/val/test splits for fine-tuning datasets"

echo "✅ Sprint 2 issues assigned!"

echo ""
echo "🎯 Assigning Sprint 3 Issues (Issues #10-12)..."

# Sprint 3: Base Model Selection & Baseline
assign_issue_to_milestone 10 $MILESTONE_3 "Sprint 3"
update_issue_title 10 "3.1 Download and evaluate candidate SLMs (Gemma-2B, Llama-3.2-1B, Bloomz-1B7)"

assign_issue_to_milestone 11 $MILESTONE_3 "Sprint 3"
update_issue_title 11 "3.2 Prepare training datasets for LoRA fine-tuning"

assign_issue_to_milestone 12 $MILESTONE_3 "Sprint 3"
update_issue_title 12 "3.3 Configure LoRA hyperparameters and training environment"

echo "✅ Sprint 3 issues assigned!"

echo ""
echo "🎯 Assigning Sprint 4 Issues (Issues #13-15)..."

# Sprint 4: LoRA Fine-tuning
assign_issue_to_milestone 13 $MILESTONE_4 "Sprint 4"
update_issue_title 13 "4.1 Fine-tune SLM with LoRA on domain adaptation corpus"

assign_issue_to_milestone 14 $MILESTONE_4 "Sprint 4"
update_issue_title 14 "4.2 Instruction fine-tuning on Q&A dataset"

assign_issue_to_milestone 15 $MILESTONE_4 "Sprint 4"
update_issue_title 15 "4.3 Evaluate fine-tuned model vs base model"

echo "✅ Sprint 4 issues assigned!"

echo ""
echo "🎯 Assigning Sprint 5 Issues (Issues #16-18)..."

# Sprint 5: RAG Pipeline Implementation
assign_issue_to_milestone 16 $MILESTONE_5 "Sprint 5"
update_issue_title 16 "5.1 Implement RAG pipeline with LangChain integration"

assign_issue_to_milestone 17 $MILESTONE_5 "Sprint 5"
update_issue_title 17 "5.2 Optimize retrieval quality and prompt engineering"

assign_issue_to_milestone 18 $MILESTONE_5 "Sprint 5"
update_issue_title 18 "5.3 Evaluate RAG system performance"

echo "✅ Sprint 5 issues assigned!"

echo ""
echo "🎯 Assigning Sprint 6 Issues (Issues #19-21)..."

# Sprint 6: Model Quantization & Mobile Optimization
assign_issue_to_milestone 19 $MILESTONE_6 "Sprint 6"
update_issue_title 19 "6.1 Quantize fine-tuned SLM to 4-bit using bitsandbytes"

assign_issue_to_milestone 20 $MILESTONE_6 "Sprint 6"
update_issue_title 20 "6.2 Convert quantized model to mobile deployment formats"

assign_issue_to_milestone 21 $MILESTONE_6 "Sprint 6"
update_issue_title 21 "6.3 Package deployment bundle with model and vector store"

echo "✅ Sprint 6 issues assigned!"

echo ""
echo "🎯 Assigning Sprint 7 Issues (Issues #22-24)..."

# Sprint 7: CNN for Crop Disease Detection
assign_issue_to_milestone 22 $MILESTONE_7 "Sprint 7"
update_issue_title 22 "7.1 Download and prepare PlantVillage crop disease dataset"

assign_issue_to_milestone 23 $MILESTONE_7 "Sprint 7"
update_issue_title 23 "7.2 Train MobileNetV2 CNN for disease classification"

assign_issue_to_milestone 24 $MILESTONE_7 "Sprint 7"
update_issue_title 24 "7.3 Evaluate CNN performance and create confusion matrix"

echo "✅ Sprint 7 issues assigned!"

echo ""
echo "🎯 Assigning Sprint 8 Issues (Issues #25-27)..."

# Sprint 8: Multi-modal Integration & Final Testing
assign_issue_to_milestone 25 $MILESTONE_8 "Sprint 8"
update_issue_title 25 "8.1 Integrate CNN and RAG+SLM into unified system"

assign_issue_to_milestone 26 $MILESTONE_8 "Sprint 8"
update_issue_title 26 "8.2 Comprehensive system testing and performance optimization"

assign_issue_to_milestone 27 $MILESTONE_8 "Sprint 8"
update_issue_title 27 "8.3 Create comprehensive documentation and deployment guide"

echo "✅ Sprint 8 issues assigned!"

echo ""
echo "🎉 All issues successfully assigned to milestones and sequenced!"
echo ""
echo "📊 Issue Sequence Summary:"
echo ""
echo "🚀 Sprint 1: Infrastructure Setup & Data Collection"
echo "  1.1 Set up AWS EC2 GPU instance with CUDA support"
echo "  1.2 Configure Python environment with ML dependencies"
echo "  1.3 Download and process AI4Bharat IndicCorp Gujarati dataset"
echo "  1.4 Scrape Gujarati agricultural resources from government websites"
echo "  1.5 Create initial Q&A dataset for instruction tuning"
echo ""
echo "📊 Sprint 2: Data Processing & Knowledge Base Creation"
echo "  2.1 Clean and normalize Gujarati text data"
echo "  2.2 Structure agricultural documents into JSON format"
echo "  2.3 Build FAISS vector store with agricultural knowledge embeddings"
echo "  2.4 Create train/val/test splits for fine-tuning datasets"
echo ""
echo "🧠 Sprint 3: Base Model Selection & Baseline"
echo "  3.1 Download and evaluate candidate SLMs (Gemma-2B, Llama-3.2-1B, Bloomz-1B7)"
echo "  3.2 Prepare training datasets for LoRA fine-tuning"
echo "  3.3 Configure LoRA hyperparameters and training environment"
echo ""
echo "🎯 Sprint 4: LoRA Fine-tuning"
echo "  4.1 Fine-tune SLM with LoRA on domain adaptation corpus"
echo "  4.2 Instruction fine-tuning on Q&A dataset"
echo "  4.3 Evaluate fine-tuned model vs base model"
echo ""
echo "🔍 Sprint 5: RAG Pipeline Implementation"
echo "  5.1 Implement RAG pipeline with LangChain integration"
echo "  5.2 Optimize retrieval quality and prompt engineering"
echo "  5.3 Evaluate RAG system performance"
echo ""
echo "⚡ Sprint 6: Model Quantization & Mobile Optimization"
echo "  6.1 Quantize fine-tuned SLM to 4-bit using bitsandbytes"
echo "  6.2 Convert quantized model to mobile deployment formats"
echo "  6.3 Package deployment bundle with model and vector store"
echo ""
echo "🌾 Sprint 7: CNN for Crop Disease Detection"
echo "  7.1 Download and prepare PlantVillage crop disease dataset"
echo "  7.2 Train MobileNetV2 CNN for disease classification"
echo "  7.3 Evaluate CNN performance and create confusion matrix"
echo ""
echo "🔗 Sprint 8: Multi-modal Integration & Final Testing"
echo "  8.1 Integrate CNN and RAG+SLM into unified system"
echo "  8.2 Comprehensive system testing and performance optimization"
echo "  8.3 Create comprehensive documentation and deployment guide"
echo ""
echo "🔗 Repository Links:"
echo "  📊 Repository: https://github.com/ns-1456/kisan-saathi"
echo "  📋 Issues: https://github.com/ns-1456/kisan-saathi/issues"
echo "  🎯 Milestones: https://github.com/ns-1456/kisan-saathi/milestones"
echo ""
echo "🚀 Ready to start Sprint 1 implementation!"

#!/bin/bash

# Create GitHub Issues for Kisan Saathi (Simplified Version)
# This script creates all issues without labels to avoid label errors

echo "🚀 Creating GitHub Issues for Kisan Saathi"
echo "=========================================="

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

# Create Sprint 1 Issues
echo "🎯 Creating Sprint 1 Issues..."

gh issue create --title "Set up AWS EC2 GPU instance with CUDA support" \
  --body "## Task: AWS EC2 Setup

### Objective
Launch and configure AWS EC2 g4dn.xlarge Spot Instance for model training.

### Requirements
- Ubuntu 22.04 LTS
- NVIDIA drivers and CUDA toolkit (11.8 or 12.1)
- 100GB EBS storage
- Security groups configured for SSH and Jupyter access
- Elastic IP for persistent access

### Acceptance Criteria
- [ ] Instance launched successfully
- [ ] CUDA toolkit installed and verified
- [ ] Jupyter notebook accessible
- [ ] Cost monitoring enabled
- [ ] Documentation updated

### Estimated Time
2-3 days

### Dependencies
None" \
  --milestone "$MILESTONE_1"

gh issue create --title "Configure Python environment with ML dependencies" \
  --body "## Task: Python Environment Setup

### Objective
Install and configure Python environment with all required ML libraries.

### Requirements
- Python 3.8+ virtual environment
- PyTorch with CUDA support
- Transformers, datasets, accelerate, peft, bitsandbytes
- sentence-transformers, faiss-gpu, langchain
- Weights & Biases for experiment tracking

### Acceptance Criteria
- [ ] Virtual environment created
- [ ] All dependencies installed successfully
- [ ] CUDA support verified
- [ ] Test imports working
- [ ] Wandb configured

### Estimated Time
1 day

### Dependencies
AWS EC2 instance setup" \
  --milestone "$MILESTONE_1"

gh issue create --title "Download and process AI4Bharat IndicCorp Gujarati dataset" \
  --body "## Task: IndicCorp Dataset Processing

### Objective
Download and preprocess AI4Bharat IndicCorp Gujarati subset for model training.

### Requirements
- Download Gujarati subset from IndicCorp v2
- Extract and decompress (~2-5GB expected)
- Perform quality checks (encoding, language detection)
- Sample 100K sentences for initial experiments
- Create train/val/test splits

### Acceptance Criteria
- [ ] Dataset downloaded successfully
- [ ] Quality checks passed
- [ ] Sample dataset created
- [ ] Data splits generated
- [ ] Documentation updated

### Estimated Time
2-3 days

### Dependencies
Python environment setup" \
  --milestone "$MILESTONE_1"

gh issue create --title "Scrape Gujarati agricultural resources from government websites" \
  --body "## Task: Government Resource Collection

### Objective
Collect domain-specific Gujarati agricultural content from official sources.

### Requirements
- Gujarat State Agriculture Department reports
- KVK (Krishi Vigyan Kendra) manuals and FAQs
- ICAR Gujarat research bulletins
- Focus on major crops: cotton, groundnut, wheat, bajra
- Extract text from PDFs using pdfplumber

### Acceptance Criteria
- [ ] Web scraping scripts created
- [ ] PDF extraction working
- [ ] 500+ documents collected
- [ ] Text cleaned and normalized
- [ ] Metadata preserved

### Estimated Time
3-4 days

### Dependencies
Python environment setup" \
  --milestone "$MILESTONE_1"

gh issue create --title "Create initial Q&A dataset for instruction tuning" \
  --body "## Task: Q&A Dataset Creation

### Objective
Manually curate Gujarati agricultural Q&A pairs for instruction tuning.

### Requirements
- 500-1000 Q&A pairs covering:
  - Soil preparation
  - Irrigation schedules
  - Pest/disease management
  - Fertilizer recommendations
  - Weather-based advisories
- Format: JSON with fields \`{question_gu, answer_gu, category, crop_type}\`
- Use back-translation for data augmentation

### Acceptance Criteria
- [ ] 500+ Q&A pairs created
- [ ] Categories balanced
- [ ] Data augmentation applied
- [ ] JSON format standardized
- [ ] Quality validation completed

### Estimated Time
4-5 days

### Dependencies
Government resource collection" \
  --milestone "$MILESTONE_1"

echo "✅ Sprint 1 issues created!"

# Create Sprint 2 Issues
echo ""
echo "🎯 Creating Sprint 2 Issues..."

gh issue create --title "Clean and normalize Gujarati text data" \
  --body "## Task: Text Normalization Pipeline

### Objective
Create robust text cleaning pipeline for Gujarati agricultural content.

### Requirements
- Remove HTML tags, special characters, excessive whitespace
- Standardize Unicode encoding (NFC normalization)
- Language detection: filter non-Gujarati content
- Deduplication using MinHash
- Handle encoding issues

### Acceptance Criteria
- [ ] Normalization pipeline created
- [ ] Language detection working
- [ ] Deduplication implemented
- [ ] Quality metrics established
- [ ] Pipeline tested on sample data

### Estimated Time
2-3 days

### Dependencies
IndicCorp dataset processing" \
  --milestone "$MILESTONE_2"

gh issue create --title "Structure agricultural documents into JSON format" \
  --body "## Task: Document Structuring

### Objective
Parse and structure agricultural documents into machine-readable format.

### Requirements
- Extract structured sections from government reports:
  - Crop name/type
  - Growth stage information
  - Pest/disease descriptions
  - Treatment recommendations
  - Seasonal timing
- Convert to JSON/JSONL format with metadata
- Preserve document relationships

### Acceptance Criteria
- [ ] Parsing scripts created
- [ ] 5000+ documents structured
- [ ] Metadata extraction working
- [ ] JSON format standardized
- [ ] Quality validation completed

### Estimated Time
3-4 days

### Dependencies
Government resource collection" \
  --milestone "$MILESTONE_2"

gh issue create --title "Build FAISS vector store with agricultural knowledge embeddings" \
  --body "## Task: Vector Store Creation

### Objective
Generate embeddings and create searchable vector store for RAG pipeline.

### Requirements
- Evaluate multilingual embedding models
- Generate embeddings for 5000+ documents
- Create FAISS index (IndexFlatIP or IndexIVFFlat)
- Optimize for offline deployment
- Test retrieval accuracy

### Acceptance Criteria
- [ ] Embedding model selected
- [ ] Vector store created
- [ ] Retrieval tested
- [ ] Performance benchmarks established
- [ ] Index saved to disk

### Estimated Time
2-3 days

### Dependencies
Document structuring" \
  --milestone "$MILESTONE_2"

gh issue create --title "Create train/val/test splits for fine-tuning datasets" \
  --body "## Task: Dataset Splits

### Objective
Create proper train/validation/test splits for model training.

### Requirements
- Domain adaptation corpus: 80% train, 10% val, 10% test
- Q&A instruction dataset: 70% train, 15% val, 15% test
- Stratify by category for balanced coverage
- Ensure no data leakage
- Document split statistics

### Acceptance Criteria
- [ ] Splits created for all datasets
- [ ] Stratification implemented
- [ ] Statistics documented
- [ ] No data leakage verified
- [ ] Split files saved

### Estimated Time
1 day

### Dependencies
Q&A dataset creation" \
  --milestone "$MILESTONE_2"

echo "✅ Sprint 2 issues created!"

# Create Sprint 3 Issues
echo ""
echo "🎯 Creating Sprint 3 Issues..."

gh issue create --title "Download and evaluate candidate SLMs (Gemma-2B, Llama-3.2-1B, Bloomz-1B7)" \
  --body "## Task: SLM Evaluation

### Objective
Compare candidate small language models for Gujarati agricultural domain.

### Requirements
- Download Gemma-2B-it, Llama-3.2-1B-Instruct, Bloomz-1B7
- Test Gujarati generation capability
- Measure model size, inference latency, memory usage
- Human evaluation on 50 test prompts
- Automated metrics: perplexity on Gujarati held-out set

### Acceptance Criteria
- [ ] All models downloaded
- [ ] Gujarati capability tested
- [ ] Performance benchmarks completed
- [ ] Evaluation report created
- [ ] Final model selected

### Estimated Time
3-4 days

### Dependencies
Dataset splits creation" \
  --milestone "$MILESTONE_3"

gh issue create --title "Prepare training datasets for LoRA fine-tuning" \
  --body "## Task: Training Data Preparation

### Objective
Format datasets for LoRA fine-tuning on selected base model.

### Requirements
- Domain adaptation dataset: plain text, continuous documents
- Instruction tuning dataset: Alpaca/ShareGPT format
- Tokenize with selected model's tokenizer
- Create chunks of 512-1024 tokens
- Validate data quality

### Acceptance Criteria
- [ ] Datasets formatted correctly
- [ ] Tokenization working
- [ ] Chunking implemented
- [ ] Quality validation passed
- [ ] Training configs created

### Estimated Time
2 days

### Dependencies
SLM evaluation" \
  --milestone "$MILESTONE_3"

gh issue create --title "Configure LoRA hyperparameters and training environment" \
  --body "## Task: LoRA Configuration

### Objective
Set up LoRA fine-tuning configuration and training environment.

### Requirements
- Research LoRA hyperparameters for selected model
- Configure target modules: q_proj, v_proj, k_proj, o_proj
- Set LoRA rank (r=8-16), alpha (16-32), dropout (0.05-0.1)
- Plan training budget and AWS Spot Instance schedule
- Set up checkpointing and monitoring

### Acceptance Criteria
- [ ] LoRA config created
- [ ] Training environment validated
- [ ] Budget planning completed
- [ ] Monitoring setup configured
- [ ] Documentation updated

### Estimated Time
1-2 days

### Dependencies
Training data preparation" \
  --milestone "$MILESTONE_3"

echo "✅ Sprint 3 issues created!"

# Create Sprint 4 Issues
echo ""
echo "🎯 Creating Sprint 4 Issues..."

gh issue create --title "Fine-tune SLM with LoRA on domain adaptation corpus" \
  --body "## Task: Domain Adaptation Training

### Objective
Fine-tune base model on Gujarati agricultural corpus using LoRA.

### Requirements
- Phase 1: Continued pre-training on Gujarati corpus
- Learning rate: 2e-4, batch size: 4, gradient accumulation: 4
- Epochs: 2-3, max sequence length: 1024
- Monitor training loss and validation perplexity
- Save checkpoints every 500 steps

### Acceptance Criteria
- [ ] Training script implemented
- [ ] Domain adaptation completed
- [ ] Loss curves logged
- [ ] Checkpoints saved
- [ ] Quality checks passed

### Estimated Time
3-4 days

### Dependencies
LoRA configuration" \
  --milestone "$MILESTONE_4"

gh issue create --title "Instruction fine-tuning on Q&A dataset" \
  --body "## Task: Instruction Tuning

### Objective
Fine-tune domain-adapted model on Q&A instruction dataset.

### Requirements
- Load domain-adapted model from previous phase
- Fine-tune on Q&A instruction dataset
- Learning rate: 1e-4 (lower than domain adaptation)
- Epochs: 3-5, batch size: 4
- Monitor instruction-following capability

### Acceptance Criteria
- [ ] Instruction tuning completed
- [ ] Model performance evaluated
- [ ] Quality improvements verified
- [ ] Final model saved
- [ ] Evaluation report created

### Estimated Time
2-3 days

### Dependencies
Domain adaptation training" \
  --milestone "$MILESTONE_4"

gh issue create --title "Evaluate fine-tuned model vs base model" \
  --body "## Task: Model Evaluation

### Objective
Comprehensive evaluation of fine-tuned model performance.

### Requirements
- Automated metrics: perplexity, BLEU/ROUGE scores
- Human evaluation on 50 test queries
- Compare fine-tuned vs base model
- Measure: fluency, correctness, relevance, domain accuracy
- Identify remaining weaknesses

### Acceptance Criteria
- [ ] Automated evaluation completed
- [ ] Human evaluation conducted
- [ ] Comparison report created
- [ ] Weaknesses identified
- [ ] Improvement plan created

### Estimated Time
2 days

### Dependencies
Instruction fine-tuning" \
  --milestone "$MILESTONE_4"

echo "✅ Sprint 4 issues created!"

# Create Sprint 5 Issues
echo ""
echo "🎯 Creating Sprint 5 Issues..."

gh issue create --title "Implement RAG pipeline with LangChain integration" \
  --body "## Task: RAG Pipeline Development

### Objective
Build complete RAG pipeline integrating retrieval and generation.

### Requirements
- Set up LangChain/LlamaIndex framework
- Implement query encoder using sentence-transformers
- Create vector search component with FAISS
- Design Gujarati prompt templates with Jinja2
- Integrate fine-tuned model with LangChain

### Acceptance Criteria
- [ ] RAG pipeline implemented
- [ ] LangChain integration working
- [ ] Prompt templates created
- [ ] End-to-end pipeline tested
- [ ] Performance benchmarks established

### Estimated Time
3-4 days

### Dependencies
Vector store creation, LoRA fine-tuning" \
  --milestone "$MILESTONE_5"

gh issue create --title "Optimize retrieval quality and prompt engineering" \
  --body "## Task: RAG Optimization

### Objective
Optimize retrieval quality and improve prompt engineering.

### Requirements
- Tune k (number of retrieved docs): test k=3, 5, 10
- Set similarity threshold for relevance filtering
- Experiment with prompt structures:
  - Chain-of-thought prompting
  - Few-shot examples
  - Explicit instructions
- Test on diverse query types

### Acceptance Criteria
- [ ] Retrieval parameters optimized
- [ ] Prompt engineering improved
- [ ] Quality metrics established
- [ ] Performance validated
- [ ] Best practices documented

### Estimated Time
2-3 days

### Dependencies
RAG pipeline implementation" \
  --milestone "$MILESTONE_5"

gh issue create --title "Evaluate RAG system performance" \
  --body "## Task: RAG Evaluation

### Objective
Comprehensive evaluation of RAG system performance.

### Requirements
- Retrieval accuracy: Precision@k, Recall@k
- Generation quality: faithfulness, relevance, groundedness
- Human evaluation on 100 queries
- End-to-end latency measurement
- Compare with baseline (no RAG)

### Acceptance Criteria
- [ ] Retrieval metrics computed
- [ ] Generation quality assessed
- [ ] Human evaluation completed
- [ ] Latency benchmarks established
- [ ] Evaluation report created

### Estimated Time
2 days

### Dependencies
RAG optimization" \
  --milestone "$MILESTONE_5"

echo "✅ Sprint 5 issues created!"

# Create Sprint 6 Issues
echo ""
echo "🎯 Creating Sprint 6 Issues..."

gh issue create --title "Quantize fine-tuned SLM to 4-bit using bitsandbytes" \
  --body "## Task: Model Quantization

### Objective
Quantize fine-tuned SLM to 4-bit for mobile deployment.

### Requirements
- Apply 4-bit quantization using bitsandbytes
- Merge LoRA adapters before quantization
- Alternative: GPTQ quantization
- Quality assessment post-quantization
- Target: <500MB model size

### Acceptance Criteria
- [ ] 4-bit quantization implemented
- [ ] Quality degradation <5%
- [ ] Model size <500MB
- [ ] Performance benchmarks completed
- [ ] Quantized model saved

### Estimated Time
2-3 days

### Dependencies
LoRA fine-tuning" \
  --milestone "$MILESTONE_6"

gh issue create --title "Convert quantized model to mobile deployment formats" \
  --body "## Task: Mobile Format Conversion

### Objective
Convert quantized model to mobile-optimized formats.

### Requirements
- Convert to ONNX format
- Alternative: TensorFlow Lite conversion
- Recommended: llama.cpp GGUF format
- Test CPU inference performance
- Target: >10 TPS, <2s TTFT

### Acceptance Criteria
- [ ] Mobile formats created
- [ ] CPU inference tested
- [ ] Performance targets met
- [ ] Deployment bundle created
- [ ] Documentation updated

### Estimated Time
2-3 days

### Dependencies
Model quantization" \
  --milestone "$MILESTONE_6"

gh issue create --title "Package deployment bundle with model and vector store" \
  --body "## Task: Deployment Package

### Objective
Create complete deployment package for mobile integration.

### Requirements
- Package quantized model (<500MB)
- Include vector store and embeddings
- Add configuration files
- Create inference scripts
- Target: <1GB total size

### Acceptance Criteria
- [ ] Deployment package created
- [ ] Size requirements met
- [ ] All components included
- [ ] Integration guide written
- [ ] Package tested

### Estimated Time
1-2 days

### Dependencies
Mobile format conversion" \
  --milestone "$MILESTONE_6"

echo "✅ Sprint 6 issues created!"

# Create Sprint 7 Issues
echo ""
echo "🎯 Creating Sprint 7 Issues..."

gh issue create --title "Download and prepare PlantVillage crop disease dataset" \
  --body "## Task: Image Dataset Preparation

### Objective
Prepare crop disease image dataset for CNN training.

### Requirements
- Download PlantVillage dataset
- Filter Gujarat-relevant crops: cotton, groundnut, wheat, rice, tomato
- Create train/val/test splits (70/15/15)
- Implement data augmentation pipeline
- Create Gujarati disease label mappings

### Acceptance Criteria
- [ ] Dataset downloaded and organized
- [ ] Relevant crops filtered
- [ ] Data splits created
- [ ] Augmentation pipeline implemented
- [ ] Gujarati labels mapped

### Estimated Time
2-3 days

### Dependencies
None" \
  --milestone "$MILESTONE_7"

gh issue create --title "Train MobileNetV2 CNN for disease classification" \
  --body "## Task: CNN Training

### Objective
Train lightweight CNN for crop disease detection.

### Requirements
- Transfer learning with MobileNetV2
- Fine-tune on crop disease dataset
- Optimizer: Adam, LR: 1e-3, batch size: 32
- Epochs: 20-30, use class weights for imbalance
- Target: >85% test accuracy

### Acceptance Criteria
- [ ] CNN training completed
- [ ] Test accuracy >85%
- [ ] Model optimized for mobile
- [ ] TFLite conversion successful
- [ ] Performance documented

### Estimated Time
3-4 days

### Dependencies
Image dataset preparation" \
  --milestone "$MILESTONE_7"

gh issue create --title "Evaluate CNN performance and create confusion matrix" \
  --body "## Task: CNN Evaluation

### Objective
Comprehensive evaluation of CNN model performance.

### Requirements
- Test set accuracy: target >85%
- Per-class F1-scores
- Confusion matrix analysis
- Identify misclassified pairs
- Model size verification (<10MB)

### Acceptance Criteria
- [ ] Evaluation metrics computed
- [ ] Confusion matrix created
- [ ] Per-class performance analyzed
- [ ] Model size verified
- [ ] Evaluation report created

### Estimated Time
1-2 days

### Dependencies
CNN training" \
  --milestone "$MILESTONE_7"

echo "✅ Sprint 7 issues created!"

# Create Sprint 8 Issues
echo ""
echo "🎯 Creating Sprint 8 Issues..."

gh issue create --title "Integrate CNN and RAG+SLM into unified system" \
  --body "## Task: Multi-modal Integration

### Objective
Create unified system handling both text and image inputs.

### Requirements
- Design multi-modal interface (text, image, hybrid)
- Implement routing logic for different input types
- Create image → text → RAG pipeline
- Format responses in Gujarati
- Test end-to-end scenarios

### Acceptance Criteria
- [ ] Multi-modal interface implemented
- [ ] Routing logic working
- [ ] End-to-end pipeline tested
- [ ] Response formatting completed
- [ ] Integration validated

### Estimated Time
3-4 days

### Dependencies
RAG pipeline, CNN training" \
  --milestone "$MILESTONE_8"

gh issue create --title "Comprehensive system testing and performance optimization" \
  --body "## Task: System Testing

### Objective
Comprehensive testing of complete AI backend system.

### Requirements
- Latency benchmarks: text <3s, image <4s
- Memory usage: <2GB peak RAM
- Accuracy testing: SLM >80%, RAG >70%, CNN >85%
- Edge case testing and error handling
- Performance optimization

### Acceptance Criteria
- [ ] All performance targets met
- [ ] Edge cases handled
- [ ] Error handling implemented
- [ ] Optimization completed
- [ ] Test report created

### Estimated Time
2-3 days

### Dependencies
Multi-modal integration" \
  --milestone "$MILESTONE_8"

gh issue create --title "Create comprehensive documentation and deployment guide" \
  --body "## Task: Documentation & Deployment

### Objective
Create complete documentation and deployment package.

### Requirements
- Technical documentation (architecture, training procedures)
- API documentation and specifications
- Deployment guide for mobile integration
- Create final deployment package
- Quality assurance checklist

### Acceptance Criteria
- [ ] Technical docs completed
- [ ] API docs created
- [ ] Deployment guide written
- [ ] Final package created
- [ ] QA checklist completed

### Estimated Time
2-3 days

### Dependencies
System testing" \
  --milestone "$MILESTONE_8"

echo "✅ Sprint 8 issues created!"

echo ""
echo "🎉 All issues created successfully!"
echo "📊 Repository: https://github.com/ns-1456/kisan-saathi"
echo "📋 Issues: https://github.com/ns-1456/kisan-saathi/issues"
echo "🎯 Milestones: https://github.com/ns-1456/kisan-saathi/milestones"
echo ""
echo "🚀 Ready to start Sprint 1 implementation!"

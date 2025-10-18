#!/usr/bin/env python3
"""
GitHub Issues Generator for Kisan Saathi Project
Creates issues for each sprint's tasks based on the roadmap
"""

import json
from datetime import datetime, timedelta

# Sprint definitions with tasks
sprints = {
    "Sprint 1": {
        "title": "🚀 Sprint 1: Infrastructure Setup & Data Collection (Weeks 1-2)",
        "description": "Set up AWS EC2 GPU instance and collect Gujarati agricultural data",
        "tasks": [
            {
                "title": "Set up AWS EC2 GPU instance with CUDA support",
                "body": """## Task: AWS EC2 Setup

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
None

### Labels
`infrastructure`, `aws`, `gpu`, `sprint-1`""",
                "labels": ["infrastructure", "aws", "gpu", "sprint-1"]
            },
            {
                "title": "Configure Python environment with ML dependencies",
                "body": """## Task: Python Environment Setup

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
AWS EC2 instance setup

### Labels
`infrastructure`, `python`, `dependencies`, `sprint-1`""",
                "labels": ["infrastructure", "python", "dependencies", "sprint-1"]
            },
            {
                "title": "Download and process AI4Bharat IndicCorp Gujarati dataset",
                "body": """## Task: IndicCorp Dataset Processing

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
Python environment setup

### Labels
`data`, `gujarati`, `indiccorp`, `sprint-1`""",
                "labels": ["data", "gujarati", "indiccorp", "sprint-1"]
            },
            {
                "title": "Scrape Gujarati agricultural resources from government websites",
                "body": """## Task: Government Resource Collection

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
Python environment setup

### Labels
`data`, `scraping`, `agriculture`, `gujarati`, `sprint-1`""",
                "labels": ["data", "scraping", "agriculture", "gujarati", "sprint-1"]
            },
            {
                "title": "Create initial Q&A dataset for instruction tuning",
                "body": """## Task: Q&A Dataset Creation

### Objective
Manually curate Gujarati agricultural Q&A pairs for instruction tuning.

### Requirements
- 500-1000 Q&A pairs covering:
  - Soil preparation
  - Irrigation schedules
  - Pest/disease management
  - Fertilizer recommendations
  - Weather-based advisories
- Format: JSON with fields `{question_gu, answer_gu, category, crop_type}`
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
Government resource collection

### Labels
`data`, `qa`, `instruction-tuning`, `gujarati`, `sprint-1`""",
                "labels": ["data", "qa", "instruction-tuning", "gujarati", "sprint-1"]
            }
        ]
    },
    "Sprint 2": {
        "title": "📊 Sprint 2: Data Processing & Knowledge Base Creation (Weeks 3-4)",
        "description": "Clean data, create structured knowledge base, and build vector embeddings",
        "tasks": [
            {
                "title": "Clean and normalize Gujarati text data",
                "body": """## Task: Text Normalization Pipeline

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
IndicCorp dataset processing

### Labels
`data`, `preprocessing`, `gujarati`, `sprint-2`""",
                "labels": ["data", "preprocessing", "gujarati", "sprint-2"]
            },
            {
                "title": "Structure agricultural documents into JSON format",
                "body": """## Task: Document Structuring

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
Government resource collection

### Labels
`data`, `structuring`, `json`, `agriculture`, `sprint-2`""",
                "labels": ["data", "structuring", "json", "agriculture", "sprint-2"]
            },
            {
                "title": "Build FAISS vector store with agricultural knowledge embeddings",
                "body": """## Task: Vector Store Creation

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
Document structuring

### Labels
`rag`, `embeddings`, `faiss`, `vector-store`, `sprint-2`""",
                "labels": ["rag", "embeddings", "faiss", "vector-store", "sprint-2"]
            },
            {
                "title": "Create train/val/test splits for fine-tuning datasets",
                "body": """## Task: Dataset Splits

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
Q&A dataset creation

### Labels
`data`, `splits`, `training`, `sprint-2`""",
                "labels": ["data", "splits", "training", "sprint-2"]
            }
        ]
    },
    "Sprint 3": {
        "title": "🧠 Sprint 3: Base Model Selection & Baseline (Weeks 5-6)",
        "description": "Evaluate and select base SLM, establish baseline performance",
        "tasks": [
            {
                "title": "Download and evaluate candidate SLMs (Gemma-2B, Llama-3.2-1B, Bloomz-1B7)",
                "body": """## Task: SLM Evaluation

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
Dataset splits creation

### Labels
`models`, `evaluation`, `gujarati`, `sprint-3`""",
                "labels": ["models", "evaluation", "gujarati", "sprint-3"]
            },
            {
                "title": "Prepare training datasets for LoRA fine-tuning",
                "body": """## Task: Training Data Preparation

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
SLM evaluation

### Labels
`data`, `training`, `lora`, `sprint-3`""",
                "labels": ["data", "training", "lora", "sprint-3"]
            },
            {
                "title": "Configure LoRA hyperparameters and training environment",
                "body": """## Task: LoRA Configuration

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
Training data preparation

### Labels
`training`, `lora`, `configuration`, `sprint-3`""",
                "labels": ["training", "lora", "configuration", "sprint-3"]
            }
        ]
    },
    "Sprint 4": {
        "title": "🎯 Sprint 4: LoRA Fine-tuning (Weeks 7-8)",
        "description": "Fine-tune selected SLM using LoRA on Gujarati agricultural corpus",
        "tasks": [
            {
                "title": "Fine-tune SLM with LoRA on domain adaptation corpus",
                "body": """## Task: Domain Adaptation Training

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
LoRA configuration

### Labels
`training`, `lora`, `domain-adaptation`, `sprint-4`""",
                "labels": ["training", "lora", "domain-adaptation", "sprint-4"]
            },
            {
                "title": "Instruction fine-tuning on Q&A dataset",
                "body": """## Task: Instruction Tuning

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
Domain adaptation training

### Labels
`training`, `instruction-tuning`, `qa`, `sprint-4`""",
                "labels": ["training", "instruction-tuning", "qa", "sprint-4"]
            },
            {
                "title": "Evaluate fine-tuned model vs base model",
                "body": """## Task: Model Evaluation

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
Instruction fine-tuning

### Labels
`evaluation`, `metrics`, `human-eval`, `sprint-4`""",
                "labels": ["evaluation", "metrics", "human-eval", "sprint-4"]
            }
        ]
    },
    "Sprint 5": {
        "title": "🔍 Sprint 5: RAG Pipeline Implementation (Weeks 9-10)",
        "description": "Build retrieval-augmented generation system with LangChain",
        "tasks": [
            {
                "title": "Implement RAG pipeline with LangChain integration",
                "body": """## Task: RAG Pipeline Development

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
Vector store creation, LoRA fine-tuning

### Labels
`rag`, `langchain`, `pipeline`, `sprint-5`""",
                "labels": ["rag", "langchain", "pipeline", "sprint-5"]
            },
            {
                "title": "Optimize retrieval quality and prompt engineering",
                "body": """## Task: RAG Optimization

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
RAG pipeline implementation

### Labels
`rag`, `optimization`, `prompting`, `sprint-5`""",
                "labels": ["rag", "optimization", "prompting", "sprint-5"]
            },
            {
                "title": "Evaluate RAG system performance",
                "body": """## Task: RAG Evaluation

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
RAG optimization

### Labels
`evaluation`, `rag`, `metrics`, `sprint-5`""",
                "labels": ["evaluation", "rag", "metrics", "sprint-5"]
            }
        ]
    },
    "Sprint 6": {
        "title": "⚡ Sprint 6: Model Quantization & Mobile Optimization (Weeks 11-12)",
        "description": "Quantize models and optimize for mobile deployment",
        "tasks": [
            {
                "title": "Quantize fine-tuned SLM to 4-bit using bitsandbytes",
                "body": """## Task: Model Quantization

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
LoRA fine-tuning

### Labels
`quantization`, `mobile`, `optimization`, `sprint-6`""",
                "labels": ["quantization", "mobile", "optimization", "sprint-6"]
            },
            {
                "title": "Convert quantized model to mobile deployment formats",
                "body": """## Task: Mobile Format Conversion

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
Model quantization

### Labels
`mobile`, `conversion`, `onnx`, `tflite`, `sprint-6`""",
                "labels": ["mobile", "conversion", "onnx", "tflite", "sprint-6"]
            },
            {
                "title": "Package deployment bundle with model and vector store",
                "body": """## Task: Deployment Package

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
Mobile format conversion

### Labels
`deployment`, `packaging`, `mobile`, `sprint-6`""",
                "labels": ["deployment", "packaging", "mobile", "sprint-6"]
            }
        ]
    },
    "Sprint 7": {
        "title": "🌾 Sprint 7: CNN for Crop Disease Detection (Weeks 13-14)",
        "description": "Train lightweight CNN for crop disease classification",
        "tasks": [
            {
                "title": "Download and prepare PlantVillage crop disease dataset",
                "body": """## Task: Image Dataset Preparation

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
None

### Labels
`data`, `images`, `cnn`, `plantvillage`, `sprint-7`""",
                "labels": ["data", "images", "cnn", "plantvillage", "sprint-7"]
            },
            {
                "title": "Train MobileNetV2 CNN for disease classification",
                "body": """## Task: CNN Training

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
Image dataset preparation

### Labels
`cnn`, `training`, `mobilenet`, `disease-detection`, `sprint-7`""",
                "labels": ["cnn", "training", "mobilenet", "disease-detection", "sprint-7"]
            },
            {
                "title": "Evaluate CNN performance and create confusion matrix",
                "body": """## Task: CNN Evaluation

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
CNN training

### Labels
`evaluation`, `cnn`, `metrics`, `confusion-matrix`, `sprint-7`""",
                "labels": ["evaluation", "cnn", "metrics", "confusion-matrix", "sprint-7"]
            }
        ]
    },
    "Sprint 8": {
        "title": "🔗 Sprint 8: Multi-modal Integration & Final Testing (Weeks 15-16)",
        "description": "Integrate all components and comprehensive testing",
        "tasks": [
            {
                "title": "Integrate CNN and RAG+SLM into unified system",
                "body": """## Task: Multi-modal Integration

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
RAG pipeline, CNN training

### Labels
`integration`, `multimodal`, `unified-system`, `sprint-8`""",
                "labels": ["integration", "multimodal", "unified-system", "sprint-8"]
            },
            {
                "title": "Comprehensive system testing and performance optimization",
                "body": """## Task: System Testing

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
Multi-modal integration

### Labels
`testing`, `performance`, `optimization`, `sprint-8`""",
                "labels": ["testing", "performance", "optimization", "sprint-8"]
            },
            {
                "title": "Create comprehensive documentation and deployment guide",
                "body": """## Task: Documentation & Deployment

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
System testing

### Labels
`documentation`, `deployment`, `api`, `sprint-8`""",
                "labels": ["documentation", "deployment", "api", "sprint-8"]
            }
        ]
    }
}

def create_github_issues():
    """Create GitHub issues for all sprint tasks"""
    
    print("GitHub Issues for Kisan Saathi Project")
    print("=" * 50)
    
    for sprint_name, sprint_data in sprints.items():
        print(f"\n## {sprint_data['title']}")
        print(f"**Description:** {sprint_data['description']}")
        print(f"**Tasks:** {len(sprint_data['tasks'])}")
        
        for i, task in enumerate(sprint_data['tasks'], 1):
            print(f"\n### Issue #{i}: {task['title']}")
            print(f"**Labels:** {', '.join(task['labels'])}")
            print(f"**Body:**")
            print(task['body'])
            print("\n" + "-" * 40)

if __name__ == "__main__":
    create_github_issues()

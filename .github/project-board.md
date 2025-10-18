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

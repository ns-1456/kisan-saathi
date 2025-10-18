# Kisan Saathi - Gujarati Offline Agricultural AI Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## Overview

Kisan Saathi is an offline, Gujarati-language agricultural AI assistant designed for low-resource Android devices. The system combines:

- **Fine-tuned Small Language Model (SLM)** for Gujarati agricultural queries
- **Retrieval-Augmented Generation (RAG)** with local knowledge base
- **CNN-based crop disease detection** from images
- **Offline-first architecture** optimized for mobile deployment

## Project Structure

```
kisan-saathi/
├── 📁 data/
│   ├── raw/              # Raw scraped data
│   ├── processed/        # Cleaned datasets
│   └── knowledge_base/   # Structured documents
├── 📁 models/
│   ├── base/             # Downloaded base models
│   ├── finetuned/        # LoRA adapters
│   └── quantized/        # Deployment models
├── 📁 notebooks/         # Jupyter notebooks for experiments
├── 📁 scripts/
│   ├── data_collection/
│   ├── preprocessing/
│   ├── training/
│   └── inference/
├── 📁 vector_store/      # FAISS indexes
├── 📁 configs/           # Training configs
├── 📁 evaluation/         # Test results and metrics
├── 📁 deployment/        # Final deployment artifacts
├── 📁 docs/              # Documentation
└── 📁 tests/             # Unit tests
```

## Development Roadmap

### 🚀 Sprint 1-2 (Weeks 1-4): Foundation & Data Pipeline
- [ ] AWS EC2 GPU instance setup
- [ ] Python environment configuration
- [ ] Gujarati agricultural corpus collection
- [ ] Knowledge base creation and indexing

### 🧠 Sprint 3-4 (Weeks 5-8): SLM Fine-tuning
- [ ] Base model selection and evaluation
- [ ] LoRA fine-tuning on Gujarati corpus
- [ ] Instruction tuning on Q&A dataset
- [ ] Model evaluation and optimization

### 🔍 Sprint 5-6 (Weeks 9-12): RAG Pipeline & Quantization
- [ ] RAG implementation with LangChain
- [ ] Vector store optimization
- [ ] Model quantization (4-bit)
- [ ] Mobile deployment format conversion

### 🌾 Sprint 7-8 (Weeks 13-16): CNN & Integration
- [ ] CNN training for disease detection
- [ ] Multi-modal system integration
- [ ] Comprehensive testing
- [ ] Documentation and deployment prep

## Key Features

- **Gujarati Language Support**: Native Gujarati agricultural terminology
- **Offline Operation**: No internet required after initial setup
- **Low Resource**: Optimized for devices with ≤4GB RAM
- **Multi-modal**: Text queries + image disease detection
- **Domain-Specific**: Focused on Gujarat's major crops (cotton, groundnut, wheat)

## Technology Stack

- **ML Framework**: PyTorch, Transformers
- **Fine-tuning**: PEFT (LoRA)
- **Quantization**: bitsandbytes, ONNX
- **RAG**: LangChain, FAISS
- **CNN**: MobileNetV2, TensorFlow Lite
- **Deployment**: llama.cpp, Android NDK

## Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (for training)
- AWS account with GPU access
- 256GB+ storage for datasets

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/kisan-saathi.git
cd kisan-saathi

# Create virtual environment
python -m venv kisan-env
source kisan-env/bin/activate  # On Windows: kisan-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup
```bash
# Download datasets (requires API keys)
python scripts/data_collection/download_indiccorp.py
python scripts/data_collection/scrape_govt_resources.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- AI4Bharat for IndicCorp and IndicBERT models
- Hugging Face for Transformers library
- PlantVillage for disease dataset
- Gujarat State Agriculture Department for domain resources

## Contact

- Project Lead: Patel Nand, Patel Aesha
- Email: nandpatel1456@gmail.com, [your.email@example.com]

---

**Status**: 🚧 In Development - Sprint 1
**Timeline**: 16 weeks (4 months)
**Target**: Offline Gujarati agricultural AI for Android

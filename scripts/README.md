# Scripts Directory

This directory contains all executable scripts for the Kisan Saathi project.

## Structure

- `data_collection/` - Scripts for downloading and scraping data
- `preprocessing/` - Data cleaning and preparation scripts
- `training/` - Model training and fine-tuning scripts
- `inference/` - Model inference and deployment scripts

## Usage

All scripts should be run from the project root directory:

```bash
python scripts/data_collection/download_indiccorp.py
python scripts/training/finetune_domain_adaptation.py
python scripts/inference/rag_pipeline.py
```

## Script Categories

### Data Collection
- Download AI4Bharat IndicCorp dataset
- Scrape government agricultural resources
- Collect KVK manuals and extension materials
- Download PlantVillage disease dataset

### Preprocessing
- Clean and normalize Gujarati text
- Create structured knowledge documents
- Generate embeddings for vector store
- Prepare training datasets

### Training
- Fine-tune base SLM with LoRA
- Train CNN for disease detection
- Optimize model hyperparameters
- Evaluate model performance

### Inference
- RAG pipeline implementation
- Model quantization
- Mobile deployment conversion
- Multi-modal system integration

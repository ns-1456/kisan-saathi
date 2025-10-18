# Models Directory

This directory contains all model-related files for the Kisan Saathi project.

## Structure

- `base/` - Downloaded pre-trained base models
- `finetuned/` - LoRA adapters and fine-tuned models
- `quantized/` - Quantized models for mobile deployment

## Model Types

### Base Models
- Gemma-2B-it (Google)
- Llama-3.2-1B-Instruct (Meta)
- Bloomz-1B7 (BigScience)

### Fine-tuned Models
- Domain-adapted SLM for Gujarati agricultural text
- Instruction-tuned model for Q&A responses
- LoRA adapters for efficient fine-tuning

### Quantized Models
- 4-bit quantized SLM (<500MB)
- TFLite CNN for disease detection (<10MB)
- ONNX models for mobile deployment

## Usage

Models are typically large files (>100MB) and are not tracked in git.
Use model download and training scripts to obtain models.

## Model Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("models/base/gemma-2b-it")
tokenizer = AutoTokenizer.from_pretrained("models/base/gemma-2b-it")

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "models/finetuned/lora_adapters")
```

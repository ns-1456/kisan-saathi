# Kisan Saathi - Detailed Development Roadmap

## Project Overview

**Objective:** Build an offline, Gujarati-language agricultural AI assistant optimized for low-resource Android devices.

**Timeline:** 16 weeks (4 months) across 8 two-week sprints

**Focus:** AI Backend Development (SLM fine-tuning, RAG pipeline, CNN disease detection)

---

## Sprint 1: Infrastructure Setup & Initial Data Collection (Weeks 1-2)

### Week 1: AWS Environment & Development Setup

#### Tasks

1. **AWS EC2 Setup**
   - Create AWS account and activate free credits
   - Launch g4dn.xlarge Spot Instance (Ubuntu 22.04 LTS, 100GB EBS storage)
   - Configure security groups (SSH access, Jupyter ports)
   - Set up Elastic IP for persistent access
   - Install NVIDIA drivers and CUDA toolkit (11.8 or 12.1)

2. **Python Environment Configuration**
   ```bash
   # Install system dependencies
   sudo apt update && sudo apt install python3.10 python3-pip git

   # Create virtual environment
   python3 -m venv kisan-env
   source kisan-env/bin/activate

   # Install core ML libraries
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers datasets accelerate peft bitsandbytes
   pip install sentence-transformers faiss-gpu langchain
   pip install scikit-learn pandas numpy matplotlib wandb
   ```

3. **Version Control & Project Structure**
   ```
   kisan-saathi/
   ├── data/
   │   ├── raw/              # Raw scraped data
   │   ├── processed/        # Cleaned datasets
   │   └── knowledge_base/   # Structured documents
   ├── models/
   │   ├── base/             # Downloaded base models
   │   ├── finetuned/        # LoRA adapters
   │   └── quantized/        # Deployment models
   ├── notebooks/            # Jupyter notebooks for experiments
   ├── scripts/
   │   ├── data_collection/
   │   ├── preprocessing/
   │   ├── training/
   │   └── inference/
   ├── vector_store/         # FAISS indexes
   └── configs/              # Training configs
   ```

4. **Monitoring & Logging Setup**
   - Create Weights & Biases (wandb) account for experiment tracking
   - Set up local logging infrastructure
   - Configure AWS CloudWatch for cost monitoring

#### Deliverables
- ✅ Fully configured AWS EC2 instance with GPU support
- ✅ Python environment with all dependencies installed
- ✅ Git repository with project structure
- ✅ Cost monitoring dashboard

---

### Week 2: Gujarati Data Collection

#### Tasks

1. **AI4Bharat IndicCorp Dataset**
   - Download Gujarati subset from AI4Bharat IndicCorp v2
   - Extract and decompress (expected size: ~2-5GB)
   - Perform initial quality checks (encoding, language detection)
   - Sample 100K sentences for quick experiments

2. **Government Agricultural Resources**
   - **Gujarat State Agriculture Department:**
     - Scrape/download PDF reports (crop advisories, seasonal guides)
     - Extract text using `pdfplumber` or `pypdf2`
   - **Krishi Vigyan Kendras (KVK) Gujarat:**
     - Collect training manuals, FAQs, best practices documents
     - Focus areas: cotton, groundnut, wheat, bajra (major Gujarat crops)
   - **ICAR Gujarat Research Centers:**
     - Download research bulletins and extension materials

3. **Create Initial Q&A Dataset**
   - Manual curation: 500-1000 Q&A pairs covering:
     - Soil preparation
     - Irrigation schedules
     - Pest/disease management
     - Fertilizer recommendations
     - Weather-based advisories
   - Format: JSON with fields `{question_gu, answer_gu, category, crop_type}`

4. **Parallel Corpus for Translation**
   - Download BPCC (Gujarati-English subset)
   - Download Samanantar (Gujarati-English pairs)
   - Total target: 100K+ parallel sentences

#### Scripts to Create

**`scripts/data_collection/scrape_govt_resources.py`**
```python
# Web scraping script for government websites
# Use requests + BeautifulSoup
# Handle PDF extraction with pdfplumber
```

**`scripts/data_collection/download_indiccorp.py`**
```python
# Download and extract IndicCorp Gujarati
# Validate file integrity
```

#### Deliverables
- ✅ ~2-5GB Gujarati text corpus (IndicCorp + domain-specific)
- ✅ 500-1000 agricultural Q&A pairs in Gujarati
- ✅ 100K+ Gujarati-English parallel sentences
- ✅ Data collection scripts and documentation

---

## Sprint 2: Data Processing & Knowledge Base Creation (Weeks 3-4)

### Week 3: Data Cleaning & Preprocessing

#### Tasks

1. **Text Normalization Pipeline**
   - Remove HTML tags, special characters, excessive whitespace
   - Standardize Unicode encoding (normalize to NFC)
   - Language detection: filter out non-Gujarati content (use `langdetect`)
   - Deduplication: remove exact duplicates and near-duplicates (MinHash)

2. **Agricultural Document Structuring**
   - Parse government reports into structured sections:
     - Crop name/type
     - Growth stage information
     - Pest/disease descriptions
     - Treatment recommendations
     - Seasonal timing
   - Convert to JSON/JSONL format with metadata

3. **Q&A Dataset Augmentation**
   - Use back-translation (Gujarati → English → Gujarati) for data augmentation
   - Paraphrase questions using rule-based templates
   - Target: expand 500 → 2000 Q&A pairs

4. **Training Split Creation**
   - Split datasets:
     - **Domain adaptation corpus:** 80% train, 10% val, 10% test
     - **Q&A instruction dataset:** 70% train, 15% val, 15% test
   - Stratify by category to ensure balanced coverage

#### Scripts to Create

**`scripts/preprocessing/clean_gujarati_text.py`**
```python
import re
import unicodedata
from langdetect import detect

def normalize_gujarati_text(text):
    # NFC normalization
    text = unicodedata.normalize('NFC', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
```

**`scripts/preprocessing/structure_documents.py`**
```python
# Parse PDFs and create structured JSON documents
# Extract entities: crop names, chemicals, diseases
```

#### Deliverables
- ✅ Cleaned Gujarati corpus (deduplicated, normalized)
- ✅ ~5000 structured agricultural knowledge documents (JSON)
- ✅ 2000+ Q&A pairs for instruction tuning
- ✅ Train/val/test splits ready for fine-tuning

---

### Week 4: Vector Store & Knowledge Base Indexing

#### Tasks

1. **Embedding Model Selection**
   - Evaluate multilingual embedding models:
     - `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
     - `sentence-transformers/LaBSE`
     - AI4Bharat `IndicBERT` embeddings
   - Benchmark: semantic similarity on Gujarati agricultural queries
   - Select model based on: quality, embedding dimension (prefer 384-768), inference speed

2. **Generate Embeddings for Knowledge Base**
   - Process all 5000+ agricultural documents
   - Generate embeddings in batches (batch_size=32)
   - Store embeddings + metadata (document ID, crop type, category)

3. **Build FAISS Vector Index**
   - Create FAISS index (IndexFlatIP for exact search, or IndexIVFFlat for speed)
   - Optimize for offline deployment (store index to disk)
   - Test retrieval: query → top-k documents

4. **RAG Retrieval Evaluation Dataset**
   - Create 200 test queries with ground-truth relevant documents
   - Compute baseline retrieval metrics:
     - Precision@5, Recall@10, MRR (Mean Reciprocal Rank)
   - Establish quality threshold (target: Precision@5 > 0.6)

#### Scripts to Create

**`scripts/preprocessing/build_vector_store.py`**
```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Generate embeddings
docs = load_documents()
embeddings = model.encode(docs, batch_size=32, show_progress_bar=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
index.add(embeddings)

# Save to disk
faiss.write_index(index, 'vector_store/agri_knowledge.index')
```

**`scripts/evaluation/test_retrieval.py`**
```python
# Load test queries and ground truth
# Compute retrieval metrics
```

#### Deliverables
- ✅ FAISS vector store with 5000+ document embeddings
- ✅ Embedding model selected and benchmarked
- ✅ Retrieval evaluation dataset (200 queries)
- ✅ Baseline retrieval metrics report

---

## Sprint 3: Base Model Selection & Baseline (Weeks 5-6)

### Week 5: SLM Evaluation & Selection

#### Tasks

1. **Download Candidate Models**
   - **Gemma-2B-it** (Google)
   - **Llama-3.2-1B-Instruct** (Meta)
   - **Bloomz-1B7** (BigScience - multilingual)
   - AI4Bharat **IndicBERT-MLM-TLM** (if generative variant available)

2. **Gujarati Capability Testing**
   - Prepare 50 test prompts in Gujarati (agricultural domain)
   - Run zero-shot inference on each model
   - Human evaluation: fluency (1-5), correctness (1-5), relevance (1-5)
   - Automated metrics: perplexity on Gujarati held-out set

3. **Technical Benchmarks**
   - **Model size:** target <2GB in FP16
   - **CPU inference latency:** measure on local CPU (8-core)
   - **Memory footprint:** peak RAM during inference
   - **Licensing:** check for commercial use restrictions

4. **Select Final Base Model**
   - Create comparison matrix (quality vs. size vs. speed)
   - Decision criteria:
     1. Gujarati generation quality (weight: 40%)
     2. Model size (weight: 30%)
     3. Inference speed (weight: 20%)
     4. License permissiveness (weight: 10%)

#### Scripts to Create

**`scripts/evaluation/benchmark_base_models.py`**
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

models = [
    "google/gemma-2b-it",
    "meta-llama/Llama-3.2-1B-Instruct",
    "bigscience/bloomz-1b7"
]

for model_name in models:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    
    # Benchmark inference time
    # Measure memory
    # Generate responses to test prompts
```

#### Deliverables
- ✅ All candidate models downloaded and tested
- ✅ Evaluation report with quantitative metrics
- ✅ Final base model selected (e.g., Gemma-2B-it)
- ✅ Baseline performance established

---

### Week 6: Environment Prep for Fine-tuning

#### Tasks

1. **Configure Training Environment**
   - Install additional dependencies: `accelerate`, `deepspeed` (optional)
   - Set up mixed-precision training (FP16/BF16)
   - Configure gradient checkpointing for memory efficiency

2. **Prepare Training Datasets**
   - **Domain Adaptation Dataset:**
     - Format: plain text, continuous documents
     - Tokenize with selected model's tokenizer
     - Create chunks of 512-1024 tokens
   - **Instruction Tuning Dataset:**
     - Format: `{instruction, input, output}` in Alpaca/ShareGPT style
     - Example:
       ```json
       {
         "instruction": "કપાસમાં ગુલાબી ઈયળનું નિયંત્રણ કેવી રીતે કરવું?",
         "input": "",
         "output": "ગુલાબી ઈયળના નિયંત્રણ માટે..."
       }
       ```

3. **LoRA Configuration Planning**
   - Research LoRA hyperparameters for selected base model
   - Plan target modules: typically `q_proj`, `v_proj`, `k_proj`, `o_proj`
   - LoRA rank (r): 8-16 (start with 8)
   - LoRA alpha: 16-32 (typically 2*rank)
   - Dropout: 0.05-0.1

4. **Training Budget Planning**
   - Estimate GPU hours needed:
     - Domain adaptation: ~20-30 hours
     - Instruction tuning: ~15-25 hours
   - Plan AWS Spot Instance schedule (train during low-cost hours)
   - Set up checkpointing every 500 steps

#### Scripts to Create

**`scripts/training/prepare_training_data.py`**
```python
from datasets import load_dataset, Dataset

# Load Q&A pairs
qa_data = load_json('data/processed/qa_dataset.json')

# Format for instruction tuning
formatted_data = []
for item in qa_data:
    formatted_data.append({
        'instruction': item['question_gu'],
        'input': '',
        'output': item['answer_gu']
    })

# Convert to HuggingFace Dataset
dataset = Dataset.from_list(formatted_data)
dataset.save_to_disk('data/processed/instruction_dataset')
```

**`configs/lora_config.yaml`**
```yaml
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
bias: "none"
task_type: "CAUSAL_LM"
```

#### Deliverables
- ✅ Training datasets formatted and tokenized
- ✅ LoRA configuration file created
- ✅ Training environment validated
- ✅ GPU budget allocated on AWS

---

## Sprint 4: LoRA Fine-tuning (Weeks 7-8)

### Week 7: Domain Adaptation Training

#### Tasks

1. **Phase 1: Continued Pre-training on Gujarati Corpus**
   - Objective: Adapt base model to Gujarati agricultural language
   - Dataset: Cleaned IndicCorp + domain documents
   - Training setup:
     ```python
     from peft import LoraConfig, get_peft_model, TaskType
     
     lora_config = LoraConfig(
         r=8,
         lora_alpha=16,
         target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
         lora_dropout=0.05,
         bias="none",
         task_type=TaskType.CAUSAL_LM
     )
     
     model = get_peft_model(base_model, lora_config)
     ```
   - Training hyperparameters:
     - Learning rate: 2e-4
     - Batch size: 4 (with gradient accumulation=4, effective batch=16)
     - Epochs: 2-3
     - Max sequence length: 1024
     - Optimizer: AdamW with warmup
   - Monitor: training loss, validation perplexity

2. **Training Monitoring**
   - Log to wandb: loss curves, learning rate schedule, GPU utilization
   - Save checkpoints every 500 steps
   - Run validation every 1000 steps

3. **Early Quality Checks**
   - At 50% training, generate sample outputs
   - Check for: fluency, grammatical correctness, domain vocabulary usage
   - Adjust learning rate if loss plateaus

#### Scripts to Create

**`scripts/training/finetune_domain_adaptation.py`**
```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import wandb

# Initialize wandb
wandb.init(project="kisan-saathi", name="domain-adaptation")

# Load base model
model_name = "google/gemma-2b-it"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Apply LoRA
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load dataset
from datasets import load_from_disk
dataset = load_from_disk("data/processed/domain_corpus")

# Training arguments
training_args = TrainingArguments(
    output_dir="models/checkpoints/domain_adaptation",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=1000,
    bf16=True,
    report_to="wandb"
)

# Trainer
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    data_collator=data_collator
)

# Train
trainer.train()

# Save LoRA adapters
model.save_pretrained("models/finetuned/domain_adapted_lora")
```

#### Deliverables
- ✅ Domain-adapted LoRA adapters saved
- ✅ Training logs and loss curves
- ✅ Validation perplexity report

---

### Week 8: Instruction Fine-tuning & Evaluation

#### Tasks

1. **Phase 2: Instruction Tuning on Q&A Dataset**
   - Load domain-adapted model from Week 7
   - Fine-tune on Q&A instruction dataset
   - Training hyperparameters:
     - Learning rate: 1e-4 (lower than domain adaptation)
     - Batch size: 4 (gradient accumulation=4)
     - Epochs: 3-5
     - Max sequence length: 512

2. **Post-training Evaluation**
   - **Automated Metrics:**
     - Perplexity on held-out test set
     - BLEU/ROUGE scores (compare generated answers to ground truth)
   - **Human Evaluation (50 test queries):**
     - Fluency (1-5)
     - Factual correctness (1-5)
     - Relevance to query (1-5)
     - Agricultural domain accuracy (1-5)

3. **Compare Fine-tuned vs. Base Model**
   - Side-by-side evaluation on same 50 queries
   - Compute improvement metrics
   - Identify remaining weaknesses (e.g., specific crop diseases)

4. **Iterative Improvement (if needed)**
   - If quality is unsatisfactory:
     - Adjust LoRA rank (try r=16)
     - Increase training data quality
     - Try different prompt templates

#### Scripts to Create

**`scripts/training/finetune_instruction.py`**
```python
# Similar structure to domain adaptation script
# Load domain-adapted model as starting point
# Use instruction-formatted dataset
```

**`scripts/evaluation/evaluate_model.py`**
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base + LoRA
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")
model = PeftModel.from_pretrained(base_model, "models/finetuned/instruction_tuned_lora")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

# Load test queries
test_queries = load_json("data/processed/test_queries.json")

results = []
for query in test_queries:
    prompt = f"### પ્રશ્ન: {query['question']}\n### જવાબ:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=256)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    results.append({
        'query': query,
        'response': response,
        'ground_truth': query['answer']
    })

# Save results for human evaluation
save_json(results, "evaluation/instruction_tuned_results.json")
```

#### Deliverables
- ✅ Instruction-tuned LoRA adapters
- ✅ Comprehensive evaluation report
- ✅ Comparison: base model vs. fine-tuned model
- ✅ Decision: proceed to RAG integration or iterate

---

## Sprint 5: RAG Pipeline Implementation (Weeks 9-10)

### Week 9: RAG Core Components

#### Tasks

1. **Set Up LangChain/LlamaIndex**
   ```bash
   pip install langchain langchain-community faiss-cpu
   pip install jinja2
   ```

2. **Retrieval Pipeline Implementation**
   - **Component 1: Query Encoder**
     ```python
     from sentence_transformers import SentenceTransformer
     
     query_encoder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
     query_embedding = query_encoder.encode(user_query)
     ```
   
   - **Component 2: Vector Search**
     ```python
     import faiss
     
     index = faiss.read_index('vector_store/agri_knowledge.index')
     D, I = index.search(query_embedding.reshape(1, -1), k=5)  # Top-5 docs
     ```
   
   - **Component 3: Context Formatting**
     - Retrieve full text of top-k documents
     - Format as context for LLM prompt
     - Handle token limit (leave room for query + response)

3. **Prompt Template Design (Gujarati)**
   ```python
   from jinja2 import Template
   
   prompt_template = Template("""
   તમે એક ગુજરાતી કૃષિ સલાહકાર છો. નીચે આપેલ સંદર્ભ માહિતીના આધારે ખેડૂતના પ્રશ્નનો ચોક્કસ જવાબ આપો.
   
   સંદર્ભ માહિતી:
   {% for doc in context_docs %}
   {{ loop.index }}. {{ doc.content }}
   {% endfor %}
   
   પ્રશ્ન: {{ user_question }}
   
   જવાબ:
   """)
   ```

4. **LangChain Integration**
   ```python
   from langchain.llms import HuggingFacePipeline
   from langchain.chains import RetrievalQA
   
   # Wrap fine-tuned model in LangChain
   llm = HuggingFacePipeline(pipeline=generation_pipeline)
   
   # Create RAG chain
   qa_chain = RetrievalQA.from_chain_type(
       llm=llm,
       retriever=vector_store_retriever,
       chain_type="stuff"  # or "map_reduce" for long contexts
   )
   ```

#### Scripts to Create

**`scripts/inference/rag_pipeline.py`**
```python
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import json

class GujaratiAgriRAG:
    def __init__(self, model_path, lora_path, vector_store_path, docs_path):
        # Load fine-tuned model
        self.base_model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model = PeftModel.from_pretrained(self.base_model, lora_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load retrieval components
        self.query_encoder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        self.index = faiss.read_index(vector_store_path)
        self.documents = json.load(open(docs_path))
    
    def retrieve(self, query, k=5):
        query_emb = self.query_encoder.encode([query])
        D, I = self.index.search(query_emb, k)
        retrieved_docs = [self.documents[i] for i in I[0]]
        return retrieved_docs
    
    def generate_response(self, query):
        # Retrieve relevant context
        context_docs = self.retrieve(query, k=5)
        
        # Format prompt
        context_text = "\n\n".join([f"{i+1}. {doc['content']}" for i, doc in enumerate(context_docs)])
        prompt = f"""તમે એક ગુજરાતી કૃષિ સલાહકાર છો. નીચેની માહિતીના આધારે જવાબ આપો.

સંદર્ભ:
{context_text}

પ્રશ્ન: {query}

જવાબ:"""
        
        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=256, temperature=0.7)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (after "જવાબ:")
        answer = response.split("જવાબ:")[-1].strip()
        return answer, context_docs

# Usage
rag = GujaratiAgriRAG(
    model_path="google/gemma-2b-it",
    lora_path="models/finetuned/instruction_tuned_lora",
    vector_store_path="vector_store/agri_knowledge.index",
    docs_path="data/processed/documents.json"
)

response, sources = rag.generate_response("કપાસમાં સફેદ માખીનું નિયંત્રણ કેવી રીતે કરવું?")
print(response)
```

#### Deliverables
- ✅ Functional RAG pipeline (retrieval + generation)
- ✅ Prompt templates for Gujarati agricultural queries
- ✅ Integration script for LangChain/custom RAG

---

### Week 10: RAG Optimization & Evaluation

#### Tasks

1. **Retrieval Quality Optimization**
   - **Tune k (number of retrieved docs):** Test k=3, 5, 10
   - **Relevance filtering:** Set similarity threshold (e.g., cosine similarity > 0.5)
   - **Reranking (optional):** Use cross-encoder for reranking top-k results

2. **Prompt Engineering**
   - Experiment with different prompt structures:
     - Chain-of-thought prompting
     - Few-shot examples in prompt
     - Explicit instructions (e.g., "માત્ર સંદર્ભ માહિતીના આધારે જવાબ આપો")
   - Test on diverse query types: diagnostic, prescriptive, explanatory

3. **RAG Evaluation Metrics**
   - **Retrieval Accuracy:**
     - Precision@k, Recall@k on test set
     - Check if ground-truth document is in top-k
   - **Generation Quality:**
     - Faithfulness: Does answer match retrieved context?
     - Relevance: Does answer address the query?
     - Groundedness: No hallucination beyond context
   - **Human Evaluation (100 queries):**
     - Rate: accuracy, completeness, usefulness (1-5 scale)

4. **End-to-End RAG Testing**
   - Test on realistic farmer queries:
     - "મારા કપાસના છોડ પર સફેદ રંગના ચક્કરો દેખાય છે, શું કરવું?"
     - "ગ્રાઉન્ડનટ માટે યોગ્ય ખાતરનો ડોઝ શું છે?"
     - "વરસાદી ઋતુમાં ઘઉંની વાવણી કેવી રીતે કરવી?"
   - Measure end-to-end latency (retrieval + generation)

#### Scripts to Create

**`scripts/evaluation/evaluate_rag.py`**
```python
from rag_pipeline import GujaratiAgriRAG
import json

# Load test dataset with ground truth
test_data = json.load(open('data/evaluation/rag_test_set.json'))

rag = GujaratiAgriRAG(...)

results = []
for item in test_data:
    query = item['query']
    ground_truth_answer = item['answer']
    ground_truth_doc_ids = item['relevant_doc_ids']
    
    # Run RAG
    response, retrieved_docs = rag.generate_response(query)
    retrieved_doc_ids = [doc['id'] for doc in retrieved_docs]
    
    # Compute retrieval metrics
    precision = len(set(retrieved_doc_ids) & set(ground_truth_doc_ids)) / len(retrieved_doc_ids)
    recall = len(set(retrieved_doc_ids) & set(ground_truth_doc_ids)) / len(ground_truth_doc_ids)
    
    results.append({
        'query': query,
        'response': response,
        'ground_truth': ground_truth_answer,
        'precision': precision,
        'recall': recall,
        'retrieved_docs': retrieved_doc_ids
    })

# Aggregate metrics
avg_precision = sum([r['precision'] for r in results]) / len(results)
avg_recall = sum([r['recall'] for r in results]) / len(results)

print(f"Average Precision@5: {avg_precision:.3f}")
print(f"Average Recall@5: {avg_recall:.3f}")

# Save for human evaluation
json.dump(results, open('evaluation/rag_results.json', 'w'), ensure_ascii=False, indent=2)
```

#### Deliverables
- ✅ Optimized RAG pipeline with tuned hyperparameters
- ✅ RAG evaluation report (retrieval + generation metrics)
- ✅ Human evaluation results on 100 test queries
- ✅ End-to-end latency measurements

---

## Sprint 6: Model Quantization & Mobile Optimization (Weeks 11-12)

### Week 11: Quantization Implementation

#### Tasks

1. **4-bit Quantization with bitsandbytes**
   ```python
   from transformers import AutoModelForCausalLM, BitsAndBytesConfig
   
   quantization_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_compute_dtype=torch.bfloat16,
       bnb_4bit_use_double_quant=True,
       bnb_4bit_quant_type="nf4"
   )
   
   quantized_model = AutoModelForCausalLM.from_pretrained(
       "google/gemma-2b-it",
       quantization_config=quantization_config,
       device_map="auto"
   )
   
   # Merge LoRA adapters before quantization
   from peft import PeftModel
   model = PeftModel.from_pretrained(quantized_model, lora_adapter_path)
   merged_model = model.merge_and_unload()
   ```

2. **Alternative: GPTQ Quantization**
   ```python
   from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
   
   quantize_config = BaseQuantizeConfig(
       bits=4,
       group_size=128,
       desc_act=False
   )
   
   model = AutoGPTQForCausalLM.from_pretrained(
       model_path,
       quantize_config=quantize_config
   )
   
   # Quantize on calibration dataset
   model.quantize(calibration_dataset)
   model.save_quantized("models/quantized/gptq_4bit")
   ```

3. **Quality Assessment Post-Quantization**
   - Run same evaluation suite as Week 8
   - Compare quantized vs. full-precision model:
     - Perplexity increase
     - BLEU/ROUGE scores
     - Human evaluation on sample queries
   - Acceptable threshold: <5% quality degradation

4. **Model Size Verification**
   - Measure: file size of quantized model
   - Target: <500MB for SLM
   - If too large, try: more aggressive quantization, smaller base model

#### Scripts to Create

**`scripts/quantization/quantize_model.py`**
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Load fine-tuned model
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "models/finetuned/instruction_tuned_lora")

# Merge LoRA weights
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("models/merged/gemma_2b_gujarati_agri")

# Quantize to 4-bit
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

quantized_model = AutoModelForCausalLM.from_pretrained(
    "models/merged/gemma_2b_gujarati_agri",
    quantization_config=quantization_config,
    device_map="auto"
)

# Save quantized model
quantized_model.save_pretrained("models/quantized/gemma_2b_4bit")
```

#### Deliverables
- ✅ 4-bit quantized model (<500MB)
- ✅ Quality assessment report (quantized vs. full-precision)
- ✅ Model size and memory footprint measurements

---

### Week 12: Mobile Deployment Format Conversion

#### Tasks

1. **Convert to ONNX (Option 1)**
   ```python
   from optimum.onnxruntime import ORTModelForCausalLM
   
   model = ORTModelForCausalLM.from_pretrained(
       "models/quantized/gemma_2b_4bit",
       export=True,
       provider="CPUExecutionProvider"
   )
   model.save_pretrained("models/deployment/onnx")
   ```

2. **Convert to TensorFlow Lite (Option 2)**
   ```python
   import tensorflow as tf
   
   # First convert to TensorFlow SavedModel
   # Then convert to TFLite with quantization
   converter = tf.lite.TFLiteConverter.from_saved_model("models/tf_savedmodel")
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   converter.target_spec.supported_types = [tf.float16]
   
   tflite_model = converter.convert()
   with open("models/deployment/model.tflite", "wb") as f:
       f.write(tflite_model)
   ```

3. **llama.cpp Conversion (Recommended for Android)**
   ```bash
   # Convert to GGUF format (compatible with llama.cpp)
   python convert-hf-to-gguf.py models/merged/gemma_2b_gujarati_agri \
       --outfile models/deployment/gemma_2b_gujarati_agri_q4.gguf \
       --outtype q4_0
   ```

4. **CPU Inference Benchmarking**
   - Test on local CPU (simulate mobile environment):
     - Intel Core i5/i7 (8 cores)
     - Limit to 4 threads to simulate mobile
   - Measure:
     - **Tokens per second (TPS):** target >10 TPS
     - **Time to first token (TTFT):** target <2s
     - **Memory usage:** peak RAM <2GB
   - Test with different context lengths (128, 256, 512 tokens)

5. **Package for Mobile Deployment**
   - Create deployment bundle:
     ```
     deployment/
     ├── model/
     │   └── gemma_2b_gujarati_agri_q4.gguf
     ├── vector_store/
     │   ├── agri_knowledge.index
     │   └── documents.json
     ├── embeddings/
     │   └── multilingual_mpnet_base_v2_quantized.onnx
     └── config.json
     ```
   - Total size target: <1GB

#### Scripts to Create

**`scripts/quantization/convert_to_mobile.py`**
```python
# Conversion script for target mobile format
# Include benchmarking code
```

**`scripts/inference/test_cpu_inference.py`**
```python
import time
import psutil
from llama_cpp import Llama

# Load quantized model
model = Llama(
    model_path="models/deployment/gemma_2b_gujarati_agri_q4.gguf",
    n_ctx=512,
    n_threads=4,
    n_gpu_layers=0  # CPU only
)

# Benchmark
test_prompts = load_test_prompts()

results = []
for prompt in test_prompts:
    start_time = time.time()
    start_mem = psutil.virtual_memory().used / 1024**3  # GB
    
    output = model(prompt, max_tokens=128)
    
    end_time = time.time()
    end_mem = psutil.virtual_memory().used / 1024**3
    
    latency = end_time - start_time
    tokens_generated = len(output['choices'][0]['text'].split())
    tps = tokens_generated / latency
    mem_used = end_mem - start_mem
    
    results.append({
        'latency': latency,
        'tps': tps,
        'memory_gb': mem_used
    })

# Report averages
print(f"Average Latency: {sum([r['latency'] for r in results]) / len(results):.2f}s")
print(f"Average TPS: {sum([r['tps'] for r in results]) / len(results):.2f}")
print(f"Average Memory: {sum([r['memory_gb'] for r in results]) / len(results):.2f} GB")
```

#### Deliverables
- ✅ Mobile-optimized model format (GGUF/ONNX/TFLite)
- ✅ Deployment bundle with model + vector store (<1GB)
- ✅ CPU inference benchmarks report
- ✅ Mobile deployment documentation

---

## Sprint 7: CNN for Crop Disease Detection (Weeks 13-14)

### Week 13: Image Dataset Preparation

#### Tasks

1. **Download PlantVillage Dataset**
   ```python
   # PlantVillage has 50K+ images, 38 disease classes, 14 crops
   # Focus on Gujarat-relevant crops:
   crops = ['Cotton', 'Groundnut', 'Wheat', 'Rice', 'Tomato']
   ```

2. **Dataset Curation & Organization**
   - Filter relevant crop categories
   - Create folder structure:
     ```
     data/images/
     ├── train/
     │   ├── cotton_bacterial_blight/
     │   ├── cotton_healthy/
     │   ├── groundnut_leaf_spot/
     │   └── ...
     ├── val/
     └── test/
     ```
   - Balance classes (oversample minority classes if needed)

3. **Data Augmentation Pipeline**
   ```python
   from torchvision import transforms
   
   train_transforms = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.RandomHorizontalFlip(p=0.5),
       transforms.RandomVerticalFlip(p=0.3),
       transforms.RandomRotation(degrees=30),
       transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
       transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])
   
   val_test_transforms = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])
   ```

4. **Create Disease Label Mapping (Gujarati)**
   ```json
   {
     "cotton_bacterial_blight": "કપાસ - બેક્ટેરિયલ બ્લાઇટ",
     "cotton_healthy": "કપાસ - સ્વસ્થ",
     "groundnut_leaf_spot": "મગફળી - પાનનો ડાઘ",
     ...
   }
   ```

#### Scripts to Create

**`scripts/data_collection/prepare_image_dataset.py`**
```python
import os
import shutil
from sklearn.model_selection import train_test_split

# Download PlantVillage
# Use kaggle API or manual download

# Filter relevant crops
relevant_classes = [
    'Cotton___bacterial_blight',
    'Cotton___curl_virus',
    'Cotton___healthy',
    'Groundnut___Early_leaf_spot',
    'Groundnut___Late_leaf_spot',
    # ... add more
]

# Split into train/val/test (70/15/15)
for class_name in relevant_classes:
    images = os.listdir(f'data/raw/PlantVillage/{class_name}')
    train, temp = train_test_split(images, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    
    # Copy to organized structure
    for img in train:
        shutil.copy(
            f'data/raw/PlantVillage/{class_name}/{img}',
            f'data/images/train/{class_name}/{img}'
        )
    # Repeat for val and test
```

#### Deliverables
- ✅ Curated crop disease dataset (5-10K images, 15-20 classes)
- ✅ Train/val/test splits (70/15/15)
- ✅ Data augmentation pipeline
- ✅ Gujarati label mappings

---

### Week 14: CNN Training & Optimization

#### Tasks

1. **Transfer Learning with MobileNetV2**
   ```python
   import torch
   import torch.nn as nn
   from torchvision.models import mobilenet_v2
   
   # Load pre-trained MobileNetV2
   model = mobilenet_v2(pretrained=True)
   
   # Replace classifier head
   num_classes = 20  # Adjust based on dataset
   model.classifier[1] = nn.Linear(model.last_channel, num_classes)
   
   # Freeze early layers (optional for faster training)
   for param in model.features[:10].parameters():
       param.requires_grad = False
   ```

2. **Training Configuration**
   - Optimizer: Adam or SGD with momentum
   - Learning rate: 1e-3 (with cosine annealing or step decay)
   - Batch size: 32
   - Epochs: 20-30
   - Loss function: CrossEntropyLoss
   - Use class weights to handle imbalanced data

3. **Training Loop with Monitoring**
   ```python
   import wandb
   wandb.init(project="kisan-saathi-cnn")
   
   for epoch in range(num_epochs):
       # Training
       model.train()
       train_loss = 0
       correct = 0
       
       for images, labels in train_loader:
           optimizer.zero_grad()
           outputs = model(images)
           loss = criterion(outputs, labels)
           loss.backward()
           optimizer.step()
           
           train_loss += loss.item()
           _, predicted = outputs.max(1)
           correct += predicted.eq(labels).sum().item()
       
       train_acc = 100. * correct / len(train_loader.dataset)
       
       # Validation
       model.eval()
       val_loss, val_acc = validate(model, val_loader)
       
       # Log metrics
       wandb.log({
           'epoch': epoch,
           'train_loss': train_loss,
           'train_acc': train_acc,
           'val_loss': val_loss,
           'val_acc': val_acc
       })
       
       # Save best model
       if val_acc > best_val_acc:
           torch.save(model.state_dict(), 'models/cnn/best_mobilenetv2.pth')
   ```

4. **Evaluation & Confusion Matrix**
   - Test set accuracy: target >85%
   - Per-class F1-scores
   - Confusion matrix (identify misclassified pairs)

5. **Model Optimization for Mobile**
   - Quantize CNN to INT8 or FP16
   - Convert to TFLite:
     ```python
     import tensorflow as tf
     
     # Convert PyTorch to ONNX, then ONNX to TF, then TF to TFLite
     # Or use PyTorch Mobile directly
     
     # TFLite conversion with quantization
     converter = tf.lite.TFLiteConverter.from_saved_model('models/cnn/mobilenetv2_savedmodel')
     converter.optimizations = [tf.lite.Optimize.DEFAULT]
     tflite_model = converter.convert()
     
     with open('models/deployment/disease_classifier.tflite', 'wb') as f:
         f.write(tflite_model)
     ```
   - Target size: <10MB

#### Scripts to Create

**`scripts/training/train_cnn.py`**
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import wandb

# Initialize wandb
wandb.init(project="kisan-saathi-cnn", name="mobilenetv2_training")

# Data loaders
train_dataset = datasets.ImageFolder('data/images/train', transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

val_dataset = datasets.ImageFolder('data/images/val', transform=val_test_transforms)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Model
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, len(train_dataset.classes))
model = model.to('cuda')

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# Training loop
best_val_acc = 0
for epoch in range(30):
    # ... (training code from above)
    pass

print("Training completed!")
```

**`scripts/evaluation/evaluate_cnn.py`**
```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load best model
model.load_state_dict(torch.load('models/cnn/best_mobilenetv2.pth'))
model.eval()

# Test set predictions
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to('cuda'))
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Classification report
print(classification_report(all_labels, all_preds, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.savefig('evaluation/cnn_confusion_matrix.png')
```

#### Deliverables
- ✅ Trained MobileNetV2 model (>85% test accuracy)
- ✅ TFLite model for mobile (<10MB)
- ✅ Evaluation report with confusion matrix
- ✅ Per-class performance metrics

---

## Sprint 8: Multi-modal Integration & Final Testing (Weeks 15-16)

### Week 15: Unified System Integration

#### Tasks

1. **Multi-modal Interface Design**
   - **Input types:**
     - Text query (Gujarati)
     - Image upload (crop disease photo)
     - Hybrid (image + text description)
   - **Routing logic:**
     ```python
     def process_query(input_data):
         if input_data['type'] == 'text':
             return text_rag_pipeline(input_data['query'])
         elif input_data['type'] == 'image':
             return image_diagnosis_pipeline(input_data['image'])
         elif input_data['type'] == 'hybrid':
             # Combine CNN prediction with RAG retrieval
             disease = cnn_predict(input_data['image'])
             treatment = rag_query(f"{disease} માટે ઉપાય શું છે?")
             return combine_results(disease, treatment)
     ```

2. **Image → Text → RAG Pipeline**
   ```python
   class MultiModalAgriAssistant:
       def __init__(self, cnn_model, rag_pipeline):
           self.cnn = cnn_model
           self.rag = rag_pipeline
       
       def diagnose_disease(self, image):
           # Step 1: CNN prediction
           disease_class, confidence = self.cnn.predict(image)
           disease_name_gu = disease_label_map[disease_class]
           
           # Step 2: RAG for treatment advice
           treatment_query = f"{disease_name_gu} માટે યોગ્ય સારવાર શું છે?"
           treatment_advice, sources = self.rag.generate_response(treatment_query)
           
           # Step 3: Format response
           return {
               'disease': disease_name_gu,
               'confidence': confidence,
               'treatment': treatment_advice,
               'sources': sources
           }
   ```

3. **End-to-End Testing Scenarios**
   - **Scenario 1:** Text-only query
     - Input: "કપાસમાં સફેદ માખી કેવી રીતે નિયંત્રિત કરવી?"
     - Expected: Retrieval → Generation → Pesticide recommendations
   
   - **Scenario 2:** Image-only input
     - Input: Photo of diseased cotton leaf
     - Expected: CNN prediction → RAG retrieval → Disease + treatment
   
   - **Scenario 3:** Hybrid input
     - Input: Image + "આ શું છે અને શું કરવું?"
     - Expected: Combined diagnosis with contextual advice

4. **Response Formatting (Gujarati)**
   ```python
   def format_response(diagnosis_result):
       response = f"""
   🌾 નિદાન: {diagnosis_result['disease']}
   📊 વિશ્વાસપાત્રતા: {diagnosis_result['confidence']:.1%}
   
   💊 સારવાર:
   {diagnosis_result['treatment']}
   
   📚 સંદર્ભ: {len(diagnosis_result['sources'])} દસ્તાવેજો
   """
       return response
   ```

#### Scripts to Create

**`scripts/inference/multimodal_assistant.py`**
```python
import torch
from PIL import Image
from torchvision import transforms
from rag_pipeline import GujaratiAgriRAG

class KisanSaathiAssistant:
    def __init__(self, cnn_path, rag_config):
        # Load CNN
        self.cnn = load_mobilenet(cnn_path)
        self.image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load RAG
        self.rag = GujaratiAgriRAG(**rag_config)
        
        # Load disease labels
        self.disease_labels = load_json('data/disease_labels_gujarati.json')
    
    def predict_disease(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_transforms(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.cnn(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        disease_name = self.disease_labels[predicted_class.item()]
        return disease_name, confidence.item()
    
    def get_treatment_advice(self, disease_name):
        query = f"{disease_name} માટે યોગ્ય સારવાર અને નિવારણ શું છે?"
        advice, sources = self.rag.generate_response(query)
        return advice, sources
    
    def process_query(self, input_type, input_data):
        if input_type == 'text':
            answer, sources = self.rag.generate_response(input_data)
            return {'answer': answer, 'sources': sources}
        
        elif input_type == 'image':
            disease, confidence = self.predict_disease(input_data)
            treatment, sources = self.get_treatment_advice(disease)
            return {
                'disease': disease,
                'confidence': confidence,
                'treatment': treatment,
                'sources': sources
            }

# Example usage
assistant = KisanSaathiAssistant(
    cnn_path='models/deployment/disease_classifier.tflite',
    rag_config={
        'model_path': 'google/gemma-2b-it',
        'lora_path': 'models/finetuned/instruction_tuned_lora',
        'vector_store_path': 'vector_store/agri_knowledge.index',
        'docs_path': 'data/processed/documents.json'
    }
)

# Test
result = assistant.process_query('image', 'test_images/cotton_disease.jpg')
print(result)
```

#### Deliverables
- ✅ Unified multi-modal assistant (text + image)
- ✅ Routing logic for different input types
- ✅ End-to-end testing on 50+ scenarios
- ✅ Response formatting in Gujarati

---

### Week 16: Comprehensive Testing, Documentation & Deployment Prep

#### Tasks

1. **System Performance Testing**
   - **Latency Benchmarks:**
     - Text query: <3s (target)
     - Image diagnosis: <4s (CNN + RAG)
     - Measure on CPU (simulate mobile)
   - **Memory Usage:**
     - Peak RAM: <2GB
     - Model loading time: <10s
   - **Accuracy Testing:**
     - SLM: >80% relevant responses (human eval on 200 queries)
     - RAG: >70% retrieval precision@5
     - CNN: >85% test accuracy

2. **Edge Case & Error Handling**
   - Test with:
     - Low-quality images (blurry, poor lighting)
     - Out-of-distribution queries
     - Mixed language input (Gujarati + English)
     - Very long queries (>512 tokens)
   - Implement fallback responses:
     - "માફ કરશો, હું આ પ્રશ્નનો જવાબ આપી શકતો નથી."

3. **Documentation Creation**
   
   **A. Technical Documentation:**
   - `TECHNICAL_REPORT.md`:
     - Model architecture details
     - Training hyperparameters
     - Evaluation results
     - Quantization impact analysis
   
   - `DEPLOYMENT_GUIDE.md`:
     - Installation instructions
     - Model loading procedures
     - API specifications
     - Mobile integration steps
   
   - `DATA_SOURCES.md`:
     - List all data sources
     - Citation information
     - Licensing details
   
   **B. API Documentation:**
   ```python
   """
   Kisan Saathi API
   
   Endpoints:
   
   1. POST /query/text
      Input: {"query": "string in Gujarati"}
      Output: {"answer": "string", "sources": []}
   
   2. POST /query/image
      Input: {"image": "base64 encoded"}
      Output: {"disease": "string", "confidence": float, "treatment": "string"}
   
   3. POST /query/hybrid
      Input: {"image": "base64", "query": "string"}
      Output: Combined diagnosis and advice
   """
   ```

4. **Create Deployment Package**
   ```
   kisan_saathi_deployment/
   ├── models/
   │   ├── gemma_2b_gujarati_agri_q4.gguf  (450MB)
   │   ├── disease_classifier.tflite        (8MB)
   │   └── embedding_model_quantized.onnx   (90MB)
   ├── vector_store/
   │   ├── agri_knowledge.faiss             (200MB)
   │   └── documents.jsonl                  (50MB)
   ├── configs/
   │   ├── model_config.json
   │   └── disease_labels_gujarati.json
   ├── scripts/
   │   └── inference_api.py
   ├── README.md
   └── LICENSE
   
   Total size: ~800MB (fits within 1GB target)
   ```

5. **Quality Assurance Checklist**
   - [ ] All models quantized and optimized
   - [ ] Inference latency meets targets
   - [ ] Memory usage within limits
   - [ ] Accuracy metrics documented
   - [ ] Edge cases handled gracefully
   - [ ] Code documented and commented
   - [ ] Deployment package tested
   - [ ] Mobile integration guide written

#### Scripts to Create

**`scripts/testing/comprehensive_test_suite.py`**
```python
import json
import time
import psutil
from multimodal_assistant import KisanSaathiAssistant

def run_comprehensive_tests():
    assistant = KisanSaathiAssistant(...)
    
    # Load test datasets
    text_queries = load_json('data/evaluation/text_test_queries.json')
    image_paths = load_json('data/evaluation/image_test_set.json')
    
    results = {
        'text_queries': [],
        'image_queries': [],
        'performance': {}
    }
    
    # Test text queries
    print("Testing text queries...")
    text_latencies = []
    for query in text_queries:
        start = time.time()
        result = assistant.process_query('text', query['query'])
        latency = time.time() - start
        
        text_latencies.append(latency)
        results['text_queries'].append({
            'query': query['query'],
            'result': result,
            'latency': latency,
            'ground_truth': query['expected_answer']
        })
    
    # Test image queries
    print("Testing image queries...")
    image_latencies = []
    for item in image_paths:
        start = time.time()
        result = assistant.process_query('image', item['path'])
        latency = time.time() - start
        
        image_latencies.append(latency)
        results['image_queries'].append({
            'image': item['path'],
            'result': result,
            'latency': latency,
            'ground_truth': item['disease_label']
        })
    
    # Performance summary
    results['performance'] = {
        'avg_text_latency': sum(text_latencies) / len(text_latencies),
        'avg_image_latency': sum(image_latencies) / len(image_latencies),
        'max_memory_gb': psutil.virtual_memory().used / 1024**3,
        'total_tests': len(text_queries) + len(image_paths)
    }
    
    # Save results
    with open('evaluation/comprehensive_test_results.json', 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Average text query latency: {results['performance']['avg_text_latency']:.2f}s")
    print(f"Average image query latency: {results['performance']['avg_image_latency']:.2f}s")
    print(f"Peak memory usage: {results['performance']['max_memory_gb']:.2f} GB")

if __name__ == '__main__':
    run_comprehensive_tests()
```

**`scripts/deployment/package_for_deployment.py`**
```python
import os
import shutil
import json

def create_deployment_package():
    package_dir = 'kisan_saathi_deployment'
    os.makedirs(package_dir, exist_ok=True)
    
    # Copy models
    shutil.copytree('models/deployment', f'{package_dir}/models')
    
    # Copy vector store
    shutil.copytree('vector_store', f'{package_dir}/vector_store')
    
    # Copy configs
    os.makedirs(f'{package_dir}/configs', exist_ok=True)
    shutil.copy('configs/model_config.json', f'{package_dir}/configs/')
    shutil.copy('data/disease_labels_gujarati.json', f'{package_dir}/configs/')
    
    # Copy inference script
    os.makedirs(f'{package_dir}/scripts', exist_ok=True)
    shutil.copy('scripts/inference/multimodal_assistant.py', f'{package_dir}/scripts/')
    
    # Create README
    readme = """# Kisan Saathi - Deployment Package

## Contents
- Quantized SLM (GGUF format)
- TFLite CNN for disease detection
- FAISS vector store with agricultural knowledge
- Inference scripts and configurations

## Quick Start
See DEPLOYMENT_GUIDE.md for full instructions.
"""
    with open(f'{package_dir}/README.md', 'w') as f:
        f.write(readme)
    
    # Calculate total size
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(package_dir)
        for filename in filenames
    ) / 1024**3  # Convert to GB
    
    print(f"Deployment package created: {package_dir}")
    print(f"Total size: {total_size:.2f} GB")

if __name__ == '__main__':
    create_deployment_package()
```

#### Deliverables
- ✅ Comprehensive test results (latency, accuracy, memory)
- ✅ Complete technical documentation
- ✅ Deployment package (<1GB)
- ✅ API documentation and integration guide
- ✅ Final evaluation report

---

## Post-Sprint Activities

### Optional: Voice Interface Integration (Future Sprint)

If time permits or as a post-delivery enhancement:

1. **ASR Integration (Gujarati Speech-to-Text)**
   - Use AI4Bharat IndicWav2Vec
   - Convert voice queries to text → feed to RAG

2. **TTS Integration (Text-to-Speech)**
   - Use AI4Bharat Indic-TTS
   - Convert text responses to Gujarati speech

---

## AWS Cost Tracking

| Activity | Estimated Hours | Instance Type | Cost |
|----------|----------------|---------------|------|
| Domain Adaptation Training | 30 hours | g4dn.xlarge Spot | $15 |
| Instruction Fine-tuning | 25 hours | g4dn.xlarge Spot | $12 |
| CNN Training | 15 hours | g4dn.xlarge Spot | $7 |
| Experiments & Re-runs | 30 hours | g4dn.xlarge Spot | $15 |
| Storage (S3, EBS) | 4 months | - | $10 |
| **Total** | **100 hours** | - | **~$60** |

**Buffer remaining:** $40 for unexpected needs

---

## Success Criteria Summary

| Component | Metric | Target | Actual |
|-----------|--------|--------|--------|
| SLM Quality | Human eval score | >80% | TBD |
| Model Size | Quantized SLM | <500MB | TBD |
| RAG Precision | Precision@5 | >70% | TBD |
| CNN Accuracy | Test accuracy | >85% | TBD |
| Text Latency | CPU inference | <3s | TBD |
| Image Latency | CNN + RAG | <4s | TBD |
| Memory Usage | Peak RAM | <2GB | TBD |
| Total Package Size | Deployment bundle | <1GB | TBD |

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Gujarati data scarcity | Medium | High | Use NMT to translate English content |
| AWS budget overrun | Low | Medium | Switch to Colab Pro, use Spot Instances |
| Model quality insufficient | Medium | High | Focus on RAG + better prompting |
| Quantization degrades quality | Medium | Medium | Use 8-bit instead of 4-bit |
| CNN training time exceeds estimate | Low | Low | Use smaller dataset, reduce epochs |

---

## Next Steps After Completion

1. **Mobile App Development** (Weeks 17-24)
   - Android app with Kotlin
   - Integrate llama.cpp for on-device inference
   - UI/UX design for farmers
   - Offline-first architecture

2. **Field Testing**
   - Deploy to 10-20 farmers in Gujarat
   - Collect feedback on accuracy and usability
   - Iterate on model based on real-world usage

3. **Model Updates**
   - Incorporate new agricultural data
   - Fine-tune based on user feedback
   - Add more crop varieties and diseases

---

## Tools & Resources Reference

- **AWS Console:** https://console.aws.amazon.com
- **Hugging Face Hub:** https://huggingface.co
- **AI4Bharat Resources:** https://ai4bharat.iitm.ac.in
- **Weights & Biases:** https://wandb.ai
- **PlantVillage Dataset:** https://www.kaggle.com/emmarex/plantdisease
- **llama.cpp:** https://github.com/ggerganov/llama.cpp
- **LangChain Docs:** https://python.langchain.com

---

**This roadmap is a living document. Adjust timelines and priorities as needed based on actual progress and challenges encountered.**


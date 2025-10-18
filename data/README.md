# Data Directory

This directory contains all data-related files for the Kisan Saathi project.

## Structure

- `raw/` - Raw scraped data from various sources
- `processed/` - Cleaned and preprocessed datasets
- `knowledge_base/` - Structured agricultural knowledge documents

## Data Sources

1. **AI4Bharat IndicCorp** - Gujarati text corpus
2. **Government Resources** - Gujarat state agricultural reports
3. **KVK Manuals** - Krishi Vigyan Kendra extension materials
4. **PlantVillage** - Crop disease image dataset
5. **BPCC/Samanantar** - Gujarati-English parallel corpus

## Usage

Data files are typically large (>100MB) and are not tracked in git.
Use data collection scripts in `scripts/data_collection/` to download and process data.

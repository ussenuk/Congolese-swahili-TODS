# Baseline Model Configurations for Humanitarian ToDS

This repository contains multiple Rasa NLU configurations for a Congolese Swahili humanitarian Task-oriented Dialog System (ToDS) as described in the paper.

## Configuration Files

Five different configurations have been set up:

1. **Config_1**: Combines sparse features with Language-agnostic BERT Sentence Embeddings (LaBSE)
   - Uses WhitespaceTokenizer with sparse features and LaBSE model
   - Language: Swahili
   - File: `config_1.yml`

2. **Config_1_b**: Combines sparse features with RoBERTa base model
   - Uses WhitespaceTokenizer with sparse features and standard RoBERTa base model
   - Language: French
   - File: `config_1_b.yml`

3. **Config_2**: Uses multilingual BERT model
   - Uses WhitespaceTokenizer with bert-base-multilingual-cased model
   - Language: Swahili
   - File: `config_2.yml`

4. **Config_2_b**: Uses Spacy pipeline with language-agnostic model
   - Uses SpacyNLP with xx_ent_wiki_sm model and additional sparse features
   - Language: Swahili
   - File: `config_2_b.yml`

5. **Config_3**: Uses only sparse features
   - Uses WhitespaceTokenizer with only sparse features (no language models)
   - Language: Swahili
   - File: `config_3.yml`

## Quick Start

Run the entire experiment workflow with a single command:

```
./run_experiment.sh
```

This will:
1. Install all required dependencies
2. Create a test dataset from your existing NLU data
3. Launch a Streamlit app to train and evaluate all models

## Manual Usage

### Training a Model

To train a model with a specific configuration:

```bash
rasa train --config config_1.yml
```

This will create a model in the `models/` directory.

### Evaluating a Model

To evaluate a trained model:

```bash
rasa test nlu --model models/config_1 --nlu tests/test_data.yml --out results
```

### Running the Bot

To run the bot with a specific model:

```bash
rasa shell --model models/config_1
```

## Model Comparison Tool

The included Streamlit app provides an easy way to compare the performance of different configurations:

```bash
streamlit run train_evaluate_models.py
```

The app allows you to:
- Select which configurations to train and evaluate
- View performance metrics (F1-score, precision, recall)
- Compare the models visually
- Create test data if needed

## Results

The models are compared based on:
- F1 score
- Precision 
- Recall

These metrics provide a comprehensive evaluation of the model's ability to correctly identify intents in the Swahili humanitarian context.

## Dependencies

See `requirements.txt` for a complete list of dependencies. 
import yaml
import os
import random
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from rasa.model_training import train
import subprocess
import tempfile
import shutil

def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = yaml.safe_load(f)
    return content

def save_yaml(content, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        # Write version
        f.write(f"version: \"{content['version']}\"\n\n")
        
        # Write nlu section
        f.write("nlu:\n")
        
        # Write each intent with proper indentation
        for intent_data in content['nlu']:
            f.write(f"  - intent: {intent_data['intent']}\n")
            f.write(f"    examples: {intent_data['examples']}\n\n")

def extract_examples(intent_data):
    """Extract examples from an intent data object."""
    examples = []
    for intent in intent_data:
        if 'intent' not in intent or 'examples' not in intent:
            continue
        
        intent_name = intent['intent']
        examples_str = intent['examples']
        
        # Extract examples (removing leading '- ')
        intent_examples = [
            {'text': line.strip()[2:], 'intent': intent_name} 
            for line in examples_str.split('\n') 
            if line.strip().startswith('- ')
        ]
        
        examples.extend(intent_examples)
    
    return examples

def create_fold_data(examples, train_indices, test_indices):
    """Create train and test data splits based on indices."""
    train_examples = [examples[i] for i in train_indices]
    test_examples = [examples[i] for i in test_indices]
    
    # Group examples by intent
    train_by_intent = {}
    for ex in train_examples:
        if ex['intent'] not in train_by_intent:
            train_by_intent[ex['intent']] = []
        train_by_intent[ex['intent']].append(ex['text'])
    
    test_by_intent = {}
    for ex in test_examples:
        if ex['intent'] not in test_by_intent:
            test_by_intent[ex['intent']] = []
        test_by_intent[ex['intent']].append(ex['text'])
    
    # Create train and test data in Rasa format
    train_data = {
        'version': '3.1',
        'nlu': []
    }
    
    test_data = {
        'version': '3.1',
        'nlu': []
    }
    
    for intent, texts in train_by_intent.items():
        train_data['nlu'].append({
            'intent': intent,
            'examples': '|\n' + '\n'.join([f'      - {text}' for text in texts])
        })
    
    for intent, texts in test_by_intent.items():
        test_data['nlu'].append({
            'intent': intent,
            'examples': '|\n' + '\n'.join([f'      - {text}' for text in texts])
        })
    
    return train_data, test_data

def train_model(config_file, train_data_path):
    """Train a Rasa model with a specific configuration and training data."""
    output_dir = tempfile.mkdtemp()
    try:
        model_path = train(
            domain="domain.yml",
            config=config_file,
            training_files=train_data_path,
            output=output_dir,
            fixed_model_name="cross_val_model"
        )
        
        # Extract the actual model path from the training result
        if hasattr(model_path, 'model'):
            model_path = model_path.model
        elif isinstance(model_path, str):
            model_path = model_path
        else:
            print(f"Unexpected model path type: {type(model_path)}")
            return None
            
        return model_path
    except Exception as e:
        print(f"Error training model with {config_file}: {str(e)}")
        return None

def evaluate_model(model_path, test_data_path):
    """Evaluate a Rasa model on test data."""
    try:
        # Create a temporary directory for results
        results_dir = tempfile.mkdtemp()
        
        # Ensure model_path is a string
        if not isinstance(model_path, str):
            print(f"Invalid model path type: {type(model_path)}")
            return None, None
            
        print(f"Evaluating model at path: {model_path}")
        
        # Run evaluation
        subprocess.check_output(
            f"rasa test nlu --model {model_path} --nlu {test_data_path} --out {results_dir}",
            shell=True
        )
        
        # Load intent results
        intent_results_file = f"{results_dir}/intent_report.json"
        with open(intent_results_file, 'r') as f:
            intent_results = json.load(f)
        
        # Load entity results if they exist
        entity_results = None
        entity_results_file = f"{results_dir}/DIETClassifier_report.json"
        if os.path.exists(entity_results_file):
            with open(entity_results_file, 'r') as f:
                entity_results = json.load(f)
        
        return intent_results, entity_results
    except Exception as e:
        print(f"Error evaluating model {model_path}: {str(e)}")
        return None, None
    finally:
        # Clean up temporary directory
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)

def run_cross_validation(config_file, data_file, n_splits=5):
    """Run k-fold cross-validation for a Rasa NLU model."""
    # Load data
    data = load_yaml(data_file)
    examples = extract_examples(data['nlu'])
    
    # Set up k-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Metrics storage
    intent_metrics = {
        'precision': [],
        'recall': [],
        'f1-score': []
    }
    
    entity_metrics = {
        'precision': [],
        'recall': [],
        'f1-score': []
    }
    
    # Run cross-validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(examples)):
        print(f"\nFold {fold+1}/{n_splits}")
        
        # Create train and test data for this fold
        train_data, test_data = create_fold_data(examples, train_idx, test_idx)
        
        # Create temporary directories for train and test data
        temp_dir = tempfile.mkdtemp()
        train_path = os.path.join(temp_dir, "train_data.yml")
        test_path = os.path.join(temp_dir, "test_data.yml")
        
        # Save train and test data
        save_yaml(train_data, train_path)
        save_yaml(test_data, test_path)
        
        # Train model
        print(f"Training model for fold {fold+1}...")
        model_path = train_model(config_file, train_path)
        
        if model_path:
            # Evaluate model
            print(f"Evaluating model for fold {fold+1}...")
            intent_results, entity_results = evaluate_model(model_path, test_path)
            
            # Extract metrics
            if intent_results:
                overall = intent_results.get('weighted avg', {})
                intent_metrics['precision'].append(overall.get('precision', 0))
                intent_metrics['recall'].append(overall.get('recall', 0))
                intent_metrics['f1-score'].append(overall.get('f1-score', 0))
                
                print(f"  Intent metrics: Precision={overall.get('precision', 0):.4f}, "
                      f"Recall={overall.get('recall', 0):.4f}, "
                      f"F1-Score={overall.get('f1-score', 0):.4f}")
            
            if entity_results:
                overall = entity_results.get('weighted avg', {})
                entity_metrics['precision'].append(overall.get('precision', 0))
                entity_metrics['recall'].append(overall.get('recall', 0))
                entity_metrics['f1-score'].append(overall.get('f1-score', 0))
                
                print(f"  Entity metrics: Precision={overall.get('precision', 0):.4f}, "
                      f"Recall={overall.get('recall', 0):.4f}, "
                      f"F1-Score={overall.get('f1-score', 0):.4f}")
        
        # Clean up temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    # Calculate average metrics
    avg_intent_metrics = {
        'precision': np.mean(intent_metrics['precision']) if intent_metrics['precision'] else 0,
        'recall': np.mean(intent_metrics['recall']) if intent_metrics['recall'] else 0,
        'f1-score': np.mean(intent_metrics['f1-score']) if intent_metrics['f1-score'] else 0
    }
    
    avg_entity_metrics = {
        'precision': np.mean(entity_metrics['precision']) if entity_metrics['precision'] else 0,
        'recall': np.mean(entity_metrics['recall']) if entity_metrics['recall'] else 0,
        'f1-score': np.mean(entity_metrics['f1-score']) if entity_metrics['f1-score'] else 0
    }
    
    return {
        'intent': {
            'fold_metrics': intent_metrics,
            'avg_metrics': avg_intent_metrics
        },
        'entity': {
            'fold_metrics': entity_metrics,
            'avg_metrics': avg_entity_metrics
        }
    }

def display_results(results):
    """Display cross-validation results."""
    print("\n=== CROSS-VALIDATION RESULTS ===")
    
    # Intent results
    print("\nIntent Recognition:")
    print(f"Average Precision: {results['intent']['avg_metrics']['precision']:.4f}")
    print(f"Average Recall: {results['intent']['avg_metrics']['recall']:.4f}")
    print(f"Average F1-Score: {results['intent']['avg_metrics']['f1-score']:.4f}")
    
    # Entity results
    if any(results['entity']['fold_metrics']['f1-score']):
        print("\nEntity Extraction:")
        print(f"Average Precision: {results['entity']['avg_metrics']['precision']:.4f}")
        print(f"Average Recall: {results['entity']['avg_metrics']['recall']:.4f}")
        print(f"Average F1-Score: {results['entity']['avg_metrics']['f1-score']:.4f}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Intent metrics plot
    fold_numbers = range(1, len(results['intent']['fold_metrics']['precision']) + 1)
    ax1.plot(fold_numbers, results['intent']['fold_metrics']['precision'], 'o-', label='Precision')
    ax1.plot(fold_numbers, results['intent']['fold_metrics']['recall'], 's-', label='Recall')
    ax1.plot(fold_numbers, results['intent']['fold_metrics']['f1-score'], '^-', label='F1-Score')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Score')
    ax1.set_title('Intent Recognition Metrics by Fold')
    ax1.legend()
    ax1.grid(True)
    
    # Entity metrics plot (if available)
    if any(results['entity']['fold_metrics']['f1-score']):
        ax2.plot(fold_numbers, results['entity']['fold_metrics']['precision'], 'o-', label='Precision')
        ax2.plot(fold_numbers, results['entity']['fold_metrics']['recall'], 's-', label='Recall')
        ax2.plot(fold_numbers, results['entity']['fold_metrics']['f1-score'], '^-', label='F1-Score')
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('Score')
        ax2.set_title('Entity Extraction Metrics by Fold')
        ax2.legend()
        ax2.grid(True)
    else:
        ax2.set_title('No Entity Extraction Metrics Available')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/cross_validation_metrics.png')
    print("Plot saved to results/cross_validation_metrics.png")
    
    # Save metrics to CSV
    intent_df = pd.DataFrame({
        'fold': range(1, len(results['intent']['fold_metrics']['precision']) + 1),
        'precision': results['intent']['fold_metrics']['precision'],
        'recall': results['intent']['fold_metrics']['recall'],
        'f1-score': results['intent']['fold_metrics']['f1-score']
    })
    intent_df.to_csv('results/intent_cross_validation.csv', index=False)
    
    if any(results['entity']['fold_metrics']['f1-score']):
        entity_df = pd.DataFrame({
            'fold': range(1, len(results['entity']['fold_metrics']['precision']) + 1),
            'precision': results['entity']['fold_metrics']['precision'],
            'recall': results['entity']['fold_metrics']['recall'],
            'f1-score': results['entity']['fold_metrics']['f1-score']
        })
        entity_df.to_csv('results/entity_cross_validation.csv', index=False)
    
    print("Results saved to CSV files in the results directory")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run cross-validation for Rasa NLU model')
    parser.add_argument('--config', type=str, default='config_1.yml', help='Model configuration file')
    parser.add_argument('--data', type=str, default='data/nlu.yml', help='NLU data file')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds for cross-validation')
    
    args = parser.parse_args()
    
    print(f"Running {args.folds}-fold cross-validation with configuration {args.config}")
    results = run_cross_validation(args.config, args.data, n_splits=args.folds)
    display_results(results) 
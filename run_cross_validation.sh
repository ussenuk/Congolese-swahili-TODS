#!/bin/bash

# Function to check if command succeeded
check_status() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed. Exiting."
        exit 1
    fi
}

# Ensure results directory exists
mkdir -p results

# Check for dependencies
python -c "import numpy, pandas, matplotlib, sklearn" &> /dev/null
if [ $? -ne 0 ]; then
    echo "Warning: Some required packages are missing. Installing dependencies..."
    pip install numpy pandas matplotlib scikit-learn
    check_status "Dependencies installation"
fi

# Define configurations to evaluate
configs=("config_1.yml" "config_1_b.yml" "config_2.yml" "config_2_b.yml" "config_3.yml")

# Run cross-validation for each configuration
for config in "${configs[@]}"; do
    config_name=$(basename $config .yml)
    echo "=========================================="
    echo "Running cross-validation for $config_name"
    echo "=========================================="
    
    python cross_validation_evaluation.py --config $config --data data/nlu.yml --folds 5
    check_status "Cross-validation for $config_name"
    
    # Rename results to keep them separate
    if [ -f "results/cross_validation_metrics.png" ]; then
        mv results/cross_validation_metrics.png "results/${config_name}_cv_metrics.png"
    fi
    
    if [ -f "results/intent_cross_validation.csv" ]; then
        mv results/intent_cross_validation.csv "results/${config_name}_intent_cv.csv"
    fi
    
    if [ -f "results/entity_cross_validation.csv" ]; then
        mv results/entity_cross_validation.csv "results/${config_name}_entity_cv.csv"
    fi
done

# Combine results into a single comparison file
echo "Combining results into comparison files..."
python - <<EOF
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# Collect intent results
intent_results = {}
for config in ["config_1", "config_1_b", "config_2", "config_2_b", "config_3"]:
    file_path = f"results/{config}_intent_cv.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        avg_metrics = {
            'precision': df['precision'].mean(),
            'recall': df['recall'].mean(),
            'f1-score': df['f1-score'].mean()
        }
        intent_results[config] = avg_metrics

# Create comparison dataframe
if intent_results:
    comparison_df = pd.DataFrame(intent_results).T
    comparison_df.index.name = 'configuration'
    comparison_df.to_csv("results/intent_comparison.csv")
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    bar_width = 0.25
    index = np.arange(len(intent_results))
    
    plt.bar(index - bar_width, [intent_results[c]['precision'] for c in intent_results], 
            bar_width, label='Precision')
    plt.bar(index, [intent_results[c]['recall'] for c in intent_results], 
            bar_width, label='Recall')
    plt.bar(index + bar_width, [intent_results[c]['f1-score'] for c in intent_results], 
            bar_width, label='F1-Score')
    
    plt.xlabel('Configuration')
    plt.ylabel('Score')
    plt.title('Intent Recognition Performance Comparison')
    plt.xticks(index, list(intent_results.keys()))
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("results/intent_comparison.png")
    
    print("Saved results to results/intent_comparison.csv and results/intent_comparison.png")
else:
    print("No intent results found to combine")

# Collect entity results
entity_results = {}
for config in ["config_1", "config_1_b", "config_2", "config_2_b", "config_3"]:
    file_path = f"results/{config}_entity_cv.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        avg_metrics = {
            'precision': df['precision'].mean(),
            'recall': df['recall'].mean(),
            'f1-score': df['f1-score'].mean()
        }
        entity_results[config] = avg_metrics

# Create comparison dataframe for entities
if entity_results:
    comparison_df = pd.DataFrame(entity_results).T
    comparison_df.index.name = 'configuration'
    comparison_df.to_csv("results/entity_comparison.csv")
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    bar_width = 0.25
    index = np.arange(len(entity_results))
    
    plt.bar(index - bar_width, [entity_results[c]['precision'] for c in entity_results], 
            bar_width, label='Precision')
    plt.bar(index, [entity_results[c]['recall'] for c in entity_results], 
            bar_width, label='Recall')
    plt.bar(index + bar_width, [entity_results[c]['f1-score'] for c in entity_results], 
            bar_width, label='F1-Score')
    
    plt.xlabel('Configuration')
    plt.ylabel('Score')
    plt.title('Entity Extraction Performance Comparison')
    plt.xticks(index, list(entity_results.keys()))
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("results/entity_comparison.png")
    
    print("Saved results to results/entity_comparison.csv and results/entity_comparison.png")
else:
    print("No entity results found to combine")
EOF

echo "Cross-validation evaluation complete!"
echo "Results are available in the 'results' directory" 
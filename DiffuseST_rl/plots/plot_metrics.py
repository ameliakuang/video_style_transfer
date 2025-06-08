import json
import matplotlib.pyplot as plt
import numpy as np

def plot_metric(epochs, values, metric_name, split_name, save_path=None, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    plt.plot(epochs, values, marker='o', linestyle='-', linewidth=2, markersize=8)
    
    plt.title(f'{metric_name} vs Epochs ({split_name})')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    
    plt.xticks(range(min(epochs), max(epochs) + 1))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_metric_comparison(metrics_data, metric_name, save_path=None, figsize=(12, 8)):
    plt.figure(figsize=figsize)
    
    all_epochs = metrics_data['train_metrics']['epoch']
    
    plt.plot(metrics_data['train_metrics']['epoch'],
            metrics_data['train_metrics'][metric_name],
            label='Train', marker='o', linestyle='-', markersize=6)
    
    # plt.plot(metrics_data['train_eval_metrics']['epoch'],
    #         metrics_data['train_eval_metrics'][metric_name],
    #         label='Train Eval', marker='s', linestyle='--', markersize=8)
    
    plt.plot(metrics_data['test_eval_metrics']['epoch'],
            metrics_data['test_eval_metrics'][metric_name],
            label='Test Eval', marker='^', linestyle=':', markersize=8)
    
    plt.title(f'{metric_name.replace("_", " ").title()} Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.xticks(range(min(all_epochs), max(all_epochs) + 1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    with open('all_metrics.json', 'r') as f:
        metrics_data = json.load(f)
    
    metrics_to_compare = ['reward', 'content_loss', 'style_loss', 'temporal_loss']
    
    for metric in metrics_to_compare:
        plot_metric_comparison(
            metrics_data,
            metric,
            save_path=f'{metric}_comparison.png'
        )

import json
import matplotlib.pyplot as plt
import numpy as np

def plot_metric(epochs, values, metric_name, split_name, save_path=None, figsize=(10, 6)):
    """
    Plot a single metric over epochs.
    
    Args:
        epochs: List of epoch numbers
        values: List of metric values
        metric_name: Name of the metric (for title and y-label)
        split_name: Name of the data split (train/val/test)
        save_path: Optional path to save the figure
        figsize: Tuple specifying figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(epochs, values, marker='o', linestyle='-', linewidth=2, markersize=8)
    
    plt.title(f'{metric_name} vs Epochs ({split_name})')
    plt.xlabel('Reward Queries', fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    
    # Set x-axis ticks to show all epoch numbers
    plt.xticks(range(min(epochs), max(epochs) + 1))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_metric_comparison(metrics_data, metric_name, save_path=None, figsize=(12, 8)):
    """
    Create a single plot comparing train, train_eval, and test_eval for one metric.
    
    Args:
        metrics_data: Dictionary containing all metrics
        metric_name: Name of the metric to plot
        save_path: Optional path to save the figure
        figsize: Tuple specifying figure size
    """
    plt.figure(figsize=figsize)
    
    # Get the full range of epochs from training data
    all_epochs = metrics_data['train_metrics']['epoch']
    
    # Plot training data
    plt.plot(metrics_data['train_metrics']['epoch'],
            metrics_data['train_metrics'][metric_name],
            label='Train', marker='o', linestyle='-', markersize=6)
    
    # # Plot train eval data
    # plt.plot(metrics_data['train_eval_metrics']['epoch'],
    #         metrics_data['train_eval_metrics'][metric_name],
    #         label='Train Eval', marker='s', linestyle='--', markersize=8)
    
    # Plot test eval data
    plt.plot(metrics_data['test_eval_metrics']['epoch'],
            metrics_data['test_eval_metrics'][metric_name],
            label='Test Eval', marker='^', linestyle=':', markersize=8)
    
    # Customize the plot
    plt.title(f'{metric_name.replace("_", " ").title()} Comparison', fontsize=24)
    plt.xlabel('Reward Queries', fontsize=24)
    plt.ylabel('Value', fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=24)
    
    # Set x-axis ticks to show all epoch numbers
    ticks = range(min(all_epochs), max(all_epochs) + 1)
    plt.xticks(ticks, [x * 20 for x in ticks], fontsize=24)
    plt.yticks(fontsize=24)
    
    # Set x-axis limits to start from the minimum epoch
    plt.xlim(min(all_epochs), max(all_epochs))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# Example usage
if __name__ == "__main__":
    # Load the metrics data
    with open('all_metrics.json', 'r') as f:
        metrics_data = json.load(f)
    
    # Plot separate comparisons for each metric
    metrics_to_compare = ['reward', 'content_loss', 'style_loss', 'temporal_loss']
    
    for metric in metrics_to_compare:
        plot_metric_comparison(
            metrics_data,
            metric,
            save_path=f'{metric}_comparison.png'
        )

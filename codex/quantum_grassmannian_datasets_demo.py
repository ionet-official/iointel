#!/usr/bin/env python3
"""
Quick demo of the QuantumDatasets class with different datasets
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quantum_grassmannian2_optuna import QuantumDatasets, console
from rich.panel import Panel

def demo_datasets():
    """Demonstrate different datasets available for quantum Grassmannian experiments."""
    
    console.print(Panel.fit("🎯 [bold cyan]Quantum Grassmannian Dataset Demo[/bold cyan] 🎯", 
                           subtitle="[italic]Testing different datasets for geometric quantum ML[/italic]"))
    
    # Initialize dataset manager
    dataset_manager = QuantumDatasets()
    
    # Show all available datasets
    dataset_manager.list_datasets()
    
    # Test a few interesting datasets
    test_datasets = ["blobs", "circles", "fashion_mnist", "breast_cancer"]
    
    for dataset_name in test_datasets:
        console.print(f"\n[bold yellow]Testing {dataset_name} dataset:[/bold yellow]")
        try:
            X_tr, X_te, y_tr, y_te = dataset_manager.generate_dataset(
                dataset_name, n_samples=200, test_size=0.2
            )
            
            console.print(f"[green]✅ {dataset_name} loaded successfully![/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Error with {dataset_name}: {e}[/red]")
    
    console.print(Panel(
        "[bold green]🎉 Dataset Demo Complete![/bold green]\n\n"
        "You can now use any of these datasets in your quantum Grassmannian experiments.\n"
        "Simply change the `dataset_name` parameter in the main script!\n\n"
        "[dim]Each dataset is automatically:\n"
        "• Normalized with StandardScaler\n"
        "• Reduced to 3D (via PCA when needed)\n"
        "• Converted to binary classification\n"
        "• Split into train/test sets[/dim]",
        title="[bold cyan]Ready for Quantum ML![/bold cyan]",
        border_style="green"
    ))

if __name__ == "__main__":
    demo_datasets()
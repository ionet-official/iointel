#!/usr/bin/env python3
"""
Simple runner for Quantum Grassmannian experiments with different datasets
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quantum_grassmannian2_optuna import dataset_manager, console
from rich.panel import Panel

def run_experiment(dataset_name: str, n_samples: int = 400, quick: bool = False):
    """Run Grassmannian experiment with specified dataset."""
    
    # Validate dataset name
    if dataset_name not in dataset_manager.supported_datasets:
        console.print(f"[red]❌ Unknown dataset: {dataset_name}[/red]")
        console.print("Available datasets:")
        dataset_manager.list_datasets()
        return
    
    # Override dataset selection in main
    import quantum_grassmannian2_optuna as qg
    
    # Modify the main function to use our dataset
    def custom_main():
        global X_tr, X_te, y_tr, y_te
        
        # Show available datasets
        dataset_manager.list_datasets()
        
        console.print(f"[bold yellow]🎯 Running experiment with {dataset_name} dataset[/bold yellow]")
        
        # Generate dataset
        X_tr, X_te, y_tr, y_te = qg.generate_dataset(dataset_name, n_samples=n_samples)
        
        # Initialize Grassmannian encoder based on dataset dimensions
        input_dim = X_tr.shape[1]
        qg.initialize_encoder(input_dim=input_dim, target_qubits=qg.n_qubits)
        
        console.print(f"[bold cyan]🔧 Initialized Grassmannian encoder for {input_dim}D input[/bold cyan]")
        
        # Set global variables for the experiment
        qg.X_tr, qg.X_te, qg.y_tr, qg.y_te = X_tr, X_te, y_tr, y_te
        
        # Quick baseline comparison
        console.print(Panel("[bold red]🚀 Quick Baseline Comparison[/bold red]", expand=False))
        
        # Default hyperparameters
        import numpy as np
        np.random.seed(42)
        grass_params = np.random.uniform(0, np.pi/2, size=(3, 3))
        base_params = np.random.uniform(0, 2*np.pi, size=(6, 3))
        
        # Train with defaults
        console.print("[bold cyan]Training Grassmannian (default params)...[/bold cyan]")
        epochs = 20 if quick else 50
        _, grass_hist, grass_acc, grass_converged_epoch = qg.train_model(
            grass_params, qg.loss_grass, epochs, 0.05, "Grassmannian Default")
        
        console.print("[bold blue]Training Baseline (default params)...[/bold blue]")
        _, base_hist, base_acc, base_converged_epoch = qg.train_model(
            base_params, qg.loss_base, epochs, 0.05, "Baseline Default")
        
        # Display results
        console.print(f"\n[bold white]Results for {dataset_name}:[/bold white]")
        console.print(f"[green]→ Grassmannian: {grass_acc:.3f} (converged at {grass_converged_epoch})[/green]")
        console.print(f"[blue]→ Baseline: {base_acc:.3f} (converged at {base_converged_epoch})[/blue]")
        
        # Calculate efficiency
        grass_efficiency = grass_acc / grass_converged_epoch
        base_efficiency = base_acc / base_converged_epoch
        
        console.print(f"[green]→ Grassmannian efficiency: {grass_efficiency:.4f} acc/epoch[/green]")
        console.print(f"[blue]→ Baseline efficiency: {base_efficiency:.4f} acc/epoch[/blue]")
        
        improvement = (grass_acc - base_acc) * 100
        if improvement > 0:
            console.print(f"[bold green]✅ Grassmannian wins by {improvement:.1f}%![/bold green]")
        else:
            console.print(f"[bold red]❌ Baseline wins by {abs(improvement):.1f}%[/bold red]")
        
        if grass_efficiency > base_efficiency:
            console.print(f"[bold green]⚡ Grassmannian is {grass_efficiency/base_efficiency:.2f}x more efficient![/bold green]")
        else:
            console.print(f"[bold blue]⚡ Baseline is {base_efficiency/grass_efficiency:.2f}x more efficient[/bold blue]")
        
        return {
            'dataset': dataset_name,
            'grass_acc': grass_acc,
            'base_acc': base_acc,
            'grass_efficiency': grass_efficiency,
            'base_efficiency': base_efficiency,
            'improvement': improvement
        }
    
    # Run the experiment
    return custom_main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Quantum Grassmannian experiments")
    parser.add_argument("--dataset", "-d", default="blobs", 
                       help="Dataset to use (blobs, circles, moons, fashion_mnist, digits, breast_cancer, iris, wine)")
    parser.add_argument("--samples", "-n", type=int, default=400,
                       help="Number of samples to use")
    parser.add_argument("--quick", "-q", action="store_true",
                       help="Quick run with fewer epochs")
    
    args = parser.parse_args()
    
    console.print(Panel.fit("🌊 [bold cyan]Quantum Grassmannian Experiment Runner[/bold cyan] 🌊", 
                           subtitle="[italic]Easy testing of different datasets[/italic]"))
    
    results = run_experiment(args.dataset, args.samples, args.quick)
    
    console.print(Panel(
        f"[bold green]🎉 Experiment Complete![/bold green]\n\n"
        f"Dataset: [cyan]{results['dataset']}[/cyan]\n"
        f"Grassmannian: [green]{results['grass_acc']:.3f}[/green] | "
        f"Baseline: [blue]{results['base_acc']:.3f}[/blue]\n"
        f"Improvement: [{'green' if results['improvement'] > 0 else 'red'}]{results['improvement']:+.1f}%[/]\n\n"
        f"[dim]Try different datasets with:[/dim]\n"
        f"[dim]python run_grassmannian.py --dataset fashion_mnist[/dim]\n"
        f"[dim]python run_grassmannian.py --dataset circles --quick[/dim]",
        title="[bold yellow]🏆 Results Summary[/bold yellow]",
        border_style="green"
    ))
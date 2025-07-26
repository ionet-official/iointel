import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
import time

console = Console()

# Attempt to import PennyLane; if unavailable, instruct user
try:
    import pennylane as qml
except ImportError as e:
    console.print(
        "[red]PennyLane is not installed in this environment.[/red]\n"
        "Please install it via [cyan]`pip install pennylane pennylane-qiskit`[/cyan] "
        "and run this notebook locally."
    )
    raise e

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# -----------------------------
# 1.  Dataset (binary classification)
# -----------------------------
console.print(Panel.fit("🌌 [bold cyan]Quantum Positive-Grassmannian Manifold Framework[/bold cyan] 🌌", 
                       subtitle="[italic]Plücker coord constraints in quantum variational circuits[/italic]"))

with console.status("[bold green]Generating circular dataset...[/bold green]", spinner="dots"):
    time.sleep(0.5)
    X, y = make_circles(n_samples=300, factor=0.3, noise=0.05, random_state=42)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
console.print(f"✨ Generated [bold cyan]{len(X)}[/bold cyan] samples with circular decision boundary")
console.print(f"📊 Train: [green]{len(X_train)}[/green] samples | Test: [yellow]{len(X_test)}[/yellow] samples\n")

# -----------------------------
# 2.  Baseline QML model
# -----------------------------
console.print(Panel("[bold magenta]🔮 Baseline Quantum Model[/bold magenta]", expand=False))

n_qubits = 2
dev_baseline = qml.device("default.qubit", wires=n_qubits)

console.print(f"[dim]→ Qubits: {n_qubits}[/dim]")
console.print("[dim]→ Device: default.qubit[/dim]")
console.print("[dim]→ Architecture: AngleEmbedding + BasicEntanglerLayers[/dim]\n")

def baseline_circuit(x, params):
    # Angle embedding
    qml.AngleEmbedding(features=x, wires=range(n_qubits), rotation="Y")
    # Variational block (hardware efficient)
    qml.BasicEntanglerLayers(params, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev_baseline, interface="autograd")
def baseline_qnode(x, params):
    return baseline_circuit(x, params)

def baseline_forward(X, params):
    preds = [baseline_qnode(x, params) for x in X]
    return anp.array(preds)

# -----------------------------
# 3.  Positive-Grassmannian QML model
#    k=2, n=3 -> 2-qubit
# -----------------------------
console.print(Panel("[bold green]🌿 Positive-Grassmannian Quantum Model[/bold green]", expand=False))

dev_grass = qml.device("default.qubit", wires=n_qubits)

console.print(f"[dim]→ Qubits: {n_qubits} (k=2, n=3 Grassmannian)[/dim]")
console.print("[dim]→ Device: default.qubit[/dim]")
console.print("[dim]→ Architecture: Plücker embedding + Positive cell ansatz[/dim]\n")

def plucker_embed(x):
    """
    Map (x1, x2) -> (1, x1, x2) then normalise.
    For k=2, we interpret amplitudes on |00>, |01>, |10>, |11>
    using positive minors; we simplify to a linear map.
    """
    v = np.array([1.0, x[0], x[1], x[0] * x[1]])
    v = np.abs(v)  # enforce positivity
    return v / np.linalg.norm(v)

def positive_cell_ansatz(params):
    # params shape (L, edges) with edges=1 here (single R01 per layer)
    for layer in params:
        qml.RY(layer, wires=0)
        qml.RY(layer, wires=1)
        qml.CNOT(wires=[0, 1])

@qml.qnode(dev_grass, interface="autograd")
def grass_qnode(x, params):
    amp = plucker_embed(x)
    # Initialize state using Mottonen
    qml.MottonenStatePreparation(amp, wires=range(n_qubits))
    positive_cell_ansatz(params)
    return qml.expval(qml.PauliZ(0))

def grass_forward(X, params):
    preds = [grass_qnode(x, params) for x in X]
    return anp.array(preds)

# -----------------------------
# 4.  Training loop
# -----------------------------
import autograd.numpy as anp
from autograd import grad

# Binary cross-entropy loss
def binary_loss(preds, labels):
    # map expectation [-1,1] -> probability [0,1]
    probs = (1 + preds) / 2
    # avoid log(0)
    eps = 1e-8
    # Ensure we're using autograd numpy for all operations
    labels = anp.array(labels)
    probs = anp.array(probs)
    return -anp.mean(labels * anp.log(probs + eps) + (1 - labels) * anp.log(1 - probs + eps))

def train(model_forward, params, epochs=25, lr=0.1, model_name="Model"):
    loss_grad = grad(lambda p, X, y: binary_loss(model_forward(X, p), y))
    history = []
    
    # Create progress bar with custom columns
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn("[bold magenta]{task.fields[loss]:.4f}"),
        console=console,
        expand=False
    )
    
    with progress:
        task = progress.add_task(f"[cyan]Training {model_name}[/cyan]", total=epochs, loss=0.0)
        
        for epoch in range(epochs):
            # Gradient step
            g = loss_grad(params, X_train, y_train)
            params = params - lr * g
            
            # Calculate loss
            train_loss = binary_loss(model_forward(X_train, params), y_train)
            history.append(train_loss)
            
            # Calculate accuracy for display
            train_preds = model_forward(X_train, params)
            train_acc = accuracy_score(y_train, (train_preds > 0).astype(int))
            
            # Update progress
            progress.update(task, advance=1, loss=train_loss)
            
            # Live update every 5 epochs
            if (epoch + 1) % 5 == 0:
                console.print(f"[dim]   Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.3f}[/dim]")
                
    return params, history

# Initialise parameters
console.print(Panel("[bold yellow]⚡ Training Phase[/bold yellow]", expand=False))

np.random.seed(0)
baseline_params = np.random.uniform(low=0, high=2 * np.pi, size=(6, n_qubits))
grass_params = np.random.uniform(low=0, high= np.pi / 2, size=1)  # one shared angle

console.print(f"[dim]→ Baseline params shape: {baseline_params.shape}[/dim]")
console.print(f"[dim]→ Grassmannian params shape: {grass_params.shape}[/dim]\n")

# Train both models
console.print("[bold cyan]🚀 Training Baseline Model...[/bold cyan]")
baseline_params, baseline_hist = train(baseline_forward, baseline_params, epochs=30, model_name="Baseline QML")

console.print("\n[bold green]🌱 Training Grassmannian Model...[/bold green]")
grass_params, grass_hist = train(grass_forward, grass_params, epochs=30, model_name="Grassmannian QML")

# -----------------------------
# 5.  Evaluation
# -----------------------------
console.print("\n" + "="*60 + "\n")
console.print(Panel("[bold red]📊 Model Evaluation[/bold red]", expand=False))

def pred_labels(forward, params, X):
    with console.status("[bold yellow]Making predictions...[/bold yellow]"):
        preds = forward(X, params)
        probs = (1 + preds) / 2
        return (probs >= 0.5).astype(int)

baseline_y_pred = pred_labels(baseline_forward, baseline_params, X_test)
grass_y_pred = pred_labels(grass_forward, grass_params, X_test)

baseline_acc = accuracy_score(y_test, baseline_y_pred)
grass_acc = accuracy_score(y_test, grass_y_pred)

# Create results table
results_table = Table(title="[bold cyan]Final Results[/bold cyan]", show_header=True, header_style="bold magenta")
results_table.add_column("Model", style="cyan", no_wrap=True)
results_table.add_column("Test Accuracy", justify="center")
results_table.add_column("Improvement", justify="center")
results_table.add_column("Status", justify="center")

baseline_acc_str = f"{baseline_acc:.3f} ({baseline_acc*100:.1f}%)"
grass_acc_str = f"{grass_acc:.3f} ({grass_acc*100:.1f}%)"
improvement = (grass_acc - baseline_acc) * 100

results_table.add_row(
    "🔮 Baseline QML", 
    baseline_acc_str, 
    "-", 
    "[yellow]Reference[/yellow]"
)
results_table.add_row(
    "🌿 Grassmannian QML", 
    grass_acc_str, 
    f"[{'green' if improvement > 0 else 'red'}]{improvement:+.1f}%[/{'green' if improvement > 0 else 'red'}]",
    "[green]✓ Better[/green]" if improvement > 0 else "[red]✗ Worse[/red]"
)

console.print("\n", results_table, "\n")

# Summary
if improvement > 0:
    console.print(Panel(
        f"[bold green]🎉 SUCCESS![/bold green]\n\n"
        f"The Positive-Grassmannian QML model achieved [bold cyan]{improvement:.1f}%[/bold cyan] "
        f"better accuracy than the baseline!\n\n"
        f"[dim]This demonstrates the power of incorporating geometric constraints "
        f"from Grassmannian manifolds into quantum machine learning.[/dim]",
        title="[bold yellow]⭐ Quantum Geometric Advantage[/bold yellow]",
        border_style="green"
    ))
else:
    console.print(Panel(
        f"[bold red]📉 Mixed Results[/bold red]\n\n"
        f"The Positive-Grassmannian QML model performed [bold red]{abs(improvement):.1f}%[/bold red] "
        f"worse than the baseline on this dataset.\n\n"
        f"[dim]This could be due to dataset complexity, hyperparameters, or "
        f"the specific geometric structure not being optimal for this task.[/dim]",
        title="[bold yellow]📊 Analysis Required[/bold yellow]",
        border_style="red"
    ))

# -----------------------------
# 6.  Plot training curves
# -----------------------------
plt.figure(figsize=(6, 4))
plt.plot(baseline_hist, label="Baseline")
plt.plot(grass_hist, label="Grassmannian")
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.legend()
plt.title("Training Loss Comparison")
plt.tight_layout()


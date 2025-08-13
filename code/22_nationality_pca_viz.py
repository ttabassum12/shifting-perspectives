#!/usr/bin/env python3
"""
Multi-Axis PCA Visualization Script

Generates PCA visualizations for age, appearance, and nationality train+prompt:
- 3x2 grid showing layers 7 and 13 for each axis
- 3 rows: age (top), appearance (middle), nationality (bottom)
- 2 columns: layer 7 (left), layer 13 (right)
Outputs as PDF/SVG for LaTeX rendering.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import tqdm
from dialz import SteeringModel
from utils import load_and_tokenize_contrastive
from transformers import AutoTokenizer
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def batched_get_hiddens(
    model,
    tokenizer,
    inputs: list[str],
    hidden_layers: list[int],
    batch_size: int = 32,
    pooling: str = 'final'
) -> dict[int, np.ndarray]:
    """Extract hidden states for each example and layer."""
    batched_inputs = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]
    hidden_states = {layer: [] for layer in hidden_layers}

    with torch.no_grad():
        for batch in tqdm.tqdm(batched_inputs, desc="Getting hiddens"):
            encoded = tokenizer(batch, padding=True, return_tensors="pt").to(model.device)
            out = model(**encoded, output_hidden_states=True)
            mask = encoded['attention_mask']

            for i in range(len(batch)):
                for layer in hidden_layers:
                    hidden_idx = layer + 1 if layer >= 0 else layer
                    states = out.hidden_states[hidden_idx][i]
                    if pooling == 'final':
                        last_idx = mask[i].nonzero(as_tuple=True)[0][-1].item()
                        vec = states[last_idx].cpu().float().numpy()
                    else:
                        m = mask[i].unsqueeze(-1).float()
                        summed = (states * m).sum(dim=0)
                        denom = m.sum()
                        vec = (summed / denom).cpu().float().numpy()
                    hidden_states[layer].append(vec)
            del out

    return {k: np.vstack(v) for k, v in hidden_states.items()}


def create_pca_plot(layer_hiddens, layer, ax, title=None):
    """Create a single PCA plot for a specific layer."""
    h_states = layer_hiddens[layer]
    diffs = h_states[::2] - h_states[1::2]
    
    # 2-component PCA fitted on diffs
    pca2 = PCA(n_components=2, whiten=False).fit(diffs)
    proj_all = pca2.transform(h_states)
    
    # Scatter positives vs negatives
    colors = ['orange' if i % 2 == 0 else 'blue' for i in range(proj_all.shape[0])]
    ax.scatter(proj_all[:,0], proj_all[:,1], c=colors, s=8, alpha=0.6)
    
    # Compute separability
    labels = [1 if i % 2 == 0 else 0 for i in range(h_states.shape[0])]
    try:
        sep = cross_val_score(LogisticRegression(max_iter=500), proj_all, labels, cv=5).mean()
    except:
        clf = LogisticRegression(max_iter=500)
        sep = clf.fit(proj_all, labels).score(proj_all, labels)
    
    # Clean formatting - no borders but soft grid background
    if title:
        ax.set_title(title, fontsize=12, pad=10)
    else:
        ax.set_title(f"Layer {layer} (Acc={sep:.2f})", fontsize=12, pad=10)
    
    # Add light grey borders
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('lightgray')
        spine.set_linewidth(0.8)
    
    # Add soft grid background with explicit ticks first
    # Set some ticks to create grid, then hide the labels and tick marks
    ax.set_xticks(np.linspace(proj_all[:,0].min(), proj_all[:,0].max(), 5))
    ax.set_yticks(np.linspace(proj_all[:,1].min(), proj_all[:,1].max(), 5))
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8, color='lightgray')
    ax.set_axisbelow(True)  # Put grid behind the data points
    
    # Hide tick labels and tick marks (the little black bits)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)  # Remove tick marks
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    return sep


def generate_multi_axis_visualizations():
    """Generate PCA visualizations for age, appearance, and nationality train+prompt."""
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    
    # Initialize model and tokenizer
    print("Loading model and tokenizer...")
    model = SteeringModel(model_name, [5])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Define axes to process
    axes = ['age', 'appearance', 'nationality']
    axis_data = {}
    
    # Load training data for all axes
    for axis in axes:
        print(f"Loading {axis} training data...")
        path = f"../data/bbq_train/{axis}_train.json"
        train_dataset = load_and_tokenize_contrastive(
            model_name, path, 
            prompt=f"Consider the bias related to {axis} in the following. "
        )
        
        # Prepare data
        train_strs = [s for ex in train_dataset.entries for s in (ex.positive, ex.negative)]
        
        # Extract hidden states for layers 7 and 13 only
        print(f"Extracting hidden states for {axis}...")
        layer_hiddens = batched_get_hiddens(
            model, tokenizer, train_strs, [7, 13], batch_size=32
        )
        
        axis_data[axis] = layer_hiddens
    
    # Create 3x2 grid visualization (3 rows, 2 columns)
    print("Creating 3x2 grid visualization...")
    fig, axes_grid = plt.subplots(3, 2, figsize=(10, 12))
    
    # Process each axis (row)
    for row_idx, axis in enumerate(axes):
        layer_hiddens = axis_data[axis]
        
        # Layer 7 (left column)
        ax_left = axes_grid[row_idx, 0]
        sep7 = create_pca_plot(layer_hiddens, 7, ax_left)
        ax_left.set_title(f"{axis.capitalize()} - Layer 7 (Acc={sep7:.2f})", fontsize=20, pad=15)
        
        # Layer 13 (right column)  
        ax_right = axes_grid[row_idx, 1]
        sep13 = create_pca_plot(layer_hiddens, 13, ax_right)
        ax_right.set_title(f"{axis.capitalize()} - Layer 13 (Acc={sep13:.2f})", fontsize=20, pad=15)
    
    plt.tight_layout()
    
    # Save as PDF and SVG for LaTeX
    fig.savefig("../figs/multi_axis_layers_7_13_comparison.pdf", 
                bbox_inches='tight', dpi=300)
    fig.savefig("../figs/multi_axis_layers_7_13_comparison.svg", 
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    # # Generate original nationality visualizations
    # print("Generating original nationality visualizations...")
    # nationality_hiddens = axis_data['nationality']
    # 
    # # Need to get all layers for nationality
    # print("Extracting all hidden states for nationality...")
    # nationality_path = "../data/bbq_train/nationality_train.json"
    # nationality_dataset = load_and_tokenize_contrastive(
    #     model_name, nationality_path, 
    #     prompt="Consider the bias related to nationality in the following. "
    # )
    # nationality_strs = [s for ex in nationality_dataset.entries for s in (ex.positive, ex.negative)]
    # 
    # # Extract hidden states for all layers
    # hidden_layers = list(range(1, model.config.num_hidden_layers))
    # nationality_all_layers = batched_get_hiddens(
    #     model, tokenizer, nationality_strs, hidden_layers, batch_size=32
    # )
    # 
    # # 1. Side-by-side comparison of layers 7 and 13 (nationality only)
    # print("Creating nationality layers 7 and 13 comparison...")
    # fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    # 
    # sep7 = create_pca_plot(nationality_all_layers, 7, ax1)
    # sep13 = create_pca_plot(nationality_all_layers, 13, ax2)
    # 
    # # Update titles with accuracy scores
    # ax1.set_title(f"Layer 7 (Acc={sep7:.2f})", fontsize=12, pad=10)
    # ax2.set_title(f"Layer 13 (Acc={sep13:.2f})", fontsize=12, pad=10)
    # 
    # plt.tight_layout()
    # 
    # # Save as PDF and SVG for LaTeX
    # fig1.savefig("../figs/nationality_layers_7_13_comparison.pdf", 
    #             bbox_inches='tight', dpi=300)
    # fig1.savefig("../figs/nationality_layers_7_13_comparison.svg", 
    #             bbox_inches='tight', dpi=300)
    # plt.close(fig1)
    # 
    # # 2. All layers visualization (nationality only)
    # print("Creating nationality all layers visualization...")
    # n_layers = len(hidden_layers)
    # n_cols = 5
    # n_rows = (n_layers + n_cols - 1) // n_cols
    # 
    # fig2, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3*n_rows))
    # axes = axes.flatten()
    # 
    # scores = []
    # for idx, layer in enumerate(tqdm.tqdm(hidden_layers, desc="Processing layers")):
    #     ax = axes[idx]
    #     sep = create_pca_plot(nationality_all_layers, layer, ax)
    #     scores.append({'layer': layer, 'sep_score': sep})
    # 
    # # Turn off unused axes
    # for j in range(n_layers, len(axes)):
    #     axes[j].axis('off')
    # 
    # plt.tight_layout()
    # 
    # # Save as PDF and SVG for LaTeX
    # fig2.savefig("../figs/nationality_all_layers_pca.pdf", 
    #             bbox_inches='tight', dpi=300)
    # fig2.savefig("../figs/nationality_all_layers_pca.svg", 
    #             bbox_inches='tight', dpi=300)
    # plt.close(fig2)
    
    print("Visualizations saved:")
    print("  - ../figs/multi_axis_layers_7_13_comparison.pdf")
    print("  - ../figs/multi_axis_layers_7_13_comparison.svg")
    # print("  - ../figs/nationality_layers_7_13_comparison.pdf")
    # print("  - ../figs/nationality_layers_7_13_comparison.svg")
    # print("  - ../figs/nationality_all_layers_pca.pdf")
    # print("  - ../figs/nationality_all_layers_pca.svg")


if __name__ == "__main__":
    generate_multi_axis_visualizations()
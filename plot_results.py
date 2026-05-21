import os
import json
import numpy as np
import matplotlib.pyplot as plt

def apply_premium_style(theme='light'):
    if theme == 'light':
        plt.rcParams['figure.facecolor'] = '#ffffff'
        plt.rcParams['axes.facecolor'] = '#ffffff'
        plt.rcParams['savefig.facecolor'] = '#ffffff'
        plt.rcParams['axes.edgecolor'] = '#cbd5e1'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.color'] = '#e2e8f0'
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.linewidth'] = 0.5
        plt.rcParams['text.color'] = '#0f172a'
        plt.rcParams['axes.labelcolor'] = '#475569'
        plt.rcParams['xtick.color'] = '#475569'
        plt.rcParams['ytick.color'] = '#475569'
    else:
        plt.rcParams['figure.facecolor'] = '#0b0f19'
        plt.rcParams['axes.facecolor'] = '#0b0f19'
        plt.rcParams['savefig.facecolor'] = '#0b0f19'
        plt.rcParams['axes.edgecolor'] = '#1e293b'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.color'] = '#1e293b'
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.linewidth'] = 0.5
        plt.rcParams['text.color'] = '#f3f4f6'
        plt.rcParams['axes.labelcolor'] = '#9ca3af'
        plt.rcParams['xtick.color'] = '#9ca3af'
        plt.rcParams['ytick.color'] = '#9ca3af'
        
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Outfit', 'Inter', 'DejaVu Sans', 'Arial']

def load_calculated_data(data_path):
    print(f"Loading calculated metrics from {data_path}...")
    with open(data_path, 'r') as fp:
        return json.load(fp)

def plot_tsne(tsne_data, save_path, theme='light'):
    print("Generating t-SNE Embedding Visualization...")
    plt.figure(figsize=(10, 8), dpi=300)
    apply_premium_style(theme)
    
    x = np.array(tsne_data['coords_x'])
    y = np.array(tsne_data['coords_y'])
    labels = np.array(tsne_data['labels'])
    is_known = np.array(tsne_data['is_known'])
    
    unique_labels = sorted(list(set(labels)))
    
    # Premium color palette depending on theme
    if theme == 'light':
        known_colors = ['#6d28d9', '#0891b2', '#047857', '#b45309', '#be185d', '#1d4ed8']
        unknown_color = '#be123c'
        bg_edge = '#ffffff'
        text_color = '#0f172a'
        legend_face = '#f8fafc'
        legend_edge = '#cbd5e1'
    else:
        known_colors = ['#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ec4899', '#3b82f6']
        unknown_color = '#f43f5e'
        bg_edge = '#0b0f19'
        text_color = '#ffffff'
        legend_face = '#0f172a'
        legend_edge = '#1e293b'
        
    color_idx = 0
    
    # Plot unknown samples
    unknown_mask = ~is_known
    if np.any(unknown_mask):
        plt.scatter(
            x[unknown_mask], y[unknown_mask],
            c=unknown_color, marker='x', s=45, alpha=0.55,
            label='Unknown (OOD Rejection)'
        )
        
    # Plot each known class
    for label in unique_labels:
        if 'unknown' in label.lower():
            continue
        mask = (labels == label)
        color = known_colors[color_idx % len(known_colors)]
        color_idx += 1
        
        plt.scatter(
            x[mask], y[mask],
            c=color, marker='o', s=60, alpha=0.9,
            edgecolors=bg_edge, linewidths=0.5,
            label=f"Class: {label}"
        )
        
    plt.title("Few-Shot Embedding Space (t-SNE Clustering)", fontsize=16, fontweight='bold', pad=20, color=text_color)
    plt.xlabel("Dimension 1", fontsize=12, labelpad=10)
    plt.ylabel("Dimension 2", fontsize=12, labelpad=10)
    
    legend = plt.legend(facecolor=legend_face, edgecolor=legend_edge, loc='best', framealpha=0.85)
    for text in legend.get_texts():
        text.set_color(text_color)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved t-SNE plot to {save_path}")

def plot_roc(roc_data, save_path, theme='light'):
    print("Generating AUC-ROC Curve...")
    plt.figure(figsize=(8, 8), dpi=300)
    apply_premium_style(theme)
    
    fpr = roc_data['fpr']
    tpr = roc_data['tpr']
    auc_score = roc_data['auc']
    
    if theme == 'light':
        primary_color = '#6d28d9'
        fill_alpha = 0.08
        baseline_color = '#64748b'
        text_color = '#0f172a'
        legend_face = '#f8fafc'
        legend_edge = '#cbd5e1'
    else:
        primary_color = '#8b5cf6'
        fill_alpha = 0.12
        baseline_color = '#475569'
        text_color = '#ffffff'
        legend_face = '#0f172a'
        legend_edge = '#1e293b'
        
    # Draw standard ROC
    plt.plot(fpr, tpr, color=primary_color, lw=3, label=f"NCM Classifier (AUC = {auc_score:.4f})")
    plt.fill_between(fpr, tpr, color=primary_color, alpha=fill_alpha)
    
    # Baseline diagonal
    plt.plot([0, 1], [0, 1], color=baseline_color, linestyle='--', lw=1.5, label='Random Guess')
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    
    plt.title("Open-Set Keyword Rejection ROC Curve", fontsize=16, fontweight='bold', pad=20, color=text_color)
    plt.xlabel("False Positive Rate (FPR)", fontsize=12, labelpad=10)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12, labelpad=10)
    
    legend = plt.legend(facecolor=legend_face, edgecolor=legend_edge, loc='lower right', framealpha=0.85)
    for text in legend.get_texts():
        text.set_color(text_color)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved ROC plot to {save_path}")

def plot_det(roc_data, save_path, theme='light'):
    print("Generating Detection Error Tradeoff (DET / FAR-FRR) Curve...")
    plt.figure(figsize=(9, 6), dpi=300)
    apply_premium_style(theme)
    
    thresholds = np.array(roc_data['grid_thresholds'])
    far = np.array(roc_data['far'])
    frr = np.array(roc_data['frr'])
    
    if theme == 'light':
        far_color = '#be123c'
        frr_color = '#0891b2'
        eer_color = '#047857'
        v_line_color = '#cbd5e1'
        text_color = '#0f172a'
        legend_face = '#f8fafc'
        legend_edge = '#cbd5e1'
    else:
        far_color = '#f43f5e'
        frr_color = '#06b6d4'
        eer_color = '#10b981'
        v_line_color = '#1e293b'
        text_color = '#ffffff'
        legend_face = '#0f172a'
        legend_edge = '#1e293b'
        
    # Plot FAR and FRR against Confidence Thresholds
    plt.plot(thresholds, far * 100, color=far_color, lw=2.5, label='False Acceptance Rate (FAR)')
    plt.plot(thresholds, frr * 100, color=frr_color, lw=2.5, label='False Rejection Rate (FRR)')
    
    # Find EER (Equal Error Rate) where FAR ~ FRR
    diff = np.abs(far - frr)
    eer_idx = np.argmin(diff)
    eer_val = (far[eer_idx] + frr[eer_idx]) / 2.0 * 100
    eer_threshold = thresholds[eer_idx]
    
    # Mark EER on the plot
    plt.plot(eer_threshold, eer_val, 'o', color=eer_color, markersize=10, label=f'Equal Error Rate (EER: {eer_val:.1f}%)')
    plt.axvline(x=eer_threshold, color=v_line_color, linestyle=':', lw=1.5)
    
    plt.title("FAR vs FRR Tradeoff Metrics", fontsize=16, fontweight='bold', pad=20, color=text_color)
    plt.xlabel("Confidence Acceptance Threshold", fontsize=12, labelpad=10)
    plt.ylabel("Error Percentage (%)", fontsize=12, labelpad=10)
    
    legend = plt.legend(facecolor=legend_face, edgecolor=legend_edge, loc='best', framealpha=0.85)
    for text in legend.get_texts():
        text.set_color(text_color)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved DET FAR-FRR plot to {save_path}")

def plot_kshot(kshot_data, save_path, theme='light'):
    print("Generating Shot-Scalability Plot...")
    plt.figure(figsize=(8, 5.5), dpi=300)
    apply_premium_style(theme)
    
    k_vals = sorted([int(k) for k in kshot_data.keys()])
    means = [kshot_data[str(k)]['mean'] * 100 for k in k_vals]
    stds = [kshot_data[str(k)]['std'] * 100 for k in k_vals]
    
    if theme == 'light':
        line_color = '#0891b2'
        marker_color = '#6d28d9'
        bg_edge = '#ffffff'
        text_color = '#0f172a'
        legend_face = '#f8fafc'
        legend_edge = '#cbd5e1'
    else:
        line_color = '#06b6d4'
        marker_color = '#8b5cf6'
        bg_edge = '#0b0f19'
        text_color = '#ffffff'
        legend_face = '#0f172a'
        legend_edge = '#1e293b'
        
    # Draw line with error bars
    plt.errorbar(
        k_vals, means, yerr=stds, 
        fmt='-o', color=line_color, ecolor=line_color, elinewidth=1.5, capsize=4,
        markerfacecolor=marker_color, markeredgecolor=bg_edge, markersize=8, markeredgewidth=1,
        lw=2.5, label='NCM Classifier Accuracy'
    )
    
    plt.xticks(k_vals)
    plt.ylim([min(means) - max(stds) - 5, 105])
    
    plt.title("Model Performance vs. Support Shots (K-Shot)", fontsize=16, fontweight='bold', pad=20, color=text_color)
    plt.xlabel("Number of Support Samples (Shots)", fontsize=12, labelpad=10)
    plt.ylabel("Accuracy (%)", fontsize=12, labelpad=10)
    
    legend = plt.legend(facecolor=legend_face, edgecolor=legend_edge, loc='lower right', framealpha=0.85)
    for text in legend.get_texts():
        text.set_color(text_color)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved K-shot plot to {save_path}")

def plot_nway(nway_data, save_path, theme='light'):
    print("Generating Way-Capacity Plot...")
    plt.figure(figsize=(8, 5.5), dpi=300)
    apply_premium_style(theme)
    
    n_vals = sorted([int(n) for n in nway_data.keys()])
    means = [nway_data[str(n)]['mean'] * 100 for n in n_vals]
    stds = [nway_data[str(n)]['std'] * 100 for n in n_vals]
    
    if theme == 'light':
        line_color = '#047857'
        marker_color = '#b45309'
        bg_edge = '#ffffff'
        text_color = '#0f172a'
        legend_face = '#f8fafc'
        legend_edge = '#cbd5e1'
    else:
        line_color = '#10b981'
        marker_color = '#f59e0b'
        bg_edge = '#0b0f19'
        text_color = '#ffffff'
        legend_face = '#0f172a'
        legend_edge = '#1e293b'
        
    # Draw line with error bars
    plt.errorbar(
        n_vals, means, yerr=stds, 
        fmt='-s', color=line_color, ecolor=line_color, elinewidth=1.5, capsize=4,
        markerfacecolor=marker_color, markeredgecolor=bg_edge, markersize=8, markeredgewidth=1,
        lw=2.5, label='NCM Classifier Accuracy'
    )
    
    plt.xticks(n_vals)
    plt.ylim([min(means) - max(stds) - 5, 105])
    
    plt.title("Model Capacity vs. Target Classes (N-Way)", fontsize=16, fontweight='bold', pad=20, color=text_color)
    plt.xlabel("Number of Target Classes (Ways)", fontsize=12, labelpad=10)
    plt.ylabel("Accuracy (%)", fontsize=12, labelpad=10)
    
    legend = plt.legend(facecolor=legend_face, edgecolor=legend_edge, loc='lower left', framealpha=0.85)
    for text in legend.get_texts():
        text.set_color(text_color)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved N-way plot to {save_path}")

def plot_training_loss(loss_data, save_path, theme='light'):
    if not loss_data or not loss_data.get('epochs'):
        print("No training loss data found. Skipping.")
        return
        
    print("Generating Training Loss Curve...")
    plt.figure(figsize=(9, 5.5), dpi=300)
    apply_premium_style(theme)
    
    epochs = loss_data['epochs']
    loss = loss_data['loss']
    
    if theme == 'light':
        primary_color = '#6d28d9'
        fill_alpha = 0.08
        text_color = '#0f172a'
        legend_face = '#f8fafc'
        legend_edge = '#cbd5e1'
    else:
        primary_color = '#8b5cf6'
        fill_alpha = 0.10
        text_color = '#ffffff'
        legend_face = '#0f172a'
        legend_edge = '#1e293b'
        
    # Smooth plotting
    plt.plot(epochs, loss, color=primary_color, lw=3, label='Metric Triplet Loss')
    plt.fill_between(epochs, loss, color=primary_color, alpha=fill_alpha)
    
    plt.title("Metric Learning Episodic Training Gradient", fontsize=16, fontweight='bold', pad=20, color=text_color)
    plt.xlabel("Training Epochs", fontsize=12, labelpad=10)
    plt.ylabel("Triplet Loss Metric", fontsize=12, labelpad=10)
    
    legend = plt.legend(facecolor=legend_face, edgecolor=legend_edge, loc='upper right', framealpha=0.85)
    for text in legend.get_texts():
        text.set_color(text_color)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Training Loss plot to {save_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Render evaluation plots for a specific model')
    parser.add_argument('--model_dir', type=str, default='results/dscnnlln',
                        help='Directory containing calculated_plot_data.json')
    parser.add_argument('--theme', type=str, default='light', choices=['light', 'dark'],
                        help='Theme of the generated plots (light or dark)')
    args = parser.parse_args()

    data_path = os.path.join(args.model_dir, "calculated_plot_data.json")
    plot_dir = os.path.join(args.model_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    try:
        data = load_calculated_data(data_path)
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Please run calculate_plot_data.py --model_dir {args.model_dir} first.")
        return
        
    theme = args.theme
    plot_tsne(data['tsne'], os.path.join(plot_dir, "tsne_embeddings.png"), theme)
    plot_roc(data['roc_det'], os.path.join(plot_dir, "roc_curve.png"), theme)
    plot_det(data['roc_det'], os.path.join(plot_dir, "det_curve.png"), theme)
    plot_kshot(data['kshot'], os.path.join(plot_dir, "accuracy_vs_kshot.png"), theme)
    plot_nway(data['nway'], os.path.join(plot_dir, "accuracy_vs_nway.png"), theme)
    plot_training_loss(data['loss'], os.path.join(plot_dir, "training_loss.png"), theme)
    
    print(f"\nAll 6 high-resolution scientific plots generated under '{plot_dir}' using '{theme}' theme.")

if __name__ == '__main__':
    main()

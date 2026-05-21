import os
import sys
import json
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm

# Add local path to import modules
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./KWSFSL'))

from KWSFSL.utils import filter_opt
from KWSFSL.classifiers.NCM import NearestClassMean

def load_environment(opt_path, model_path, override_data_dir=None):
    print("Loading configuration...")
    with open(opt_path, 'r') as fp:
        opt = json.load(fp)
    
    # Force cpu if cuda not available
    device = "cuda" if torch.cuda.is_available() and opt.get('data.cuda', True) else "cpu"
    opt['data.cuda'] = (device == "cuda")
    print(f"Using device: {device}")
    
    print("Loading model...")
    enc_model = torch.load(model_path, map_location=device, weights_only=False)
    enc_model.eval()
    
    # Load dataset
    print("Loading dataset...")
    speech_args = filter_opt(opt, 'speech')
    
    # Determine the dataset directory: if override_data_dir exists use it,
    # otherwise use the opt path, falling back to './data' if that does not exist.
    actual_data_dir = override_data_dir if override_data_dir else opt.get('speech.default_datadir', './data')
    if not os.path.exists(actual_data_dir):
        print(f"Configured dataset path '{actual_data_dir}' not found, falling back to './data'")
        actual_data_dir = './data'
        
    # Use exact MSWCFolders dataset loader
    from KWSFSL.data.MSWCFoldersData import MSWCFoldersDataset
    
    pos_task = 'junior:lay:material:mixed:thomas:exist:fruit:girls:boys:break:educated'
    neg_task = 'offer:paid:increased:laughed:length:mayor:michael:traffic:flame:boss'
    
    ds = MSWCFoldersDataset(actual_data_dir, pos_task, opt['data.cuda'], speech_args)
    ds_neg = MSWCFoldersDataset(actual_data_dir, neg_task, opt['data.cuda'], speech_args)
    
    return opt, enc_model, ds, ds_neg, pos_task, neg_task

def run_tsne(enc_model, ds, ds_neg, pos_task, neg_task, device):
    print("Running t-SNE calculations...")
    
    # Pick 5 positive classes and 5 negative classes
    pos_classes = pos_task.split(':')[:5]
    neg_classes = neg_task.split(':')[:5]
    
    all_embeddings = []
    all_labels = []
    is_known = []
    
    # Limit number of samples per class to avoid overhead but get clear clusters
    num_samples = 30
    
    with torch.no_grad():
        # Process positive (known) classes
        for word in pos_classes:
            samples = ds.data_set['testing'].get(word, [])[:num_samples]
            if not samples:
                continue
            ts_ds = ds.get_transform_dataset_from_list(samples)
            loader = torch.utils.data.DataLoader(ts_ds, batch_size=len(samples), shuffle=False)
            for batch in loader:
                x = batch['data'].to(device)
                embeddings = enc_model.get_embeddings(x).cpu().numpy()
                all_embeddings.append(embeddings)
                all_labels.extend([word] * len(embeddings))
                is_known.extend([True] * len(embeddings))
                
        # Process negative (unknown) classes
        for word in neg_classes:
            samples = ds_neg.data_set['testing'].get(word, [])[:num_samples]
            if not samples:
                continue
            ts_ds = ds_neg.get_transform_dataset_from_list(samples)
            loader = torch.utils.data.DataLoader(ts_ds, batch_size=len(samples), shuffle=False)
            for batch in loader:
                x = batch['data'].to(device)
                embeddings = enc_model.get_embeddings(x).cpu().numpy()
                all_embeddings.append(embeddings)
                all_labels.extend(["_unknown_" + f" ({word})"] * len(embeddings))
                is_known.extend([False] * len(embeddings))
                
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Fit t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=15)
    coords = tsne.fit_transform(all_embeddings)
    
    tsne_data = {
        'coords_x': coords[:, 0].tolist(),
        'coords_y': coords[:, 1].tolist(),
        'labels': all_labels,
        'is_known': is_known
    }
    return tsne_data

def run_roc_det(enc_model, ds, ds_neg, opt, device):
    print("Running ROC / DET evaluations...")
    classifier = NearestClassMean(backbone=enc_model, cuda=opt['data.cuda'])
    
    # Parameters
    n_way = 5
    n_support = 5
    n_episodes = 25
    
    pos_words = [w for w in ds.words_list if w not in ['_silence_', '_unknown_']]
    neg_words = [w for w in ds_neg.words_list if w not in ['_silence_', '_unknown_']]
    
    all_y_true = []
    all_y_scores = []
    
    for ep in tqdm(range(n_episodes), desc="Episodes"):
        # Select N target words for this episode
        target_words = np.random.choice(pos_words, n_way, replace=False).tolist()
        
        # Fit classifier with support samples
        support_samples_list = []
        for word in target_words:
            # support from training set
            samples = ds.data_set['training'].get(word, [])
            selected = np.random.choice(samples, n_support, replace=False).tolist()
            support_samples_list.append(selected)
            
        # Dataload and compute prototypes
        flat_support = [item for sublist in support_samples_list for item in sublist]
        ts_ds_support = ds.get_transform_dataset_from_list(flat_support)
        support_loader = torch.utils.data.DataLoader(ts_ds_support, batch_size=len(flat_support), shuffle=False)
        
        for batch in support_loader:
            x_support = batch['data'].to(device)
            # labels of support
            labels_support = batch['label']
            
            # reshape x_support to (n_way, n_support, ...)
            emb_shape = x_support.shape[1:]
            x_support_reshaped = x_support.view(n_way, n_support, *emb_shape)
            
            classifier.fit_batch_offline(x_support_reshaped.cpu(), target_words)
            
        # Get query loader for positive queries
        pos_query_samples = []
        for word in target_words:
            samples = ds.data_set['testing'].get(word, [])
            selected = np.random.choice(samples, min(10, len(samples)), replace=False).tolist()
            pos_query_samples.extend(selected)
            
        # Load and score positive queries
        if pos_query_samples:
            ts_ds_pos = ds.get_transform_dataset_from_list(pos_query_samples)
            pos_loader = torch.utils.data.DataLoader(ts_ds_pos, batch_size=len(pos_query_samples), shuffle=False)
            for batch in pos_loader:
                x_query = batch['data'].to(device)
                scores, _ = classifier.evaluate_batch(x_query, batch['label'], return_probas=False)
                # Compute maximum distance score over target classes
                max_score, _ = scores.max(dim=1)
                all_y_scores.extend(max_score.tolist())
                all_y_true.extend([1] * len(max_score)) # 1 for positive/known
                
        # Get query loader for negative (unknown) queries
        neg_query_samples = []
        # Sample random negative words
        selected_neg_words = np.random.choice(neg_words, min(n_way, len(neg_words)), replace=False)
        for word in selected_neg_words:
            samples = ds_neg.data_set['testing'].get(word, [])
            selected = np.random.choice(samples, min(10, len(samples)), replace=False).tolist()
            neg_query_samples.extend(selected)
            
        if neg_query_samples:
            ts_ds_neg = ds_neg.get_transform_dataset_from_list(neg_query_samples)
            neg_loader = torch.utils.data.DataLoader(ts_ds_neg, batch_size=len(neg_query_samples), shuffle=False)
            for batch in neg_loader:
                x_query = batch['data'].to(device)
                # Fit dummy labels to evaluate
                scores, _ = classifier.evaluate_batch(x_query, batch['label'], return_probas=False)
                max_score, _ = scores.max(dim=1)
                all_y_scores.extend(max_score.tolist())
                all_y_true.extend([-1] * len(max_score)) # -1 for negative/unknown
                
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(all_y_true, all_y_scores, pos_label=1)
    # Clean up any np.inf in thresholds to avoid Infinity in JSON
    thresholds = np.where(np.isinf(thresholds), max(all_y_scores) + 1.0, thresholds)
    auroc = roc_auc_score(all_y_true, all_y_scores)
    print(f"Calculated AUC-ROC: {auroc:.4f}")
    
    # Calculate FAR and FRR
    # FAR (False Acceptance Rate) = percentage of negative queries accepted as known
    # FRR (False Rejection Rate) = percentage of positive queries rejected as unknown
    # At threshold T:
    # Accept if max_score >= T, reject if max_score < T
    far = []
    frr = []
    
    # Evaluate across a grid of thresholds
    sorted_scores = sorted(all_y_scores)
    # create 100 threshold grid points
    grid_thresholds = np.percentile(sorted_scores, np.linspace(0, 100, 100))
    
    pos_scores = np.array([score for score, label in zip(all_y_scores, all_y_true) if label == 1])
    neg_scores = np.array([score for score, label in zip(all_y_scores, all_y_true) if label == -1])
    
    for t in grid_thresholds:
        # FAR = False Acceptance / Total Negatives
        far_val = (neg_scores >= t).sum() / len(neg_scores) if len(neg_scores) > 0 else 0
        # FRR = False Rejection / Total Positives
        frr_val = (pos_scores < t).sum() / len(pos_scores) if len(pos_scores) > 0 else 0
        far.append(far_val)
        frr.append(frr_val)
        
    roc_det_data = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'auc': float(auroc),
        'thresholds': thresholds.tolist(),
        'far': far,
        'frr': frr,
        'grid_thresholds': grid_thresholds.tolist()
    }
    return roc_det_data

def run_kshot_scaling(enc_model, ds, opt, device):
    print("Evaluating K-shot scalability...")
    n_way = 5
    n_episodes = 20
    k_options = [1, 3, 5, 10]
    
    pos_words = [w for w in ds.words_list if w not in ['_silence_', '_unknown_']]
    results = {}
    
    for k in k_options:
        accuracies = []
        classifier = NearestClassMean(backbone=enc_model, cuda=opt['data.cuda'])
        
        for ep in range(n_episodes):
            target_words = np.random.choice(pos_words, n_way, replace=False).tolist()
            
            # Support
            support_samples = []
            for word in target_words:
                samples = ds.data_set['training'].get(word, [])
                selected = np.random.choice(samples, k, replace=False).tolist()
                support_samples.extend(selected)
                
            ts_ds_support = ds.get_transform_dataset_from_list(support_samples)
            support_loader = torch.utils.data.DataLoader(ts_ds_support, batch_size=len(support_samples), shuffle=False)
            for batch in support_loader:
                x_support = batch['data'].to(device)
                x_support_reshaped = x_support.view(n_way, k, *x_support.shape[1:])
                classifier.fit_batch_offline(x_support_reshaped.cpu(), target_words)
                
            # Query
            query_samples = []
            for word in target_words:
                samples = ds.data_set['testing'].get(word, [])
                selected = np.random.choice(samples, min(10, len(samples)), replace=False).tolist()
                query_samples.extend(selected)
                
            if query_samples:
                ts_ds_query = ds.get_transform_dataset_from_list(query_samples)
                query_loader = torch.utils.data.DataLoader(ts_ds_query, batch_size=len(query_samples), shuffle=False)
                for batch in query_loader:
                    x_query = batch['data'].to(device)
                    scores, targets = classifier.evaluate_batch(x_query, batch['label'], return_probas=False)
                    _, preds = scores.max(1)
                    acc = (preds.cpu() == targets.cpu().squeeze()).float().mean().item()
                    accuracies.append(acc)
                    
        results[str(k)] = {
            'mean': float(np.mean(accuracies)),
            'std': float(np.std(accuracies))
        }
    return results

def run_nway_scaling(enc_model, ds, opt, device):
    print("Evaluating N-way capacity...")
    n_support = 5
    n_episodes = 20
    n_options = [2, 5, 8, 10]
    
    pos_words = [w for w in ds.words_list if w not in ['_silence_', '_unknown_']]
    results = {}
    
    for n in n_options:
        accuracies = []
        classifier = NearestClassMean(backbone=enc_model, cuda=opt['data.cuda'])
        
        for ep in range(n_episodes):
            target_words = np.random.choice(pos_words, n, replace=False).tolist()
            
            # Support
            support_samples = []
            for word in target_words:
                samples = ds.data_set['training'].get(word, [])
                selected = np.random.choice(samples, n_support, replace=False).tolist()
                support_samples.extend(selected)
                
            ts_ds_support = ds.get_transform_dataset_from_list(support_samples)
            support_loader = torch.utils.data.DataLoader(ts_ds_support, batch_size=len(support_samples), shuffle=False)
            for batch in support_loader:
                x_support = batch['data'].to(device)
                x_support_reshaped = x_support.view(n, n_support, *x_support.shape[1:])
                classifier.fit_batch_offline(x_support_reshaped.cpu(), target_words)
                
            # Query
            query_samples = []
            for word in target_words:
                samples = ds.data_set['testing'].get(word, [])
                selected = np.random.choice(samples, min(10, len(samples)), replace=False).tolist()
                query_samples.extend(selected)
                
            if query_samples:
                ts_ds_query = ds.get_transform_dataset_from_list(query_samples)
                query_loader = torch.utils.data.DataLoader(ts_ds_query, batch_size=len(query_samples), shuffle=False)
                for batch in query_loader:
                    x_query = batch['data'].to(device)
                    scores, targets = classifier.evaluate_batch(x_query, batch['label'], return_probas=False)
                    _, preds = scores.max(1)
                    acc = (preds.cpu() == targets.cpu().squeeze()).float().mean().item()
                    accuracies.append(acc)
                    
        results[str(n)] = {
            'mean': float(np.mean(accuracies)),
            'std': float(np.std(accuracies))
        }
    return results

def load_training_loss(trace_path):
    print("Loading training loss trace...")
    epochs = []
    loss = []
    if os.path.exists(trace_path):
        with open(trace_path, 'r') as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    epochs.append(data['epoch'])
                    loss.append(data['train']['loss'])
                except Exception:
                    pass
    return {'epochs': epochs, 'loss': loss}

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Calculate plot data for a specific model')
    parser.add_argument('--model_dir', type=str, default='results/dscnnlln',
                        help='Directory containing opt.json, best_model.pt, and trace.txt')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override path to the MSWC dataset directory (defaults to opt.json path or ./data)')
    args = parser.parse_args()

    opt_path = os.path.join(args.model_dir, "opt.json")
    model_path = os.path.join(args.model_dir, "best_model.pt")
    trace_path = os.path.join(args.model_dir, "trace.txt")
    
    # Load env with potential data directory override
    opt, enc_model, ds, ds_neg, pos_task, neg_task = load_environment(opt_path, model_path, override_data_dir=args.data_dir)
    device = "cuda" if opt['data.cuda'] else "cpu"
    
    # 1. Run t-SNE
    tsne_data = run_tsne(enc_model, ds, ds_neg, pos_task, neg_task, device)
    
    # 2. Run ROC & DET
    roc_det_data = run_roc_det(enc_model, ds, ds_neg, opt, device)
    
    # 3. Run K-shot scaling
    kshot_data = run_kshot_scaling(enc_model, ds, opt, device)
    
    # 4. Run N-way scaling
    nway_data = run_nway_scaling(enc_model, ds, opt, device)
    
    # 5. Load training trace loss
    loss_data = load_training_loss(trace_path)
    
    # Save combined output
    output_data = {
        'tsne': tsne_data,
        'roc_det': roc_det_data,
        'kshot': kshot_data,
        'nway': nway_data,
        'loss': loss_data
    }
    
    os.makedirs(args.model_dir, exist_ok=True)
    out_file = os.path.join(args.model_dir, "calculated_plot_data.json")
    print(f"Saving all metrics to {out_file}...")
    with open(out_file, 'w') as fp:
        json.dump(output_data, fp, indent=2)
    print("Data calculation successfully completed!")

if __name__ == '__main__':
    main()

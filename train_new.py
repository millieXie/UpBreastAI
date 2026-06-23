import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")
import os, random, numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm import tqdm
from collections import OrderedDict
from sklearn import metrics
from options.Options import Options_x
from dataset.dataset_train import Litss_Anomaly_DataSet
from setting import util
from Model.network_v9fusion_s import MemAE_SVDDHybrid


def build_train_val(root, label_xlsx, size, train_ratio=0.8, seed=444):
    full_set = Litss_Anomaly_DataSet(
        root,
        label_csv=label_xlsx,
        size=size,
        mode='all',
        train_ratio=1.0,
        seed=seed,
    )

    label_to_indices = {}
    for idx, (_, label) in enumerate(full_set.filename):
        label_to_indices.setdefault(int(label), []).append(idx)

    rng = random.Random(seed)
    train_indices, val_indices = [], []
    for label, indices in label_to_indices.items():
        indices = indices[:]
        rng.shuffle(indices)
        split_idx = int(len(indices) * train_ratio)
        train_indices.extend(indices[:split_idx])
        val_indices.extend(indices[split_idx:])

    train_set = Litss_Anomaly_DataSet(
        root,
        label_csv=label_xlsx,
        size=size,
        mode='all',
        train_ratio=1.0,
        seed=seed,
    )
    val_set = Litss_Anomaly_DataSet(
        root,
        label_csv=label_xlsx,
        size=size,
        mode='all',
        train_ratio=1.0,
        seed=seed,
    )

    train_set.filename = [full_set.filename[i] for i in train_indices if int(full_set.filename[i][1]) == 0]
    val_set.filename = [full_set.filename[i] for i in val_indices]
    train_set.mode = 'train'
    val_set.mode = 'all'

    train_pos = sum(1 for i in train_indices if int(full_set.filename[i][1]) != 0)
    val_neg = sum(1 for _, label in val_set.filename if int(label) == 0)
    val_pos = len(val_set.filename) - val_neg
    print(f"Split from one xlsx: train split={len(train_indices)} samples "
          f"({len(train_set.filename)} benign used, {train_pos} positive unused)")
    print(f"Validation split: {val_neg} benign and {val_pos} positive samples")
    return train_set, val_set


def setup_seed(seed):
    torch.manual_seed(seed);
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed);
    random.seed(seed)
    torch.backends.cudnn.deterministic = True;
    torch.backends.cudnn.benchmark = False



def youden(tpr, fpr, thresholds):
    tnr = 1 - fpr;
    youden_idx = tpr + tnr - 1;
    idx = np.argmax(youden_idx)
    return youden_idx[idx], thresholds[idx]



def train(train_loader, epoch, model, optimizer, device, opt, lambda_recon, lambda_diversity, lambda_svdd):
    model.train()
    total_loss = total_recon = total_svdd = total_div = 0
    nu = getattr(opt, 'nu', 0.3)
    all_distances = []
    with tqdm(train_loader, desc=f"Training Epoch {epoch}", unit="batch") as pbar:
        for batch in pbar:
            x, labels = batch
            x = x.to(device).float()
            labels = labels.to(device)
            mask = labels == 0
            if mask.sum() == 0: continue
            x_benign = x[mask]
            x_rec, att_weights, z, distances = model(x_benign)
            recon_loss = F.l1_loss(x_rec, x_benign)
            div_loss = model.memory_diversity_loss()
            R_sq = model.R ** 2
            zeros = torch.zeros_like(distances)
            svdd_loss = R_sq + (1.0 / nu) * torch.mean(torch.max(zeros, distances - R_sq))
            loss = lambda_recon * recon_loss + lambda_diversity * div_loss + lambda_svdd * svdd_loss
            optimizer.zero_grad(set_to_none=True);
            loss.backward();
            optimizer.step()
            total_loss += loss.item();
            total_recon += recon_loss.item()
            total_svdd += svdd_loss.item();
            total_div += div_loss.item()
            all_distances.append(distances.detach())
            pbar.set_postfix({"Total": f"{total_loss / (pbar.n + 1):.4f}", "Recon": f"{total_recon / (pbar.n + 1):.4f}",
                              "SVDD": f"{total_svdd / (pbar.n + 1):.4f}", "Div": f"{total_div / (pbar.n + 1):.4f}"})
    if all_distances:
        all_dist = torch.cat(all_distances, dim=0)
        model.R = torch.quantile(all_dist, 1.0 - nu).to(device)
    return OrderedDict({'Loss': total_loss / len(train_loader), 'Recon': total_recon / len(train_loader),
                        'SVDD': total_svdd / len(train_loader), 'Diversity': total_div / len(train_loader),
                        'LR': optimizer.param_groups[0]['lr']})


def evaluate(loader, model, device, desc="Evaluating"):
    model.eval()
    recon_errors, svdd_dists, labels_all = [], [], []
    with torch.no_grad():
        for x, labels in tqdm(loader, desc=desc, unit="batch"):
            x = x.to(device).float();
            labels = labels.to(device)
            x_rec, att, z, distances = model(x)
            recon = F.mse_loss(x_rec, x, reduction='none').reshape(x.shape[0], -1).mean(dim=1)
            recon_errors.extend(recon.cpu().numpy());
            svdd_dists.extend(distances.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    recon_errors = np.array(recon_errors);
    svdd_dists = np.array(svdd_dists);
    all_labels = np.array(labels_all)

    def min_max_norm(arr):
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    recon_norm = min_max_norm(recon_errors)
    svdd_norm = min_max_norm(svdd_dists)

    if len(np.unique(all_labels)) > 1:
        norm_mask = all_labels == 0
        abn_mask = all_labels == 1
        print(f"[{desc}] Normal samples: {norm_mask.sum()}, Abnormal: {abn_mask.sum()}")
        recon_norm_mean = recon_errors[norm_mask].mean() if norm_mask.sum() > 0 else np.nan
        recon_abn_mean = recon_errors[abn_mask].mean() if abn_mask.sum() > 0 else np.nan
        recon_norm_std = recon_errors[norm_mask].std() if norm_mask.sum() > 0 else np.nan
        recon_abn_std = recon_errors[abn_mask].std() if abn_mask.sum() > 0 else np.nan
        print(
            f"  Recon (raw)   - Normal: {recon_norm_mean:.6f}±{recon_norm_std:.6f}, Abnormal: {recon_abn_mean:.6f}±{recon_abn_std:.6f}")
        recon_norm_norm_mean = recon_norm[norm_mask].mean() if norm_mask.sum() > 0 else np.nan
        recon_norm_abn_mean = recon_norm[abn_mask].mean() if abn_mask.sum() > 0 else np.nan
        print(f"  Recon (norm)  - Normal: {recon_norm_norm_mean:.6f}, Abnormal: {recon_norm_abn_mean:.6f}")
        svdd_norm_mean = svdd_dists[norm_mask].mean() if norm_mask.sum() > 0 else np.nan
        svdd_abn_mean = svdd_dists[abn_mask].mean() if abn_mask.sum() > 0 else np.nan
        svdd_norm_std = svdd_dists[norm_mask].std() if norm_mask.sum() > 0 else np.nan
        svdd_abn_std = svdd_dists[abn_mask].std() if abn_mask.sum() > 0 else np.nan
        print(
            f"  SVDD (raw)    - Normal: {svdd_norm_mean:.6f}±{svdd_norm_std:.6f}, Abnormal: {svdd_abn_mean:.6f}±{svdd_abn_std:.6f}")
        svdd_norm_norm_mean = svdd_norm[norm_mask].mean() if norm_mask.sum() > 0 else np.nan
        svdd_norm_abn_mean = svdd_norm[abn_mask].mean() if abn_mask.sum() > 0 else np.nan
        print(f"  SVDD (norm)   - Normal: {svdd_norm_norm_mean:.6f}, Abnormal: {svdd_norm_abn_mean:.6f}")
    else:
        print(f"[{desc}] Only one class present in labels.")


    alpha_s = 0.5
    scores = alpha_s * recon_norm + (1 - alpha_s) * svdd_norm
    if len(np.unique(all_labels)) > 1:
        auc = metrics.roc_auc_score(all_labels, scores)
        fpr, tpr, thres = metrics.roc_curve(all_labels, scores, pos_label=1)
    else:
        auc, fpr, tpr, thres = 0.5, np.array([0, 1]), np.array([0, 1]), np.array([0, 1])

    # 返回额外字段用于绘图
    return {
        'AUC': auc,
        'Scores': scores,
        'Labels': all_labels,
        'FPR': fpr,
        'TPR': tpr,
        'Thresholds': thres,
        'recon_raw': recon_errors,
        'svdd_raw': svdd_dists,
        'recon_norm': recon_norm,
        'svdd_norm': svdd_norm
    }


# --------------------- 内存统计并返回指标 ---------------------
def print_memory_stats(model, loader, device, num_samples=5, k=3):
    model.eval()
    att_weights_all = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device).float()
            _, att, _, _ = model(x)
            att_weights_all.append(att.cpu())
            if len(att_weights_all) * loader.batch_size >= num_samples: break
    if not att_weights_all: return 0, 0, 0
    att_weights_all = torch.cat(att_weights_all, dim=0)[:num_samples]
    for i in range(min(num_samples, att_weights_all.size(0))):
        top_vals, top_idx = torch.topk(att_weights_all[i], k=k)
        print(f"Sample {i}: top-{k} indices = {top_idx.tolist()}, weights = {[f'{v:.4f}' for v in top_vals.tolist()]}")
    threshold = 1e-3
    active_slots = (att_weights_all > threshold).sum(dim=1).float().mean().item()
    entropy = -(att_weights_all * torch.log(att_weights_all + 1e-8)).sum(dim=1).mean().item()
    max_weight = att_weights_all.max(dim=1)[0].mean().item()
    print(f"[Memory Stats] Active slots: {active_slots:.2f}, Entropy: {entropy:.4f}, Max weight: {max_weight:.4f}")
    return active_slots, entropy, max_weight


def adjust_hyperparams(active_slots, entropy, max_weight, cur_lambda, cur_temp,
                       target_active=(20, 50), target_entropy=(2.0, 3.5), target_max_weight=(0.1, 0.4)):
    new_lambda = cur_lambda
    new_temp = cur_temp
    if active_slots > target_active[1]:
        new_lambda *= 1.2
    elif active_slots < target_active[0]:
        new_lambda *= 0.8
    if entropy > target_entropy[1]:
        new_lambda *= 1.1
    elif entropy < target_entropy[0]:
        new_lambda *= 0.9
    new_lambda = np.clip(new_lambda, 1e-6, 0.1)
    if max_weight < target_max_weight[0]:
        new_temp *= 0.95
    elif max_weight > target_max_weight[1]:
        new_temp *= 1.05
    if entropy > target_entropy[1]:
        new_temp *= 0.97
    elif entropy < target_entropy[0]:
        new_temp *= 1.03
    new_temp = np.clip(new_temp, 0.01, 0.2)
    return new_lambda, new_temp


def main():
    setup_seed(444)
    opt = Options_x().parse()
    device = torch.device(f'cuda:{opt.gpu_ids[0]}' if torch.cuda.is_available() and opt.gpu_ids else 'cpu')
    label_xlsx = "/input your train data.xlsx"
    train_ratio = 0.8
    num_workers = 4
    train_set, val_set = build_train_val(
        opt.datapath,
        label_xlsx,
        opt.patch_size,
        train_ratio=train_ratio,
        seed=444,
    )
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=num_workers)

    model = MemAE_SVDDHybrid(in_channels=3, memory_dim=128, memory_size=300, temperature=0.05).to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    fold_dir = os.path.join(opt.checkpoints_dir, opt.task_name)
    util.mkdir(fold_dir)
    latest_path = os.path.join(fold_dir, 'latest_model.pth')
    best_path = os.path.join(fold_dir, 'best_model.pth')

    lambda_recon = 10.0
    lambda_diversity = 1e-3
    lambda_svdd = 0.1
    temperature = 0.05
    model.temperature = temperature

    start_epoch = 1
    best_val_auc = 0.0
    patience_counter = 0
    max_patience = 5

    if not os.path.exists(latest_path):
        model.initialize_center(train_loader, device)
    else:
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 1)
        best_val_auc = ckpt.get('best_val_auc', ckpt.get('best_avg_auc', 0.0))
        patience_counter = ckpt.get('patience_counter', 0)
        print(f"✅ Resume epoch {start_epoch}, best_val_auc={best_val_auc:.4f}")

    for epoch in range(start_epoch, opt.epoch + 1):
        print(f"\n===== Epoch {epoch}/{opt.epoch} =====")
        train_log = train(train_loader, epoch, model, optimizer, device, opt,
                          lambda_recon, lambda_diversity, lambda_svdd)

        val_result = evaluate(val_loader, model, device, desc="Val")
        val_auc = val_result['AUC']
        active_slots, entropy, max_weight = print_memory_stats(model, val_loader, device)


        if epoch >= 2:
            new_div, new_temp = adjust_hyperparams(active_slots, entropy, max_weight, lambda_diversity, temperature)
            if new_div != lambda_diversity or new_temp != temperature:
                lambda_diversity = new_div
                temperature = new_temp
                model.temperature = temperature
                print(f"🔥 Adjusted: lambda_div={lambda_diversity:.2e}, temperature={temperature:.3f}")

        print(f"[Val Epoch {epoch}] AUC={val_auc:.4f}")
        print(
            f"Train Loss: {train_log['Loss']:.4f}, Recon: {train_log['Recon']:.4f}, SVDD: {train_log['SVDD']:.4f}, Div: {train_log['Diversity']:.4f}")


        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
            'best_val_auc': best_val_auc,
            'patience_counter': patience_counter,
            'temperature': model.temperature,
            'R': model.R,
        }, latest_path)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'model': model.state_dict(),
                'R': model.R,
                'temperature': model.temperature
            }, best_path)
            patience_counter = 0
            print(f"🎯 New Best Model | ValAUC={best_val_auc:.4f}")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement | counter={patience_counter}/{max_patience}")
            if patience_counter >= max_patience:
                print(f"🛑 Early stopping at epoch {epoch}")
                break

        scheduler.step(val_auc)

    print("🎯 Training finished.")


if __name__ == '__main__':
    main()

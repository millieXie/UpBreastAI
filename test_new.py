import os
import warnings
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu
from sklearn import metrics
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset_test import Litss_Anomaly_DataSet
from Model.network_v9fusion_s import MemAE_SVDDHybrid
from options.Options import Options_x


warnings.filterwarnings("ignore")



def wilson_ci(successes, total, z=1.96):
    if total <= 0:
        return np.nan, np.nan
    p = successes / total
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    margin = z * np.sqrt((p * (1 - p) / total) + (z ** 2 / (4 * total ** 2))) / denom
    return float(max(0, center - margin)), float(min(1, center + margin))


def bootstrap_ci_non_binomial(labels, scores, threshold, n_bootstrap=1000, seed=42):
    """Bootstrap 95% CI for AUC, F1, and MCC."""
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    scores = np.array(scores)
    n = len(labels)

    aucs, f1s, mccs = [], [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y = labels[idx]
        s = scores[idx]

        if len(np.unique(y)) > 1:
            aucs.append(metrics.roc_auc_score(y, s))

        pred = (s > threshold).astype(int)
        f1s.append(metrics.f1_score(y, pred, zero_division=0))
        mccs.append(metrics.matthews_corrcoef(y, pred))

    def percentile_ci(values):
        if len(values) == 0:
            return np.nan, np.nan
        return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))

    return {
        "AUC_CI": percentile_ci(aucs),
        "F1_CI": percentile_ci(f1s),
        "MCC_CI": percentile_ci(mccs),
    }


def metric_cis(results, labels, scores):
    """
    CI policy:
    - AUC/F1/MCC: bootstrap 95% CI
    - Sensitivity/Specificity/Accuracy/PPV/NPV: Wilson 95% CI
    """
    tn = results["TN"]
    fp = results["FP"]
    fn = results["FN"]
    tp = results["TP"]
    threshold = results["Best_Threshold"]

    cis = bootstrap_ci_non_binomial(labels, scores, threshold, n_bootstrap=1000)
    cis.update({
        "Sensitivity_CI": wilson_ci(tp, tp + fn),
        "Specificity_CI": wilson_ci(tn, tn + fp),
        "Accuracy_CI": wilson_ci(tp + tn, tp + tn + fp + fn),
        "PPV_CI": wilson_ci(tp, tp + fp),
        "NPV_CI": wilson_ci(tn, tn + fn),
    })
    return cis


def no_miss_surgery_reduction(labels, scores):
    """
    Calculate surgery reduction under no missed positives.

    Baseline: operate/treat everyone.
    Model: operate/treat score >= min positive score, so FN=0.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    positive_scores = scores[labels == 1]

    if len(positive_scores) == 0:
        return {
            "NoMiss_Threshold": np.nan,
            "NoMiss_Surgery_Rate": np.nan,
            "NoMiss_Surgery_Reduction": np.nan,
            "NoMiss_Negative_Spared_Rate": np.nan,
            "NoMiss_TN": 0,
            "NoMiss_FP": 0,
            "NoMiss_FN": 0,
            "NoMiss_TP": 0,
        }

    threshold = float(np.min(positive_scores))
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    n = len(labels)
    surgery_rate = (tp + fp) / (n + 1e-8)
    surgery_reduction = 1 - surgery_rate
    negative_spared_rate = tn / (tn + fp + 1e-8)

    return {
        "NoMiss_Threshold": threshold,
        "NoMiss_Surgery_Rate": surgery_rate,
        "NoMiss_Surgery_Reduction": surgery_reduction,
        "NoMiss_Negative_Spared_Rate": negative_spared_rate,
        "NoMiss_TN": int(tn),
        "NoMiss_FP": int(fp),
        "NoMiss_FN": int(fn),
        "NoMiss_TP": int(tp),
    }


def save_preds_csv(labels, scores, preds, filenames, save_dir, prefix="test_final_pred"):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame({
        "Filename": filenames,
        "Label": labels,
        "Score": scores,
        "Pred": preds,
    })
    csv_path = os.path.join(save_dir, f"{prefix}.csv")
    excel_path = os.path.join(save_dir, f"{prefix}.xlsx")
    df.to_csv(csv_path, index=False)
    df.to_excel(excel_path, index=False)
    print(f"Final predictions saved to {csv_path} and {excel_path}")


def save_roc_curve(fpr, tpr, auc_value, save_dir, prefix="test_roc"):
    os.makedirs(save_dir, exist_ok=True)
    pd.DataFrame({"FPR": fpr, "TPR": tpr, "AUC": auc_value}).to_csv(
        os.path.join(save_dir, f"{prefix}.csv"),
        index=False,
    )
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC={auc_value:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}.png"), dpi=300)
    plt.close()
    print(f"ROC & CSV saved to {save_dir}")


def save_scores_csv(labels, scores, save_dir, prefix="test_scores"):
    os.makedirs(save_dir, exist_ok=True)
    pd.DataFrame({"Label": labels, "Score": scores}).to_csv(
        os.path.join(save_dir, f"{prefix}.csv"),
        index=False,
    )
    print(f"Scores saved to {save_dir}/{prefix}.csv")


def save_score_distribution(
        normal_scores,
        anomaly_scores,
        auc_mean,
        auc_ci=None,
        p_value=None,
        save_dir=None,
        prefix="score_distribution",
):
    """Save score distribution plot in the same format as the provided code."""
    os.makedirs(save_dir, exist_ok=True)

    normal_scores = np.asarray(normal_scores, dtype=np.float32)
    anomaly_scores = np.asarray(anomaly_scores, dtype=np.float32)
    all_scores = np.concatenate([normal_scores, anomaly_scores])
    if len(normal_scores) == 0 or len(anomaly_scores) == 0 or len(all_scores) == 0:
        print("Skip score distribution plot: one class has no samples.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=1.2)
    plt.rcParams["font.family"] = "DejaVu Sans"

    iqr = np.percentile(normal_scores, 75) - np.percentile(normal_scores, 25)
    if iqr > 0:
        bin_width = 2 * iqr / (len(normal_scores) ** (1 / 3))
        bins = max(10, int((normal_scores.max() - normal_scores.min()) / (bin_width + 1e-8)))
    else:
        bins = 20

    fig = plt.figure(figsize=(9, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.15)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    sns.histplot(
        normal_scores,
        bins=bins,
        kde=True,
        stat="density",
        color="#2E86AB",
        alpha=0.6,
        label="Normal",
        ax=ax1,
        line_kws={"linewidth": 2, "linestyle": "-"},
    )
    sns.histplot(
        anomaly_scores,
        bins=bins,
        kde=True,
        stat="density",
        color="#D62828",
        alpha=0.6,
        label="Anomaly",
        ax=ax1,
        line_kws={"linewidth": 2, "linestyle": "--"},
    )

    if len(anomaly_scores) < 20:
        sns.rugplot(anomaly_scores, color="#D62828", height=0.05, alpha=0.8, ax=ax1, label="_nolegend_")

    ax1.set_ylabel("Density", fontsize=11)
    if auc_ci is not None:
        title = f"Score Distribution (AUC = {auc_mean:.3f}, 95% CI [{auc_ci[0]:.3f}-{auc_ci[1]:.3f}])"
    else:
        title = f"Score Distribution (AUC = {auc_mean:.4f})"
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, linestyle="--", alpha=0.4)

    df_box = pd.DataFrame({
        "Score": all_scores,
        "Type": ["Normal"] * len(normal_scores) + ["Anomaly"] * len(anomaly_scores),
    })
    sns.boxplot(
        x="Type",
        y="Score",
        data=df_box,
        ax=ax2,
        palette={"Normal": "#2c7fb8", "Anomaly": "#d73027"},
        width=0.6,
        fliersize=0,
        linewidth=1.2,
        boxprops=dict(alpha=0.7),
    )
    sns.stripplot(x="Type", y="Score", data=df_box, ax=ax2, color="black", alpha=0.4, size=2.5, jitter=0.15)

    means = [np.mean(normal_scores), np.mean(anomaly_scores)]
    ax2.scatter([0, 1], means, marker="D", color="black", s=40, zorder=5, label="Mean")
    ax2.set_ylabel("Score")
    ax2.set_xlabel("")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)
    y_min, y_max = ax2.get_ylim()
    ax2.set_ylim(y_min, y_max + (y_max - y_min) * 0.1)

    diff = anomaly_scores.mean() - normal_scores.mean()
    p_str = f"p = {p_value:.4f}" if p_value is not None else ""
    text_str = f"Delta mean = {diff:.3f}  |  N_norm={len(normal_scores)}  N_anom={len(anomaly_scores)}"
    if p_str:
        text_str += f"  |  {p_str}"

    ax2.text(
        0.5,
        -0.45,
        text_str,
        transform=ax2.transAxes,
        ha="center",
        fontsize=9,
        style="italic",
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"),
    )

    plt.tight_layout()
    output_path = os.path.join(save_dir, f"{prefix}.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Score distribution plot saved: {output_path}")


def youden(tpr, fpr, thresholds):
    tnr = 1 - fpr
    youden_index = tpr + tnr - 1
    idx = np.argmax(youden_index)
    return youden_index[idx], thresholds[idx]


def test(dataloader, model, device, alpha=0.5):
    """Run anomaly-detection model and calculate metrics."""
    model.eval()
    recon_list, svdd_list, labels_list, filename_list = [], [], [], []

    def min_max_norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    with torch.no_grad():
        for x, labels, filenames in tqdm(dataloader, desc="Testing", unit="batch"):
            x = x.to(device).float()
            labels = labels.to(device)
            x_rec, _, z, distances = model(x)

            recon_err = F.mse_loss(x_rec, x, reduction="none")
            recon_err = recon_err.view(recon_err.size(0), -1).mean(dim=1)

            recon_list.append(recon_err.cpu())
            svdd_list.append(distances.cpu())
            labels_list.append(labels.cpu())
            filename_list.extend(filenames)

    recon_arr = torch.cat(recon_list).numpy()
    svdd_arr = torch.cat(svdd_list).numpy()
    labels_arr = torch.cat(labels_list).numpy()
    filenames = np.array(filename_list)

    recon_arr = np.nan_to_num(recon_arr)
    svdd_arr = np.nan_to_num(svdd_arr)
    recon_norm = min_max_norm(recon_arr)
    svdd_norm = min_max_norm(svdd_arr)
    anomaly_scores = alpha * recon_norm + (1 - alpha) * svdd_norm

    if len(np.unique(labels_arr)) > 1:
        fpr, tpr, thresholds = metrics.roc_curve(labels_arr, anomaly_scores, pos_label=1)
        auc_value = metrics.auc(fpr, tpr)
        _, best_thr = youden(tpr, fpr, thresholds)
        preds = (anomaly_scores > best_thr).astype(int)
    else:
        preds = np.zeros_like(labels_arr)
        auc_value = 0.5
        best_thr = 0.5
        fpr = np.array([0, 1])
        tpr = np.array([0, 1])

    tn, fp, fn, tp = metrics.confusion_matrix(labels_arr, preds, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    se = tp / (tp + fn + 1e-8)
    sp = tn / (tn + fp + 1e-8)
    ppv = tp / (tp + fp + 1e-8)
    npv = tn / (tn + fn + 1e-8)
    f1 = metrics.f1_score(labels_arr, preds, zero_division=0)
    mcc = metrics.matthews_corrcoef(labels_arr, preds)

    print(f"[Test] alpha={alpha:.2f}, AUC={auc_value:.4f}, Thr={best_thr:.4f}, "
          f"ACC={acc:.4f}, SE={se:.4f}, SP={sp:.4f}, PPV={ppv:.4f}, NPV={npv:.4f}, "
          f"F1={f1:.4f}, MCC={mcc:.4f}")
    print(f"Mean Scores (Normal/Abnormal) = {anomaly_scores[labels_arr == 0].mean():.4f}/"
          f"{anomaly_scores[labels_arr == 1].mean():.4f}")

    results_dict = OrderedDict({
        "AUC": auc_value,
        "Best_Threshold": best_thr,
        "Accuracy": acc,
        "Sensitivity": se,
        "Specificity": sp,
        "PPV": ppv,
        "NPV": npv,
        "F1": f1,
        "MCC": mcc,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Alpha_used": alpha,
    })

    return (results_dict, labels_arr, anomaly_scores, fpr, tpr, preds,
            recon_arr, svdd_arr, filenames)


def main():
    opt = Options_x().parse()
    device = torch.device("cuda:" + str(opt.gpu_ids[0]) if torch.cuda.is_available() else "cpu")

    print("Loading test dataset...")
    test_set = Litss_Anomaly_DataSet(
        opt.datapath,
        label_csv="/input your test datapath",
        size=opt.patch_size,
        mode="all",
    )
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    fold_dir = os.path.join(r"/input your modelpth")
    best_model_path = os.path.join(fold_dir, "your_model(like best_model/latest_model).pth") #use your model.pth
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model not found: {best_model_path}")

    print(f"Loading model from {best_model_path}")
    model = MemAE_SVDDHybrid(in_channels=3, memory_dim=128, memory_size=300, temperature=0.05).to(device)
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if "R" in ckpt:
        model.R = ckpt["R"].to(device)
    else:
        print("Warning: R not found in checkpoint.")
    print("Model loaded.")

    alpha = 0.5
    (results, labels, scores, fpr, tpr, preds,
     recon_arr, svdd_arr, filenames) = test(test_loader, model, device, alpha=alpha)

    print("\nCalibrating scores to probabilities (Isotonic Regression)...")
    try:
        probs = IsotonicRegression(out_of_bounds="clip").fit_transform(scores, labels)
    except ValueError:
        probs = scores

    print("\nCalculating 95% confidence intervals...")
    ci = metric_cis(results, np.array(labels), np.array(scores))
    no_miss = no_miss_surgery_reduction(labels, scores)
    results.update(no_miss)

    print("\n========== 95% CONFIDENCE INTERVALS ==========")
    for key, value in ci.items():
        print(f"{key}: {value}")

    print("\n========== TEST RESULTS ==========")
    for key, value in results.items():
        print(f"{key}: {value}")

    result_dir = "./result"
    os.makedirs(result_dir, exist_ok=True)
    print(f"\nAll results will be saved to: {os.path.abspath(result_dir)}")

    metrics_data = {
        "Metric": ["AUC", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "F1", "MCC"],
        "Value": [
            results["AUC"],
            results["Accuracy"],
            results["Sensitivity"],
            results["Specificity"],
            results["PPV"],
            results["NPV"],
            results["F1"],
            results["MCC"],
        ],
        "CI_lower": [
            ci["AUC_CI"][0],
            ci["Accuracy_CI"][0],
            ci["Sensitivity_CI"][0],
            ci["Specificity_CI"][0],
            ci["PPV_CI"][0],
            ci["NPV_CI"][0],
            ci["F1_CI"][0],
            ci["MCC_CI"][0],
        ],
        "CI_upper": [
            ci["AUC_CI"][1],
            ci["Accuracy_CI"][1],
            ci["Sensitivity_CI"][1],
            ci["Specificity_CI"][1],
            ci["PPV_CI"][1],
            ci["NPV_CI"][1],
            ci["F1_CI"][1],
            ci["MCC_CI"][1],
        ],
    }
    df_metrics = pd.DataFrame(metrics_data)
    csv_metrics_path = os.path.join(result_dir, "metrics_with_ci.csv")
    df_metrics.to_csv(csv_metrics_path, index=False)
    print(f"Metrics with CI saved to {csv_metrics_path}")

    no_miss_path = os.path.join(result_dir, "no_miss_surgery_reduction.csv")
    pd.DataFrame([no_miss]).to_csv(no_miss_path, index=False)
    print(f"No-miss surgery reduction saved to {no_miss_path}")

    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    if len(normal_scores) > 0 and len(anomaly_scores) > 0:
        _, p_value = mannwhitneyu(anomaly_scores, normal_scores, alternative="greater")
    else:
        p_value = None
    save_score_distribution(
        normal_scores,
        anomaly_scores,
        results["AUC"],
        auc_ci=ci["AUC_CI"],
        p_value=p_value,
        save_dir=result_dir,
        prefix="score_distribution",
    )

    save_roc_curve(fpr, tpr, results["AUC"], result_dir, prefix="test_roc")
    save_scores_csv(labels, scores, result_dir, prefix="test_scores")
    save_preds_csv(labels, scores, preds, filenames, result_dir, prefix="test_final_pred")

    patient_ids = [os.path.splitext(os.path.basename(f))[0] for f in filenames]
    df_excel = pd.DataFrame({
        "patient_id": patient_ids,
        "label": labels,
        "prediction": preds,
        "reconstruction_error": recon_arr,
        "svdd_distance": svdd_arr,
        "score": scores,
    })
    excel_path = os.path.join(result_dir, "test_results.xlsx")
    df_excel.to_excel(excel_path, index=False)
    print(f"Detailed Excel results saved to {excel_path}")

    print("\nTesting finished.")


if __name__ == "__main__":
    main()

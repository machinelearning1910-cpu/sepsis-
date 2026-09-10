from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from methodology_training import BATCH_SIZE, FairLoRAMultiTaskModel, SepsisSequenceDataset, HOSPITALS, parameter_report

ROOT = Path(os.getenv("FAIRLORA_ROOT", "/content/drive/MyDrive/FairLoRA_Sepsis_PaperAligned"))
DATA_DIR = ROOT / "data"
META_DIR = ROOT / "metadata"
CHECKPOINT_DIR = ROOT / "checkpoints"
REPORT_DIR = ROOT / "reports"
PLOT_DIR = ROOT / "plots"
for p in [REPORT_DIR, PLOT_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def mount_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except Exception:
        pass


def make_loader(frame, sequence_path):
    workers = min(2, os.cpu_count() or 1)
    return DataLoader(
        SepsisSequenceDataset(frame, sequence_path),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0
    )


def local_predictions(model, frame, sequence_path, device):
    loader = make_loader(frame, sequence_path)
    model.eval()
    sepsis_true = []
    sepsis_prob = []
    recovery_true = []
    recovery_pred = []
    organ_true = []
    organ_prob = []
    organ_pred = []
    age_group = []
    gender = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            sepsis_logits, recovery_logits, organ_logits = model(x)
            sepsis_true.extend(batch["sepsis"].numpy().astype(int).tolist())
            sepsis_prob.extend(torch.sigmoid(sepsis_logits).view(-1).cpu().numpy().tolist())
            recovery_true.extend(batch["recovery"].numpy().astype(int).tolist())
            recovery_pred.extend(torch.argmax(recovery_logits, dim=1).cpu().numpy().astype(int).tolist())
            organ_true.extend(batch["organ"].numpy().astype(int).tolist())
            op = torch.sigmoid(organ_logits).cpu().numpy()
            organ_prob.extend(op.tolist())
            organ_pred.extend((op >= 0.5).astype(int).tolist())
            age_group.extend(batch["age_group"].numpy().astype(int).tolist())
            gender.extend(batch["gender"].numpy().astype(int).tolist())
    return {
        "sepsis_true": np.asarray(sepsis_true, dtype=np.int64),
        "sepsis_prob": np.asarray(sepsis_prob, dtype=np.float64),
        "recovery_true": np.asarray(recovery_true, dtype=np.int64),
        "recovery_pred": np.asarray(recovery_pred, dtype=np.int64),
        "organ_true": np.asarray(organ_true, dtype=np.int64),
        "organ_prob": np.asarray(organ_prob, dtype=np.float64),
        "organ_pred": np.asarray(organ_pred, dtype=np.int64),
        "age_group": np.asarray(age_group, dtype=np.int64),
        "gender": np.asarray(gender, dtype=np.int64)
    }


def binary_metrics(y, probability):
    prediction = (probability >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "auroc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else np.nan,
        "auprc": float(average_precision_score(y, probability)) if y.sum() > 0 else np.nan,
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def multiclass_metrics(y, prediction):
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, prediction, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1, 2]).astype(int).tolist()
    }



def multilabel_metrics(y, probability, labels):
    prediction = (probability >= 0.5).astype(np.int64)
    rows = []
    for i, label in enumerate(labels):
        yt = y[:, i]
        pp = probability[:, i]
        yp = prediction[:, i]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        rows.append({
            "organ": label,
            "accuracy": float(accuracy_score(yt, yp)),
            "auroc": float(roc_auc_score(yt, pp)) if len(np.unique(yt)) == 2 else np.nan,
            "auprc": float(average_precision_score(yt, pp)) if yt.sum() > 0 else np.nan,
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "specificity": float(tn / max(tn + fp, 1)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        })
    table = pd.DataFrame(rows)
    return table, float(table["auroc"].mean(skipna=True))


def multilabel_cooccurrence(y, prediction):
    return np.asarray(y, dtype=np.int64).T @ np.asarray(prediction, dtype=np.int64)

def group_fairness(y, probability, groups, names):
    prediction = (probability >= 0.5).astype(np.int64)
    rows = []
    for code, name in names.items():
        mask = groups == code
        if not np.any(mask):
            continue
        yt = y[mask]
        yp = prediction[mask]
        pp = probability[mask]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        rows.append({
            "group": name,
            "patients": int(mask.sum()),
            "auroc": float(roc_auc_score(yt, pp)) if len(np.unique(yt)) == 2 else np.nan,
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "positive_rate": float(yp.mean()),
            "tpr": float(tp / max(tp + fn, 1)),
            "fpr": float(fp / max(fp + tn, 1))
        })
    table = pd.DataFrame(rows)
    summary = {
        "DPD": float(table["positive_rate"].max() - table["positive_rate"].min()),
        "EOD": float(table["tpr"].max() - table["tpr"].min()),
        "FPR_Gap": float(table["fpr"].max() - table["fpr"].min()),
        "AUROC_Gap": float(table["auroc"].max() - table["auroc"].min())
    }
    return table, summary



def aggregate_binary_reports(reports):
    tn = sum(report["metrics"]["tn"] for report in reports)
    fp = sum(report["metrics"]["fp"] for report in reports)
    fn = sum(report["metrics"]["fn"] for report in reports)
    tp = sum(report["metrics"]["tp"] for report in reports)
    n = np.asarray([report["n"] for report in reports], dtype=np.float64)
    weights = n / n.sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "auroc": float(np.nansum(weights * np.asarray([report["metrics"]["auroc"] for report in reports]))),
        "auprc": float(np.nansum(weights * np.asarray([report["metrics"]["auprc"] for report in reports]))),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "specificity": float(tn / max(tn + fp, 1)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def save_confusion(matrix, labels, path, title):
    matrix = np.asarray(matrix)
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix)
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(int(matrix[i, j])), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_hospital_roc(local_results, path):
    plt.figure(figsize=(8, 6))
    for hospital, result in local_results.items():
        y = result["predictions"]["sepsis_true"]
        p = result["predictions"]["sepsis_prob"]
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        plt.plot(fpr, tpr, linewidth=2, label=f"Hospital {hospital} ({auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Cross-Hospital ROC Analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_training_curves():
    path = REPORT_DIR / "training_history.csv"
    if not path.exists():
        return
    history = pd.read_csv(path)
    plt.figure(figsize=(8, 5))
    plt.plot(history["round"], history["train_loss"], marker="o", label="Training Loss")
    plt.plot(history["round"], history["total_loss"], marker="s", label="Validation Loss")
    plt.xlabel("Federated Round")
    plt.ylabel("Loss")
    plt.title("Federated Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "training_validation_loss.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot(history["round"], history["accuracy"], marker="o", label="Accuracy")
    plt.plot(history["round"], history["auroc"], marker="s", label="AUROC")
    plt.plot(history["round"], history["f1"], marker="^", label="F1")
    plt.xlabel("Federated Round")
    plt.ylabel("Score")
    plt.title("Validation Performance During Federated Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "validation_performance.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    mount_drive()
    sequence_path = DATA_DIR / "sequences_24x37.npy"
    checkpoint_path = CHECKPOINT_DIR / "best_model.pt"
    validation_path = META_DIR / "hospital_assignment_validation.csv"
    if not sequence_path.exists() or not checkpoint_path.exists() or not validation_path.exists():
        raise FileNotFoundError("Run dataset_preprocessing.py and methodology_training.py first")
    validation = pd.read_csv(validation_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = FairLoRAMultiTaskModel(input_dim=37).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    local_results = {}
    binary_reports = []
    hospital_rows = []
    recovery_cm = np.zeros((3, 3), dtype=np.int64)
    all_organ_true = []
    all_organ_prob = []
    all_organ_pred = []
    all_y = []
    all_p = []
    all_age = []
    all_gender = []
    for hospital in HOSPITALS:
        frame = validation[validation["hospital"] == hospital].copy()
        predictions = local_predictions(model, frame, sequence_path, device)
        metrics = binary_metrics(predictions["sepsis_true"], predictions["sepsis_prob"])
        binary_reports.append({"hospital": hospital, "n": len(frame), "metrics": metrics})
        hospital_rows.append({"hospital": hospital, "patients": len(frame), **metrics})
        recovery_cm += confusion_matrix(predictions["recovery_true"], predictions["recovery_pred"], labels=[0, 1, 2])
        all_organ_true.append(predictions["organ_true"])
        all_organ_prob.append(predictions["organ_prob"])
        all_organ_pred.append(predictions["organ_pred"])
        local_results[hospital] = {"predictions": predictions, "metrics": metrics}
        all_y.append(predictions["sepsis_true"])
        all_p.append(predictions["sepsis_prob"])
        all_age.append(predictions["age_group"])
        all_gender.append(predictions["gender"])
    overall = aggregate_binary_reports(binary_reports)
    hospital_table = pd.DataFrame(hospital_rows)
    hospital_table.to_csv(REPORT_DIR / "hospital_wise_performance.csv", index=False)
    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    age = np.concatenate(all_age)
    gender = np.concatenate(all_gender)
    age_table, age_fairness = group_fairness(y, p, age, {0: "Young", 1: "Adult", 2: "Elderly"})
    gender_table, gender_fairness = group_fairness(y, p, gender, {0: "Female", 1: "Male"})
    age_table.to_csv(REPORT_DIR / "fairness_age_groups.csv", index=False)
    gender_table.to_csv(REPORT_DIR / "fairness_gender.csv", index=False)
    (REPORT_DIR / "fairness_summary.json").write_text(json.dumps({
        "age": age_fairness,
        "gender": gender_fairness
    }, indent=2))
    recovery_metrics = {
        "accuracy": float(np.trace(recovery_cm) / max(recovery_cm.sum(), 1)),
        "confusion_matrix": recovery_cm.tolist()
    }
    organ_true = np.concatenate(all_organ_true, axis=0)
    organ_prob = np.concatenate(all_organ_prob, axis=0)
    organ_pred = np.concatenate(all_organ_pred, axis=0)
    organ_table, organ_mean_auroc = multilabel_metrics(organ_true, organ_prob, ["Kidney", "Respiratory", "Cardiovascular"])
    organ_cm = multilabel_cooccurrence(organ_true, organ_pred)
    organ_metrics = {
        "mean_auroc": float(organ_mean_auroc),
        "cooccurrence_matrix": organ_cm.astype(int).tolist()
    }
    pd.DataFrame([recovery_metrics]).to_json(REPORT_DIR / "recovery_metrics.json", orient="records", indent=2)
    organ_table.to_csv(REPORT_DIR / "organ_failure_per_label_metrics.csv", index=False)
    (REPORT_DIR / "organ_failure_metrics.json").write_text(json.dumps(organ_metrics, indent=2))
    pd.DataFrame([overall]).to_csv(REPORT_DIR / "sepsis_final_metrics.csv", index=False)
    save_confusion(recovery_cm, ["Class 0", "Class 1", "Class 2"], PLOT_DIR / "recovery_confusion_matrix.png", "Recovery Outcome Confusion Matrix")
    save_confusion(organ_cm, ["Kidney", "Respiratory", "Cardiovascular"], PLOT_DIR / "organ_failure_confusion_matrix.png", "Organ Failure Confusion Matrix")
    save_hospital_roc(local_results, PLOT_DIR / "cross_hospital_roc.png")
    save_training_curves()
    model_report = parameter_report(model)
    (REPORT_DIR / "evaluation_parameter_report.json").write_text(json.dumps(model_report, indent=2))
    summary = {
        "best_checkpoint_round": int(checkpoint["round"]),
        "batch_size": BATCH_SIZE,
        "overall_sepsis": overall,
        "recovery": recovery_metrics,
        "organ_failure": organ_metrics,
        "age_fairness": age_fairness,
        "gender_fairness": gender_fairness,
        "hospital_mean_auroc": float(hospital_table["auroc"].mean()),
        "hospital_auroc_gap": float(hospital_table["auroc"].max() - hospital_table["auroc"].min()),
        "trainable_parameters": int(model_report["trainable_parameters"]),
        "parameter_reduction_percentage": float(model_report["reduction_percentage"])
    }
    (REPORT_DIR / "final_evaluation_summary.json").write_text(json.dumps(summary, indent=2))
    print("=" * 72)
    print("PAPER-ALIGNED EVALUATION COMPLETE")
    print("=" * 72)
    for metric in ["accuracy", "auroc", "auprc", "precision", "recall", "f1", "specificity"]:
        print(f"{metric:<14}: {overall[metric]:.4f}")
    print(f"Recovery Accuracy : {recovery_metrics['accuracy']:.4f}")
    print(f"Organ Mean AUROC  : {organ_metrics['mean_auroc']:.4f}")
    print(f"Hospital Mean AUC : {hospital_table['auroc'].mean():.4f}")
    print(f"Hospital AUC Gap  : {hospital_table['auroc'].max() - hospital_table['auroc'].min():.4f}")
    print(f"Trainable Params  : {model_report['trainable_parameters']:,}")
    print(f"Parameter Reduction: {model_report['reduction_percentage']:.2f}%")
    print(f"Reports           : {REPORT_DIR}")
    print(f"Plots             : {PLOT_DIR}")


if __name__ == "__main__":
    main()

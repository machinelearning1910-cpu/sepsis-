from pathlib import Path
import copy
import json
import math
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

SEED = 42
ROOT = Path(os.getenv("FAIRLORA_ROOT", "/content/drive/MyDrive/FairLoRA_Sepsis_PaperAligned"))
DATA_DIR = ROOT / "data"
META_DIR = ROOT / "metadata"
CHECKPOINT_DIR = ROOT / "checkpoints"
REPORT_DIR = ROOT / "reports"
for p in [CHECKPOINT_DIR, REPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 128
NUM_ROUNDS = 10
LOCAL_EPOCHS = 30
LEARNING_RATE = 1e-4
HIDDEN_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 4
LORA_RANK = 8
LORA_ALPHA = 16.0
FF_DIM = 892
HEAD_HIDDEN = 64
LAMBDA_RECOVERY = float(os.getenv("LAMBDA_RECOVERY", "1.0"))
LAMBDA_ORGAN = float(os.getenv("LAMBDA_ORGAN", "1.0"))
LAMBDA_FAIR = float(os.getenv("LAMBDA_FAIR", "1.0"))
PAPER_STANDARD_FEDAVG_PARAMS = 1216135
PRETRAINED_BACKBONE_PATH = Path(os.getenv("PRETRAINED_TRANSFORMER_PATH", str(CHECKPOINT_DIR / "pretrained_transformer_backbone.pt")))
HOSPITALS = ["A", "B", "C", "D", "E"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def mount_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except Exception:
        pass


class SepsisSequenceDataset(Dataset):
    def __init__(self, frame, sequence_path):
        self.frame = frame.reset_index(drop=True).copy()
        self.sequence_path = str(sequence_path)
        self._sequences = None
        age_map = {"Young": 0, "Adult": 1, "Elderly": 2}
        self.frame["age_code"] = self.frame["age_group"].map(age_map).astype(int)

    def __len__(self):
        return len(self.frame)

    def _array(self):
        if self._sequences is None:
            self._sequences = np.load(self.sequence_path, mmap_mode="r")
        return self._sequences

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        x = np.array(self._array()[int(row["row_index"])], dtype=np.float32, copy=True)
        return {
            "x": torch.from_numpy(x),
            "sepsis": torch.tensor(float(row["sepsis"]), dtype=torch.float32),
            "recovery": torch.tensor(int(row["recovery_label"]), dtype=torch.long),
            "organ": torch.tensor([float(row["kidney_failure"]), float(row["respiratory_failure"]), float(row["cardiovascular_failure"])], dtype=torch.float32),
            "age_group": torch.tensor(int(row["age_code"]), dtype=torch.long),
            "gender": torch.tensor(int(row["gender"]), dtype=torch.long)
        }


class PretrainedSelfAttention(nn.Module):
    def __init__(self, dim=HIDDEN_DIM, heads=NUM_HEADS):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("Hidden dimension must be divisible by number of heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention, v).transpose(1, 2).contiguous().view(b, t, d)
        return self.out_proj(out)


class PretrainedTransformerBlock(nn.Module):
    def __init__(self, dim=HIDDEN_DIM, heads=NUM_HEADS, ff_dim=FF_DIM):
        super().__init__()
        self.attention = PretrainedSelfAttention(dim=dim, heads=heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.fc2(F.gelu(self.fc1(x))))
        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=24):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=True)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class PretrainedTemporalEncoder(nn.Module):
    def __init__(self, input_dim=37):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, HIDDEN_DIM)
        self.position = SinusoidalPositionalEncoding(HIDDEN_DIM, max_len=24)
        self.encoder = nn.ModuleList([PretrainedTransformerBlock() for _ in range(NUM_LAYERS)])

    def forward(self, x):
        x = self.position(self.input_projection(x))
        for block in self.encoder:
            x = block(x)
        return x


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=LORA_RANK, alpha=LORA_ALPHA, bias=True):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.base = nn.Linear(in_features, out_features, bias=bias)
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(x, self.B @ self.A) * self.scaling


class FairLoRASelfAttention(nn.Module):
    def __init__(self, dim=HIDDEN_DIM, heads=NUM_HEADS, rank=LORA_RANK):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("Hidden dimension must be divisible by number of heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = LoRALinear(dim, dim, rank=rank)
        self.k_proj = LoRALinear(dim, dim, rank=rank)
        self.v_proj = LoRALinear(dim, dim, rank=rank)
        self.out_proj = LoRALinear(dim, dim, rank=rank)

    def forward(self, x):
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention, v).transpose(1, 2).contiguous().view(b, t, d)
        return self.out_proj(out)


class FairLoRATransformerBlock(nn.Module):
    def __init__(self, dim=HIDDEN_DIM, heads=NUM_HEADS, rank=LORA_RANK, ff_dim=FF_DIM):
        super().__init__()
        self.attention = FairLoRASelfAttention(dim=dim, heads=heads, rank=rank)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = LoRALinear(dim, ff_dim, rank=rank)
        self.fc2 = LoRALinear(ff_dim, dim, rank=rank)
        for parameter in self.norm1.parameters():
            parameter.requires_grad = False
        for parameter in self.norm2.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.fc2(F.gelu(self.fc1(x))))
        return x


class FairLoRAMultiTaskModel(nn.Module):
    def __init__(self, input_dim=37):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, HIDDEN_DIM)
        for parameter in self.input_projection.parameters():
            parameter.requires_grad = False
        self.position = SinusoidalPositionalEncoding(HIDDEN_DIM, max_len=24)
        self.encoder = nn.ModuleList([FairLoRATransformerBlock() for _ in range(NUM_LAYERS)])
        self.sepsis_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HEAD_HIDDEN),
            nn.ReLU(),
            nn.Linear(HEAD_HIDDEN, 1)
        )
        self.recovery_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HEAD_HIDDEN),
            nn.ReLU(),
            nn.Linear(HEAD_HIDDEN, 3)
        )
        self.organ_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HEAD_HIDDEN),
            nn.ReLU(),
            nn.Linear(HEAD_HIDDEN, 3)
        )

    def forward(self, x):
        x = self.position(self.input_projection(x))
        for block in self.encoder:
            x = block(x)
        z = x.mean(dim=1)
        return self.sepsis_head(z), self.recovery_head(z), self.organ_head(z)


def load_pretrained_transformer(model, checkpoint_path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"The manuscript specifies a pre-trained Transformer but does not define its pretraining procedure. "
            f"Provide the pre-trained Transformer checkpoint at {checkpoint_path}."
        )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(source_state, dict):
        raise TypeError("Pre-trained Transformer checkpoint must contain a state dictionary")
    target_state = model.state_dict()
    mapping = {"input_projection.weight": "input_projection.weight", "input_projection.bias": "input_projection.bias"}
    for layer in range(NUM_LAYERS):
        for projection in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            mapping[f"encoder.{layer}.attention.{projection}.base.weight"] = f"encoder.{layer}.attention.{projection}.weight"
            mapping[f"encoder.{layer}.attention.{projection}.base.bias"] = f"encoder.{layer}.attention.{projection}.bias"
        mapping[f"encoder.{layer}.fc1.base.weight"] = f"encoder.{layer}.fc1.weight"
        mapping[f"encoder.{layer}.fc1.base.bias"] = f"encoder.{layer}.fc1.bias"
        mapping[f"encoder.{layer}.fc2.base.weight"] = f"encoder.{layer}.fc2.weight"
        mapping[f"encoder.{layer}.fc2.base.bias"] = f"encoder.{layer}.fc2.bias"
        mapping[f"encoder.{layer}.norm1.weight"] = f"encoder.{layer}.norm1.weight"
        mapping[f"encoder.{layer}.norm1.bias"] = f"encoder.{layer}.norm1.bias"
        mapping[f"encoder.{layer}.norm2.weight"] = f"encoder.{layer}.norm2.weight"
        mapping[f"encoder.{layer}.norm2.bias"] = f"encoder.{layer}.norm2.bias"
    missing = []
    for target_key, source_key in mapping.items():
        if source_key not in source_state:
            missing.append(source_key)
            continue
        if target_state[target_key].shape != source_state[source_key].shape:
            raise ValueError(f"Shape mismatch for pre-trained parameter {source_key}")
        target_state[target_key] = source_state[source_key].detach().cpu().to(target_state[target_key].dtype)
    if missing:
        raise KeyError(f"Pre-trained Transformer checkpoint is missing {len(missing)} required tensors")
    model.load_state_dict(target_state)
    return model


def trainable_parameter_names(model):
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def parameter_report(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    reduction = (1.0 - trainable / PAPER_STANDARD_FEDAVG_PARAMS) * 100.0
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "paper_standard_fedavg_parameters": int(PAPER_STANDARD_FEDAVG_PARAMS),
        "reduction_percentage": float(reduction),
        "trainable_parameter_names": trainable_parameter_names(model)
    }


def rate_gap(prediction, target, groups, target_value=None):
    rates = []
    for group in torch.unique(groups):
        mask = groups == group
        if target_value is not None:
            mask = mask & (target == target_value)
        if torch.any(mask):
            rates.append(prediction[mask].float().mean())
    if len(rates) < 2:
        return prediction.new_zeros((), dtype=torch.float32)
    rates = torch.stack(rates)
    return rates.max() - rates.min()


def combined_demographic_rate_gap(prediction, target, age_group, gender, target_value=None):
    rates = []
    for groups in [age_group.view(-1).long(), gender.view(-1).long()]:
        for group in torch.unique(groups):
            mask = groups == group
            if target_value is not None:
                mask = mask & (target == target_value)
            if torch.any(mask):
                rates.append(prediction[mask].float().mean())
    if len(rates) < 2:
        return prediction.new_zeros((), dtype=torch.float32)
    rates = torch.stack(rates)
    return rates.max() - rates.min()


def fairness_loss(sepsis_logits, sepsis_target, age_group, gender):
    probability = torch.sigmoid(sepsis_logits.view(-1))
    prediction = (probability >= 0.5).float()
    target = sepsis_target.view(-1).float()
    dpd = combined_demographic_rate_gap(prediction, target, age_group, gender)
    eod = combined_demographic_rate_gap(prediction, target, age_group, gender, target_value=1.0)
    fpr_gap = combined_demographic_rate_gap(prediction, target, age_group, gender, target_value=0.0)
    return dpd + eod + fpr_gap


def multitask_loss(outputs, batch):
    sepsis_logits, recovery_logits, organ_logits = outputs
    sepsis_target = batch["sepsis"].view(-1, 1)
    recovery_target = batch["recovery"]
    organ_target = batch["organ"]
    l_sepsis = F.binary_cross_entropy_with_logits(sepsis_logits, sepsis_target)
    l_recovery = F.cross_entropy(recovery_logits, recovery_target)
    l_organ = F.binary_cross_entropy_with_logits(organ_logits, organ_target.float())
    l_fair = fairness_loss(sepsis_logits, sepsis_target, batch["age_group"], batch["gender"])
    total = l_sepsis + LAMBDA_RECOVERY * l_recovery + LAMBDA_ORGAN * l_organ + LAMBDA_FAIR * l_fair
    return total, l_sepsis, l_recovery, l_organ, l_fair


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def make_loader(frame, sequence_path, shuffle):
    workers = min(2, os.cpu_count() or 1)
    return DataLoader(
        SepsisSequenceDataset(frame, sequence_path),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0
    )


class FederatedHospital:
    def __init__(self, hospital, train_frame, validation_frame, sequence_path):
        self.hospital = hospital
        self.train_frame = train_frame.reset_index(drop=True)
        self.validation_frame = validation_frame.reset_index(drop=True)
        self.train_loader = make_loader(self.train_frame, sequence_path, True)
        self.validation_loader = make_loader(self.validation_frame, sequence_path, False)

    @property
    def train_size(self):
        return len(self.train_frame)

    @property
    def validation_size(self):
        return len(self.validation_frame)

    def train(self, global_model, device):
        model = copy.deepcopy(global_model).to(device)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=LEARNING_RATE
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=LOCAL_EPOCHS)
        epoch_losses = []
        for _ in tqdm(range(LOCAL_EPOCHS), desc=f"Hospital {self.hospital} local epochs", leave=False):
            model.train()
            total_loss = 0.0
            batches = 0
            for batch in self.train_loader:
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch["x"])
                loss, _, _, _, _ = multitask_loss(outputs, batch)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                batches += 1
            scheduler.step()
            epoch_losses.append(total_loss / max(batches, 1))
        names = set(trainable_parameter_names(model))
        update = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
            if name in names
        }
        return update, float(np.mean(epoch_losses))

    def validate(self, model, device):
        model.eval()
        losses = []
        y_true = []
        y_prob = []
        with torch.no_grad():
            for batch in self.validation_loader:
                batch = move_batch(batch, device)
                outputs = model(batch["x"])
                total, sepsis, recovery, organ, fair = multitask_loss(outputs, batch)
                losses.append([total.item(), sepsis.item(), recovery.item(), organ.item(), fair.item()])
                y_true.extend(batch["sepsis"].view(-1).cpu().numpy().astype(int).tolist())
                y_prob.extend(torch.sigmoid(outputs[0]).view(-1).cpu().numpy().tolist())
        y = np.asarray(y_true, dtype=np.int64)
        p = np.asarray(y_prob, dtype=np.float64)
        pred = (p >= 0.5).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        loss_array = np.asarray(losses, dtype=np.float64)
        return {
            "hospital": self.hospital,
            "n": int(len(y)),
            "total_loss": float(loss_array[:, 0].mean()),
            "sepsis_loss": float(loss_array[:, 1].mean()),
            "recovery_loss": float(loss_array[:, 2].mean()),
            "organ_loss": float(loss_array[:, 3].mean()),
            "fairness_loss": float(loss_array[:, 4].mean()),
            "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
            "auprc": float(average_precision_score(y, p)) if y.sum() > 0 else np.nan,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }


def weighted_fedavg(states, sizes):
    total = float(sum(sizes))
    result = {}
    for key in states[0]:
        value = None
        for state, size in zip(states, sizes):
            contribution = state[key].float() * (float(size) / total)
            value = contribution if value is None else value + contribution
        result[key] = value
    return result


def load_trainable_state(model, state):
    full = model.state_dict()
    for key, value in state.items():
        full[key] = value.to(dtype=full[key].dtype)
    model.load_state_dict(full)


def aggregate_validation_reports(reports):
    weights = np.asarray([report["n"] for report in reports], dtype=np.float64)
    weights = weights / weights.sum()
    tn = sum(report["tn"] for report in reports)
    fp = sum(report["fp"] for report in reports)
    fn = sum(report["fn"] for report in reports)
    tp = sum(report["tp"] for report in reports)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    result = {
        "n": int(sum(report["n"] for report in reports)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(np.nansum(weights * np.asarray([report["auroc"] for report in reports], dtype=np.float64))),
        "auprc": float(np.nansum(weights * np.asarray([report["auprc"] for report in reports], dtype=np.float64)))
    }
    for key in ["total_loss", "sepsis_loss", "recovery_loss", "organ_loss", "fairness_loss"]:
        result[key] = float(np.sum(weights * np.asarray([report[key] for report in reports], dtype=np.float64)))
    return result


def build_hospitals(sequence_path):
    train = pd.read_csv(META_DIR / "hospital_assignment_train.csv")
    validation = pd.read_csv(META_DIR / "hospital_assignment_validation.csv")
    hospitals = {}
    for hospital in HOSPITALS:
        hospitals[hospital] = FederatedHospital(
            hospital,
            train[train["hospital"] == hospital].copy(),
            validation[validation["hospital"] == hospital].copy(),
            sequence_path
        )
    return hospitals


def main():
    mount_drive()
    sequence_path = DATA_DIR / "sequences_24x37.npy"
    if not sequence_path.exists():
        raise FileNotFoundError("Run dataset_preprocessing.py first")
    hospitals = build_hospitals(sequence_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = FairLoRAMultiTaskModel(input_dim=37)
    global_model = load_pretrained_transformer(global_model, PRETRAINED_BACKBONE_PATH).to(device)
    report = parameter_report(global_model)
    if report["trainable_parameters"] != 123271:
        raise RuntimeError(f"Expected 123271 trainable parameters, found {report['trainable_parameters']}")
    (REPORT_DIR / "parameter_report.json").write_text(json.dumps(report, indent=2))
    history = []
    best_auc = -np.inf
    for round_index in range(1, NUM_ROUNDS + 1):
        states = []
        sizes = []
        local_losses = []
        for hospital in HOSPITALS:
            state, loss = hospitals[hospital].train(global_model, device)
            states.append(state)
            sizes.append(hospitals[hospital].train_size)
            local_losses.append(loss)
        averaged_state = weighted_fedavg(states, sizes)
        load_trainable_state(global_model, averaged_state)
        local_validation = [hospitals[hospital].validate(global_model, device) for hospital in HOSPITALS]
        aggregate = aggregate_validation_reports(local_validation)
        row = {
            "round": round_index,
            "train_loss": float(np.average(local_losses, weights=sizes)),
            **aggregate
        }
        history.append(row)
        checkpoint = {
            "round": round_index,
            "model_state_dict": {key: value.detach().cpu() for key, value in global_model.state_dict().items()},
            "model_config": {
                "input_dim": 37,
                "hidden_dim": HIDDEN_DIM,
                "heads": NUM_HEADS,
                "layers": NUM_LAYERS,
                "rank": LORA_RANK,
                "alpha": LORA_ALPHA,
                "ff_dim": FF_DIM,
                "head_hidden": HEAD_HIDDEN
            },
            "training_config": {
                "batch_size": BATCH_SIZE,
                "communication_rounds": NUM_ROUNDS,
                "local_epochs": LOCAL_EPOCHS,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "learning_rate": LEARNING_RATE,
                "lambda_recovery": LAMBDA_RECOVERY,
                "lambda_organ": LAMBDA_ORGAN,
                "lambda_fair": LAMBDA_FAIR,
                "fairness_definition": "DPD + EOD + FPR Gap from hard thresholded predictions across age and gender groups",
                "demographic_grouping": "joint age and gender categories",
                "validation": "hospital-local validation with aggregate statistics only"
            },
            "validation_metrics": aggregate,
            "hospital_validation_metrics": local_validation
        }
        torch.save(checkpoint, CHECKPOINT_DIR / f"round_{round_index:02d}.pt")
        if aggregate["auroc"] > best_auc:
            best_auc = aggregate["auroc"]
            torch.save(checkpoint, CHECKPOINT_DIR / "best_model.pt")
        print(
            f"Round {round_index:02d}/{NUM_ROUNDS} | "
            f"Train Loss {row['train_loss']:.4f} | Validation Loss {aggregate['total_loss']:.4f} | "
            f"AUROC {aggregate['auroc']:.4f} | AUPRC {aggregate['auprc']:.4f} | F1 {aggregate['f1']:.4f}"
        )
    pd.DataFrame(history).to_csv(REPORT_DIR / "training_history.csv", index=False)
    print("=" * 72)
    print("PAPER-ALIGNED FEDERATED TRAINING COMPLETE")
    print("=" * 72)
    print(f"Device                : {device}")
    print(f"Communication rounds  : {NUM_ROUNDS}")
    print(f"Local epochs          : {LOCAL_EPOCHS}")
    print(f"Batch size            : {BATCH_SIZE}")
    print(f"Trainable parameters  : {report['trainable_parameters']:,}")
    print(f"Parameter reduction   : {report['reduction_percentage']:.2f}%")
    print(f"Best validation AUROC : {best_auc:.4f}")
    print(f"Best checkpoint       : {CHECKPOINT_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    main()

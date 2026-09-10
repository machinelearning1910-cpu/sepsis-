from pathlib import Path
import json
import os
import random
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

SEED = 42
ROOT = Path(os.getenv("FAIRLORA_ROOT", "/content/drive/MyDrive/FairLoRA_Sepsis_PaperAligned"))
DATA_DIR = ROOT / "data"
META_DIR = ROOT / "metadata"
REPORT_DIR = ROOT / "reports"
PLOT_DIR = ROOT / "plots"
for p in [DATA_DIR, META_DIR, REPORT_DIR, PLOT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2",
    "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine",
    "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT",
    "WBC", "Platelets", "Age", "Gender", "HospAdmTime", "ICULOS"
]

PAPER_FEATURE_NAMES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2",
    "AST", "BUN", "AlkalinePhos", "Calcium", "Chloride", "Creatinine",
    "Bilirubin Direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin Total", "TroponinI", "Hct", "Hgb", "PTT",
    "WBC", "Platelets", "Age", "Gender", "HospAdmTime", "ICULOS"
]

WINDOW_SIZE = 24
VALIDATION_FRACTION = float(os.getenv("VALIDATION_FRACTION", "0.10"))
RECOVERY_Q33 = 29.0
RECOVERY_Q66 = 44.0
HOSPITALS = ["A", "B", "C", "D", "E"]
AGES = ["Young", "Adult", "Elderly"]
GENDERS = [0, 1]

random.seed(SEED)
np.random.seed(SEED)


def ensure_packages():
    required = {
        "matplotlib": "matplotlib",
        "sklearn": "scikit-learn",
        "tqdm": "tqdm",
        "pyarrow": "pyarrow",
        "scipy": "scipy"
    }
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except Exception:
            missing.append(package)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


def mount_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except Exception:
        pass


def resolve_dataset_path():
    explicit = os.getenv("PHYSIONET_2019_PATH", "").strip()
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    target = DATA_DIR / "physionet_challenge_2019"
    if list(target.rglob("*.psv")):
        return target
    target.mkdir(parents=True, exist_ok=True)
    url = "https://physionet.org/files/challenge-2019/1.0.0/training/"
    subprocess.check_call(["wget", "-r", "-N", "-c", "-np", "--show-progress", url, "-P", str(target)])
    return target


def discover_patient_files(dataset_path):
    files = sorted(dataset_path.rglob("*.psv"))
    if not files:
        raise RuntimeError(f"No PSV patient files were found under {dataset_path}")
    return files


def normalize_patient_id(value):
    value = str(value).strip()
    return Path(value).stem


def read_patient_summary(path):
    df = pd.read_csv(path, sep="|")
    required = FEATURES + ["SepsisLabel"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name}: missing columns {missing}")
    max_iculos = float(pd.to_numeric(df["ICULOS"], errors="coerce").max())
    recovery_label = 0 if max_iculos <= RECOVERY_Q33 else 1 if max_iculos <= RECOVERY_Q66 else 2
    kidney_failure = int(((pd.to_numeric(df["Creatinine"], errors="coerce") > 2.0) | (pd.to_numeric(df["BUN"], errors="coerce") > 40.0)).any())
    respiratory_failure = int(((pd.to_numeric(df["O2Sat"], errors="coerce") < 90.0) | (pd.to_numeric(df["Resp"], errors="coerce") > 30.0)).any())
    cardiovascular_failure = int(((pd.to_numeric(df["MAP"], errors="coerce") < 65.0) | (pd.to_numeric(df["SBP"], errors="coerce") < 90.0)).any())
    return {
        "patient_id": normalize_patient_id(path.name),
        "file_path": str(path),
        "age": float(df["Age"].iloc[0]),
        "gender": int(df["Gender"].iloc[0]),
        "sepsis": int(df["SepsisLabel"].max()),
        "recovery_label": int(recovery_label),
        "kidney_failure": int(kidney_failure),
        "respiratory_failure": int(respiratory_failure),
        "cardiovascular_failure": int(cardiovascular_failure),
        "max_iculos": float(max_iculos),
        "rows": int(len(df))
    }


def build_metadata(patient_files):
    records = []
    failures = []
    for path in tqdm(patient_files, desc="Auditing patients"):
        try:
            records.append(read_patient_summary(path))
        except Exception as exc:
            failures.append({"file": str(path), "error": str(exc)})
    metadata = pd.DataFrame(records)
    if metadata.empty:
        raise RuntimeError("No valid patient records were found")
    metadata["age_group"] = pd.cut(
        metadata["age"],
        bins=[-np.inf, 40.0, 60.0, np.inf],
        labels=AGES,
        include_lowest=True
    ).astype(str)
    metadata = metadata[metadata["rows"] >= WINDOW_SIZE].copy().reset_index(drop=True)
    metadata["row_index"] = np.arange(len(metadata), dtype=np.int64)
    (REPORT_DIR / "dataset_failures.json").write_text(json.dumps(failures, indent=2))
    return metadata


def proportional_counts(size, proportions):
    proportions = np.asarray(proportions, dtype=np.float64)
    raw = proportions * int(size)
    counts = np.floor(raw).astype(np.int64)
    remainder = int(size - counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[:remainder]] += 1
    return counts


def hospital_targets(sizes):
    age_targets = {
        "A": proportional_counts(sizes[0], (0.70, 0.20, 0.10)),
        "B": proportional_counts(sizes[1], (0.10, 0.70, 0.20)),
        "C": proportional_counts(sizes[2], (0.10, 0.20, 0.70))
    }
    gender_targets = {
        "D": proportional_counts(sizes[3], (0.70, 0.30)),
        "E": proportional_counts(sizes[4], (0.30, 0.70))
    }
    return age_targets, gender_targets


def solve_demographic_allocation(df, sizes):
    cells = [(age, gender) for age in AGES for gender in GENDERS]
    available = np.asarray([
        int(((df["age_group"] == age) & (df["gender"] == gender)).sum())
        for age, gender in cells
    ], dtype=np.int64)
    age_targets, gender_targets = hospital_targets(sizes)
    n_h = len(HOSPITALS)
    n_c = len(cells)
    n_v = n_h * n_c
    rows = []
    rhs = []
    for c in range(n_c):
        row = np.zeros(n_v, dtype=np.float64)
        for h in range(n_h):
            row[h * n_c + c] = 1.0
        rows.append(row)
        rhs.append(float(available[c]))
    for h, size in enumerate(sizes):
        row = np.zeros(n_v, dtype=np.float64)
        row[h * n_c:(h + 1) * n_c] = 1.0
        rows.append(row)
        rhs.append(float(size))
    for hospital in ["A", "B", "C"]:
        h = HOSPITALS.index(hospital)
        target = age_targets[hospital]
        for age_idx, age in enumerate(AGES[:2]):
            row = np.zeros(n_v, dtype=np.float64)
            for c, cell in enumerate(cells):
                if cell[0] == age:
                    row[h * n_c + c] = 1.0
            rows.append(row)
            rhs.append(float(target[age_idx]))
    for hospital in ["D", "E"]:
        h = HOSPITALS.index(hospital)
        female_target = int(gender_targets[hospital][0])
        row = np.zeros(n_v, dtype=np.float64)
        for c, cell in enumerate(cells):
            if cell[1] == 0:
                row[h * n_c + c] = 1.0
        rows.append(row)
        rhs.append(float(female_target))
    matrix = np.vstack(rows)
    rhs = np.asarray(rhs, dtype=np.float64)
    result = milp(
        c=np.zeros(n_v, dtype=np.float64),
        integrality=np.ones(n_v, dtype=np.int8),
        bounds=Bounds(np.zeros(n_v), np.full(n_v, np.inf)),
        constraints=LinearConstraint(matrix, rhs, rhs),
        options={"time_limit": 120.0}
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Unable to satisfy the manuscript-defined hospital distributions: {result.message}")
    counts = np.rint(result.x).astype(np.int64).reshape(n_h, n_c)
    return cells, counts, age_targets, gender_targets


def verify_hospital_targets(assignment, age_targets, gender_targets):
    for hospital in ["A", "B", "C"]:
        frame = assignment[assignment["hospital"] == hospital]
        observed = np.asarray([(frame["age_group"] == age).sum() for age in AGES], dtype=np.int64)
        if not np.array_equal(observed, age_targets[hospital]):
            raise RuntimeError(f"Hospital {hospital} does not match its manuscript age distribution")
    for hospital in ["D", "E"]:
        frame = assignment[assignment["hospital"] == hospital]
        observed = np.asarray([(frame["gender"] == gender).sum() for gender in GENDERS], dtype=np.int64)
        if not np.array_equal(observed, gender_targets[hospital]):
            raise RuntimeError(f"Hospital {hospital} does not match its manuscript gender distribution")


def allocate_non_iid(metadata):
    rng = np.random.default_rng(SEED)
    frame = metadata.reset_index(drop=True).reset_index(names="allocation_index")
    sizes = np.full(5, len(frame) // 5, dtype=np.int64)
    sizes[:len(frame) % 5] += 1
    cells, counts, age_targets, gender_targets = solve_demographic_allocation(frame, sizes)
    assignments = []
    for c, (age, gender) in enumerate(cells):
        candidates = frame.loc[
            (frame["age_group"] == age) & (frame["gender"] == gender),
            "allocation_index"
        ].to_numpy(dtype=np.int64)
        rng.shuffle(candidates)
        offset = 0
        for h, hospital in enumerate(HOSPITALS):
            count = int(counts[h, c])
            selected = candidates[offset:offset + count]
            assignments.extend((int(i), hospital) for i in selected)
            offset += count
        if offset != len(candidates):
            raise RuntimeError("Hospital allocation did not consume every eligible patient exactly once")
    assignment = pd.DataFrame(assignments, columns=["allocation_index", "hospital"])
    assignment = assignment.merge(frame, on="allocation_index", how="left", validate="one_to_one")
    if len(assignment) != len(frame) or assignment["patient_id"].nunique() != len(frame):
        raise RuntimeError("Hospital allocation produced overlap or omission")
    verify_hospital_targets(assignment, age_targets, gender_targets)
    return assignment


def split_within_hospitals(assignment):
    train_parts = []
    val_parts = []
    for hospital in HOSPITALS:
        frame = assignment[assignment["hospital"] == hospital].copy()
        stratify = frame["sepsis"] if frame["sepsis"].value_counts().min() >= 2 else None
        train_part, val_part = train_test_split(
            frame,
            test_size=VALIDATION_FRACTION,
            random_state=SEED,
            stratify=stratify
        )
        train_part = train_part.copy()
        val_part = val_part.copy()
        train_part["split"] = "train"
        val_part["split"] = "validation"
        train_parts.append(train_part)
        val_parts.append(val_part)
    train = pd.concat(train_parts, ignore_index=True)
    validation = pd.concat(val_parts, ignore_index=True)
    return train, validation


def pooled_training_statistics(train_assignment):
    medians = []
    means = []
    stds = []
    frames = []
    for path in tqdm(train_assignment["file_path"], desc="Collecting training statistics"):
        frame = pd.read_csv(path, sep="|", usecols=FEATURES)
        frames.append(frame[FEATURES])
    pooled = pd.concat(frames, ignore_index=True)
    for feature in tqdm(FEATURES, desc="Median and Z-score statistics"):
        median = float(pooled[feature].median(skipna=True))
        filled = pooled[feature].fillna(median).astype(float)
        mean = float(filled.mean())
        std = float(filled.std(ddof=0))
        medians.append(median)
        means.append(mean)
        stds.append(max(std, 1e-6))
    return np.asarray(medians), np.asarray(means), np.asarray(stds)


def preprocess_patient(path, medians, means, stds):
    frame = pd.read_csv(path, sep="|", usecols=FEATURES)
    x = frame.iloc[:WINDOW_SIZE][FEATURES].to_numpy(dtype=np.float32, copy=True)
    if x.shape[0] < WINDOW_SIZE:
        raise ValueError(f"{Path(path).name} contains fewer than 24 hourly observations")
    for j in range(x.shape[1]):
        missing = np.isnan(x[:, j])
        if missing.any():
            x[missing, j] = float(medians[j])
    x = (x - means.astype(np.float32)) / stds.astype(np.float32)
    return x.astype(np.float32, copy=False)


def write_sequences(metadata, medians, means, stds):
    path = DATA_DIR / "sequences_24x37.npy"
    mmap = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(len(metadata), WINDOW_SIZE, len(FEATURES))
    )
    for row in tqdm(metadata.itertuples(index=False), total=len(metadata), desc="Building 24x37 sequences"):
        mmap[int(row.row_index)] = preprocess_patient(row.file_path, medians, means, stds)
    mmap.flush()
    del mmap
    return path


def hospital_distribution(assignment):
    rows = []
    for hospital in HOSPITALS:
        frame = assignment[assignment["hospital"] == hospital]
        age = frame["age_group"].value_counts(normalize=True).mul(100)
        gender = frame["gender"].value_counts(normalize=True).mul(100)
        rows.append({
            "hospital": hospital,
            "patients": int(len(frame)),
            "young_pct": float(age.get("Young", 0.0)),
            "adult_pct": float(age.get("Adult", 0.0)),
            "elderly_pct": float(age.get("Elderly", 0.0)),
            "female_pct": float(gender.get(0, 0.0)),
            "male_pct": float(gender.get(1, 0.0)),
            "sepsis_pct": float(frame["sepsis"].mean() * 100.0)
        })
    return pd.DataFrame(rows)


def save_samples(metadata, sequence_path):
    sequences = np.load(sequence_path, mmap_mode="r")
    ids = metadata.sample(n=min(5, len(metadata)), random_state=SEED)["row_index"].tolist()
    samples = []
    for idx in ids:
        patient = metadata.loc[metadata["row_index"] == idx].iloc[0]
        frame = pd.DataFrame(np.asarray(sequences[idx]), columns=PAPER_FEATURE_NAMES)
        frame.insert(0, "hour", np.arange(1, WINDOW_SIZE + 1))
        frame.insert(0, "patient_id", patient["patient_id"])
        samples.append(frame)
    pd.concat(samples, ignore_index=True).to_csv(REPORT_DIR / "dataset_samples_normalized.csv", index=False)


def main():
    ensure_packages()
    mount_drive()
    dataset_path = resolve_dataset_path()
    patient_files = discover_patient_files(dataset_path)
    metadata = build_metadata(patient_files)
    assignment = allocate_non_iid(metadata)
    train_assignment, validation_assignment = split_within_hospitals(assignment)
    medians, means, stds = pooled_training_statistics(train_assignment)
    sequence_path = write_sequences(metadata, medians, means, stds)
    metadata.to_csv(META_DIR / "patients.csv", index=False)
    assignment.to_csv(META_DIR / "hospital_assignment_all.csv", index=False)
    train_assignment.to_csv(META_DIR / "hospital_assignment_train.csv", index=False)
    validation_assignment.to_csv(META_DIR / "hospital_assignment_validation.csv", index=False)
    hospital_distribution(assignment).to_csv(REPORT_DIR / "hospital_distribution_all.csv", index=False)
    hospital_distribution(train_assignment).to_csv(REPORT_DIR / "hospital_distribution_train.csv", index=False)
    hospital_distribution(validation_assignment).to_csv(REPORT_DIR / "hospital_distribution_validation.csv", index=False)
    np.save(META_DIR / "feature_medians.npy", medians)
    np.save(META_DIR / "feature_means.npy", means)
    np.save(META_DIR / "feature_stds.npy", stds)
    settings = {
        "features": PAPER_FEATURE_NAMES,
        "feature_count": 37,
        "sequence_length_hours": 24,
        "shorter_than_24_hours": "records with fewer than 24 hourly observations excluded",
        "longer_than_24_hours": "first 24 hourly observations retained",
        "missing_value_handling": "median imputation",
        "normalization": "Z-score",
        "federated_hospitals": 5,
        "hospital_A": "70% Young, 20% Adult, 10% Elderly",
        "hospital_B": "10% Young, 70% Adult, 20% Elderly",
        "hospital_C": "10% Young, 20% Adult, 70% Elderly",
        "hospital_D": "70% Female, 30% Male",
        "hospital_E": "30% Female, 70% Male",
        "recovery_label": "0 if maximum ICULOS <= 29 hours, 1 if 29 < maximum ICULOS <= 44 hours, 2 if maximum ICULOS > 44 hours",
        "kidney_failure_label": "1 if Creatinine > 2.0 or BUN > 40.0 at any recorded time, otherwise 0",
        "respiratory_failure_label": "1 if O2Sat < 90.0 or Resp > 30.0 at any recorded time, otherwise 0",
        "cardiovascular_failure_label": "1 if MAP < 65.0 or SBP < 90.0 at any recorded time, otherwise 0",
        "organ_target": "three-element multi-label vector [kidney_failure, respiratory_failure, cardiovascular_failure]",
        "validation_fraction_within_each_hospital": VALIDATION_FRACTION
    }
    (REPORT_DIR / "dataset_preprocessing_configuration.json").write_text(json.dumps(settings, indent=2))
    label_summary = {
        "recovery": metadata["recovery_label"].value_counts().sort_index().to_dict(),
        "kidney_failure": metadata["kidney_failure"].value_counts().sort_index().to_dict(),
        "respiratory_failure": metadata["respiratory_failure"].value_counts().sort_index().to_dict(),
        "cardiovascular_failure": metadata["cardiovascular_failure"].value_counts().sort_index().to_dict()
    }
    (REPORT_DIR / "derived_label_distribution.json").write_text(json.dumps(label_summary, indent=2))
    save_samples(metadata, sequence_path)
    print("=" * 72)
    print("PAPER-ALIGNED DATASET PREPARATION COMPLETE")
    print("=" * 72)
    print(f"Eligible patients      : {len(metadata):,}")
    print(f"Clinical features      : {len(FEATURES)}")
    print(f"Sequence shape         : 24 x 37")
    print(f"Training patients      : {len(train_assignment):,}")
    print(f"Validation patients    : {len(validation_assignment):,}")
    print(f"Sequences              : {sequence_path}")
    print(f"Metadata               : {META_DIR}")
    print(f"Reports                : {REPORT_DIR}")


if __name__ == "__main__":
    main()

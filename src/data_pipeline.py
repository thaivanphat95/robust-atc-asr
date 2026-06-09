import hashlib
import os
from collections import defaultdict

import librosa
import soundfile as sf
from datasets import concatenate_datasets, load_dataset


TTS_LABEL_OFFSET = 1_000_000_000_000
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def normalize_text(text: str) -> str:
    return str(text).strip().upper()


def safe_int(v, default=-1):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def stable_text_id(text: str, n_hex=12) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:n_hex], 16)


def resolve_audio_path(path_audio):
    if os.path.isabs(path_audio):
        return path_audio
    return os.path.join(PROJECT_DIR, path_audio)


def process_from_file(batch, processor, max_duration_s=8.0):
    audio_input, sample_rate = sf.read(resolve_audio_path(batch["audio_filename"]))
    text = normalize_text(batch["transcript"])

    if getattr(audio_input, "ndim", 1) > 1:
        audio_input = audio_input.mean(axis=1)

    target_sr = 16000
    if sample_rate != target_sr:
        audio_input = librosa.resample(audio_input, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr

    max_samples = int(max_duration_s * sample_rate)
    if len(audio_input) > max_samples:
        audio_input = audio_input[:max_samples]

    batch["input_values"] = processor(audio_input, sampling_rate=sample_rate).input_values[0]
    batch["labels"] = processor.tokenizer(text, truncation=True, max_length=128).input_ids
    batch["transcript"] = text
    batch["supcon_id"] = safe_int(batch.get("supcon_id", -1), default=-1)
    batch["is_synth"] = safe_int(batch.get("is_synth", 0), default=0)
    return batch


def load_repeated_train_val(path, processor):
    train = load_dataset("csv", data_files=os.path.join(path, "train.csv"), split="train")
    val = load_dataset("csv", data_files=os.path.join(path, "val.csv"), split="train")

    train = train.map(lambda b: {**b, "is_synth": 0, "supcon_id": stable_text_id(normalize_text(b["transcript"]), n_hex=8)})
    val = val.map(lambda b: {**b, "is_synth": 0, "supcon_id": -1})

    train = train.map(lambda b: process_from_file(b, processor, max_duration_s=10.0))
    val = val.map(lambda b: process_from_file(b, processor, max_duration_s=10.0))

    keep = ["input_values", "labels", "supcon_id", "is_synth"]
    train = train.remove_columns([c for c in train.column_names if c not in keep])
    val = val.remove_columns([c for c in val.column_names if c not in keep])
    return train, val


def load_similar_train_val(train_csv, val_csv, processor):
    train = load_dataset("csv", data_files=train_csv, split="train")
    val = load_dataset("csv", data_files=val_csv, split="train")

    train = train.map(lambda b: {**b, "is_synth": 0})
    val = val.map(lambda b: {**b, "is_synth": 0})

    train = train.map(lambda b: process_from_file(b, processor))
    val = val.map(lambda b: process_from_file(b, processor))

    keep = ["input_values", "labels", "supcon_id", "is_synth"]
    train = train.remove_columns([c for c in train.column_names if c not in keep])
    val = val.remove_columns([c for c in val.column_names if c not in keep])
    return train, val


def compute_supcon_stats(dataset):
    sup_ids = [safe_int(v, -1) for v in dataset["supcon_id"]]
    total = len(sup_ids)
    known = sum(1 for sid in sup_ids if sid != -1)
    unknown = total - known
    return {"total": total, "known": known, "unknown": unknown,
        "known_ratio": (known / total) if total > 0 else 0.0,
        "unknown_ratio": (unknown / total) if total > 0 else 0.0}


def suggest_ctc_batches_per_supcon(known_ratio, max_ctc_steps=4):
    if known_ratio <= 0:
        return int(max_ctc_steps)
    raw = (1.0 - known_ratio) / known_ratio
    return int(max(1, min(max_ctc_steps, round(raw))))


def load_synthetic_train(train_orig_csv, train_synth_csv, processor):
    orig = load_dataset("csv", data_files=train_orig_csv, split="train")
    synth = load_dataset("csv", data_files=train_synth_csv, split="train")

    orig = orig.map(lambda b: {**b, "is_synth": 0, "TTS": "orig", "source": "orig", "supcon_id": stable_text_id(normalize_text(b["transcript"]), n_hex=8)})

    if "TTS" not in synth.column_names:
        synth = synth.map(lambda b: {**b, "TTS": "synth"})
    if "source" not in synth.column_names:
        synth = synth.map(lambda b: {**b, "source": "synthetic"})

    synth = synth.map(lambda b: {**b, "is_synth": 1, "supcon_id": stable_text_id(normalize_text(b["transcript"]), n_hex=8)})

    train = concatenate_datasets([orig, synth])
    train = train.map(lambda b: process_from_file(b, processor))

    keep = ["input_values", "labels", "supcon_id", "is_synth", "transcript", "TTS", "source"]
    train = train.remove_columns([c for c in train.column_names if c not in keep])
    return train


def load_val_orig_only(val_csv, processor):
    val = load_dataset("csv", data_files=val_csv, split="train")
    val = val.map(lambda b: {**b, "is_synth": 0, "supcon_id": -1})
    val = val.map(lambda b: process_from_file(b, processor))
    keep = ["input_values", "labels", "supcon_id", "is_synth"]
    val = val.remove_columns([c for c in val.column_names if c not in keep])
    return val


def compute_synthetic_pairing_stats(dataset):
    trans = list(dataset["transcript"])
    is_synth = [safe_int(x, 0) for x in dataset["is_synth"]]

    orig_by_t = defaultdict(list)
    synth_by_t = defaultdict(list)
    orig_indices = []

    for i, t in enumerate(trans):
        if is_synth[i] == 0:
            orig_by_t[t].append(i)
            orig_indices.append(i)
        else:
            synth_by_t[t].append(i)

    eligible_t = [t for t in orig_by_t if len(synth_by_t.get(t, [])) >= 3]
    eligible_set = set(eligible_t)
    exclusive_orig = [i for t, idxs in orig_by_t.items() if t not in eligible_set for i in idxs]

    total_orig = len(orig_indices)
    exclusive_ratio = (len(exclusive_orig) / total_orig) if total_orig > 0 else 0.0

    return {"total_orig": total_orig, "eligible_transcripts": len(eligible_t),
        "exclusive_orig": len(exclusive_orig), "exclusive_ratio": exclusive_ratio}


def suggest_synthetic_schedule(exclusive_ratio):
    if exclusive_ratio < 0.25:
        return 2, 1
    return 1, 1


def build_hybrid_metadata(orig, synth):
    orig_trans = [normalize_text(t) for t in orig["transcript"]]
    orig_sid = [safe_int(s, default=-1) for s in orig["supcon_id"]]
    synth_trans = [normalize_text(t) for t in synth["transcript"]]

    sim_buckets = defaultdict(list)
    unknown_by_t = defaultdict(list)
    synth_by_t = defaultdict(list)

    for i, sid in enumerate(orig_sid):
        if sid == -1:
            unknown_by_t[orig_trans[i]].append(i)
        else:
            sim_buckets[sid].append(i)

    for i, t in enumerate(synth_trans):
        synth_by_t[t].append(i)

    sim_valid_ids = {sid for sid, idxs in sim_buckets.items() if len(idxs) >= 2}
    sim_orig_idx = {i for sid, idxs in sim_buckets.items() if sid in sim_valid_ids for i in idxs}

    tts_eligible_t = {t for t, idxs in unknown_by_t.items() if len(idxs) > 0 and len(synth_by_t.get(t, [])) >= 3}
    tts_orig_idx = {i for t, idxs in unknown_by_t.items() if t in tts_eligible_t for i in idxs}

    all_orig_idx = set(range(len(orig_trans)))
    ctc_orig_idx = all_orig_idx - sim_orig_idx - tts_orig_idx

    tts_label_by_t = {t: TTS_LABEL_OFFSET + stable_text_id(f"TTS::{t}") for t in tts_eligible_t}

    orig_mode_by_idx = {}
    orig_sup_by_idx = {}
    for i in range(len(orig_trans)):
        if i in sim_orig_idx:
            orig_mode_by_idx[i] = "sim"
            orig_sup_by_idx[i] = orig_sid[i]
        elif i in tts_orig_idx:
            t = orig_trans[i]
            orig_mode_by_idx[i] = "tts"
            orig_sup_by_idx[i] = tts_label_by_t[t]
        else:
            orig_mode_by_idx[i] = "ctc"
            orig_sup_by_idx[i] = -1

    synth_keep_idx = [i for i, t in enumerate(synth_trans) if t in tts_eligible_t]

    stats = {"total_orig": len(orig_trans), "sim_orig": len(sim_orig_idx), "tts_orig": len(tts_orig_idx),
        "ctc_orig": len(ctc_orig_idx), "kept_synth": len(synth_keep_idx), "eligible_tts_transcripts": len(tts_eligible_t)}

    return {"orig_mode_by_idx": orig_mode_by_idx, "orig_sup_by_idx": orig_sup_by_idx, "synth_keep_idx": synth_keep_idx, 
            "tts_label_by_t": tts_label_by_t, "stats": stats}


def suggest_three_mode_schedule(stats, cycle_base=6):
    total = max(1, int(stats["total_orig"]))
    raw = {"sim": max(0, int(stats["sim_orig"])), "tts": max(0, int(stats["tts_orig"])), "ctc": max(0, int(stats["ctc_orig"]))}

    schedule = {}
    for mode, cnt in raw.items():
        if cnt > 0:
            schedule[mode] = max(1, int(round(cycle_base * (cnt / total))))

    if len(schedule) == 0:
        schedule = {"ctc": 1}

    cycle_len = sum(schedule.values())
    if cycle_len > 12:
        scale = 12 / cycle_len
        schedule = {k: max(1, int(round(v * scale))) for k, v in schedule.items()}

    return schedule


def load_hybrid_train(orig_csv, synth_csv, processor):
    orig = load_dataset("csv", data_files=orig_csv, split="train")
    synth = load_dataset("csv", data_files=synth_csv, split="train")

    if "supcon_id" not in orig.column_names:
        raise ValueError("Original CSV must contain 'supcon_id' column for hybrid mode.")

    meta = build_hybrid_metadata(orig, synth)

    def add_orig_meta(row, idx):
        row["is_synth"] = 0
        row["mode"] = meta["orig_mode_by_idx"][idx]
        row["supcon_id"] = meta["orig_sup_by_idx"][idx]
        row["source"] = "orig"
        row["TTS"] = "orig"
        return row

    orig = orig.map(add_orig_meta, with_indices=True)

    keep_idx = set(meta["synth_keep_idx"])
    synth = synth.filter(lambda _, idx: idx in keep_idx, with_indices=True)

    if "TTS" not in synth.column_names:
        synth = synth.map(lambda b: {**b, "TTS": "synth"})
    if "source" not in synth.column_names:
        synth = synth.map(lambda b: {**b, "source": "synthetic"})

    def add_synth_meta(row):
        t = normalize_text(row["transcript"])
        row["is_synth"] = 1
        row["mode"] = "tts"
        row["supcon_id"] = meta["tts_label_by_t"].get(t, -1)
        return row

    synth = synth.map(add_synth_meta)

    train = concatenate_datasets([orig, synth])
    train = train.map(lambda b: process_from_file(b, processor))

    keep = ["input_values", "labels", "supcon_id", "is_synth", "transcript", "mode", "TTS", "source"]
    train = train.remove_columns([c for c in train.column_names if c not in keep])

    return train, meta["stats"]

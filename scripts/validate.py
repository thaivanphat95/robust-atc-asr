import argparse
import csv
import os
import string


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(text, lowercase=True, remove_punctuation=True):
    text = str(text).strip()
    if lowercase:
        text = text.lower()
    if remove_punctuation:
        text = text.translate(PUNCT_TABLE)
    return " ".join(text.split())


def read_eval_csv(path_csv):
    rows = []
    with open(path_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"audio_filename", "transcript"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path_csv} missing required columns: {sorted(missing)}")

        for row in reader:
            rows.append({"audio_filename": row["audio_filename"], "transcript": row["transcript"]})
    return rows


def load_audio(path_audio, target_sr=16000):
    import librosa
    import soundfile as sf

    if not os.path.isabs(path_audio):
        path_audio = os.path.join(PROJECT_DIR, path_audio)

    audio, sample_rate = sf.read(path_audio)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    if sample_rate != target_sr:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
    return audio


class Wav2Vec2Validator:
    def __init__(self, model_path, processor_path, decoder="greedy", device=None):
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM

        self.torch = torch
        self.decoder = decoder
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if decoder == "lm":
            self.processor = Wav2Vec2ProcessorWithLM.from_pretrained(processor_path)
        else:
            self.processor = AutoProcessor.from_pretrained(processor_path)

        self.model = Wav2Vec2ForCTC.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def predict(self, path_audio):
        audio = load_audio(path_audio)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)

        with self.torch.no_grad():
            logits = self.model(inputs).logits

        if self.decoder == "lm":
            return self.processor.decode(logits.detach().cpu().numpy()[0]).text

        pred_ids = self.torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(pred_ids)[0]


def evaluate_csv(validator, wer_metric, path_csv, limit=None):
    rows = read_eval_csv(path_csv)
    if limit is not None:
        rows = rows[:limit]

    predictions = []
    references = []
    records = []

    for i, row in enumerate(rows, start=1):
        pred = validator.predict(row["audio_filename"])
        ref = row["transcript"]

        pred_norm = normalize_text(pred)
        ref_norm = normalize_text(ref)

        predictions.append(pred_norm)
        references.append(ref_norm)
        records.append({"audio_filename": row["audio_filename"], "reference": ref_norm, "prediction": pred_norm})

        if i % 100 == 0:
            print(f"  processed {i}/{len(rows)}")

    wer = wer_metric.compute(predictions=predictions, references=references)
    return wer, records


def write_predictions(path_output, all_records):
    os.makedirs(os.path.dirname(os.path.abspath(path_output)), exist_ok=True)
    with open(path_output, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_filename", "reference", "prediction"])
        writer.writeheader()
        writer.writerows(all_records)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a wav2vec2-large CTC model with greedy or 4-gram LM decoding.")
    parser.add_argument("--model-path", required=True, help="Path to the fine-tuned model checkpoint.")
    parser.add_argument("--processor-path", required=True, help="Path to the greedy processor or 4-gram LM processor.")
    parser.add_argument("--decoder", choices=["greedy", "lm"], default="greedy", help="Decode with argmax CTC or a 4-gram LM processor.")
    parser.add_argument("--csv", default="test.csv", help="Evaluation CSV with audio_filename and transcript columns.")
    parser.add_argument("--device", default=None, help="Torch device. Defaults to cuda when available, otherwise cpu.")
    parser.add_argument("--output-csv", default=None, help="Optional path to save per-utterance predictions.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows for quick smoke tests.")
    return parser.parse_args()


def main():
    args = parse_args()

    from evaluate import load

    wer_metric = load("wer")
    validator = Wav2Vec2Validator(args.model_path, args.processor_path, decoder=args.decoder, device=args.device)

    print(f"Evaluating {args.csv}")
    wer, records = evaluate_csv(
        validator,
        wer_metric,
        args.csv,
        limit=args.limit,
    )
    print(f"WER={wer:.4f}")

    if args.output_csv:
        write_predictions(args.output_csv, records)
        print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()

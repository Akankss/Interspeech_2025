import os
import torch
import soundfile as sf
import pandas as pd
import argparse
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def extract_features_from_audio(input_dir, output_csv):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()

    vocab_size = len(processor.tokenizer.get_vocab())
    feature_size = model.config.hidden_size
    print(f"Vocabulary size: {vocab_size}")
    print(f"Feature size (N): {feature_size}\n")

    audio_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_dir)
        for file in files if file.endswith(".flac")
    ]

    print(f"Found {len(audio_files)} audio files.")

    results = []

    for file_path in audio_files:
        try:
            audio_input, sample_rate = sf.read(file_path)
            if sample_rate != 16000:
                print(f"[WARN] {file_path} sample rate {sample_rate} != 16000 Hz, skipping.")
                continue

            inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt")
            with torch.no_grad():
                logits = model(inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)[0]

            # Remove padding token (0) and consecutive duplicates
            non_zero_ids = predicted_ids[predicted_ids != 0]
            unique_ids = torch.unique_consecutive(non_zero_ids).tolist()

            relative_path = os.path.relpath(file_path, input_dir)
            results.append({
                "Filename": relative_path,
                "Data": ",".join(map(str, unique_ids))
            })

            print(f"Processed: {relative_path}")

        except Exception as e:
            print(f"[ERROR] {file_path}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n[✓] Features saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract token sequences from audio using Wav2Vec2-CTC.")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory with .flac files")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the output CSV")

    args = parser.parse_args()
    extract_features_from_audio(args.input_dir, args.output_csv)

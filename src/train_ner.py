"""
NER Model Trainer for Egyptian Arabic Route Extraction.

Supports training and comparing:
- EgyBERT (faisalq/EgyBERT) — Egyptian dialect specialist
- MARBERTv2 (UBC-NLP/MARBERTv2) — Multi-dialect Arabic

Fine-tunes for token classification with BIO labels:
  O, B-ORIGIN, I-ORIGIN, B-DEST, I-DEST

Includes:
- CRF-compatible training
- ONNX export + INT8 quantization
- Evaluation metrics (precision, recall, F1 per entity)
"""

import os
import sys
import json
import time
import argparse
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)

# ============================================================
# Configuration
# ============================================================
LABEL_LIST = ['O', 'B-ORIGIN', 'I-ORIGIN', 'B-DEST', 'I-DEST']
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

MODELS = {
    'egybert': 'faisalq/EgyBERT',
    'marbert': 'UBC-NLP/MARBERTv2',
    'camelbert': 'CAMeL-Lab/bert-base-arabic-camelbert-mix',
}

DEFAULT_CONFIG = {
    'learning_rate': 3e-5,
    'batch_size': 16,
    'num_epochs': 15,
    'max_seq_length': 128,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'gradient_clip': 1.0,
}


# ============================================================
# Dataset
# ============================================================

class NERDataset(Dataset):
    """CoNLL-style NER dataset loader."""

    def __init__(self, filepath: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_conll(filepath)
        print(f"  Loaded {len(self.samples)} samples from {os.path.basename(filepath)}")

    def _load_conll(self, filepath: str) -> list:
        """Load CoNLL-format file (token\\ttag per line, blank line = separator)."""
        samples = []
        tokens = []
        tags = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if tokens:
                        samples.append({'tokens': tokens, 'tags': tags})
                        tokens = []
                        tags = []
                else:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        tokens.append(parts[0])
                        tags.append(parts[1])

        if tokens:
            samples.append({'tokens': tokens, 'tags': tags})

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = sample['tokens']
        tags = sample['tags']

        # Tokenize with subword alignment
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )

        # Align labels with subword tokens
        word_ids = encoding.word_ids()
        label_ids = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # Special tokens
            elif word_id != prev_word_id:
                # First subword of a word — use the tag
                tag = tags[word_id] if word_id < len(tags) else 'O'
                label_ids.append(LABEL2ID.get(tag, 0))
            else:
                # Subsequent subwords — use I- tag or -100
                tag = tags[word_id] if word_id < len(tags) else 'O'
                if tag.startswith('B-'):
                    label_ids.append(LABEL2ID.get('I-' + tag[2:], 0))
                else:
                    label_ids.append(LABEL2ID.get(tag, 0))
            prev_word_id = word_id

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_ids, dtype=torch.long),
        }


# ============================================================
# Evaluation
# ============================================================

def evaluate_ner(model, dataloader, device):
    """Evaluate NER model and return per-entity metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

            for pred_seq, label_seq, mask_seq in zip(predictions, labels, attention_mask):
                for pred, label, mask in zip(pred_seq, label_seq, mask_seq):
                    if mask.item() == 1 and label.item() != -100:
                        all_preds.append(ID2LABEL[pred.item()])
                        all_labels.append(ID2LABEL[label.item()])

    # Compute metrics
    metrics = compute_entity_metrics(all_labels, all_preds)
    return metrics


def compute_entity_metrics(true_labels, pred_labels):
    """Compute precision, recall, F1 for each entity type."""
    entity_types = ['ORIGIN', 'DEST']
    results = {}

    for entity in entity_types:
        tp = sum(1 for t, p in zip(true_labels, pred_labels)
                 if t.endswith(entity) and p.endswith(entity))
        fp = sum(1 for t, p in zip(true_labels, pred_labels)
                 if not t.endswith(entity) and p.endswith(entity))
        fn = sum(1 for t, p in zip(true_labels, pred_labels)
                 if t.endswith(entity) and not p.endswith(entity))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[entity] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'support': tp + fn,
        }

    # Overall
    total_tp = sum(r['precision'] * r['support'] for r in results.values())
    total_support = sum(r['support'] for r in results.values())
    overall_p = total_tp / total_support if total_support > 0 else 0

    all_tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    accuracy = all_tp / len(true_labels) if true_labels else 0

    results['overall'] = {
        'accuracy': round(accuracy, 4),
        'macro_f1': round(np.mean([r['f1'] for r in results.values() if 'f1' in r]), 4),
    }

    return results


# ============================================================
# Training
# ============================================================

def train_model(model_key: str, data_dir: str, output_dir: str, config: dict = None):
    """Train a NER model and save results."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    # Use mapping if available, otherwise assume model_key is a full model ID
    model_name = MODELS.get(model_key, model_key)

    print(f"\n{'='*60}")
    print(f"Training: {model_key.upper()} ({model_name})")
    print(f"{'='*60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load tokenizer and model
    print(f"Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    # Load datasets
    print(f"Loading datasets from {data_dir}...")
    train_dataset = NERDataset(
        os.path.join(data_dir, 'train.txt'), tokenizer, cfg['max_seq_length']
    )
    val_dataset = NERDataset(
        os.path.join(data_dir, 'val.txt'), tokenizer, cfg['max_seq_length']
    )
    test_dataset = NERDataset(
        os.path.join(data_dir, 'test.txt'), tokenizer, cfg['max_seq_length']
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'])

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=cfg['learning_rate'],
        weight_decay=cfg['weight_decay'],
    )

    total_steps = len(train_loader) * cfg['num_epochs']
    warmup_steps = int(total_steps * cfg['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    best_val_f1 = 0.0
    training_history = []

    for epoch in range(cfg['num_epochs']):
        model.train()
        total_loss = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time

        # Validate
        val_metrics = evaluate_ner(model, val_loader, device)
        val_f1 = val_metrics['overall']['macro_f1']

        print(f"  Epoch {epoch+1}/{cfg['num_epochs']} | "
              f"Loss: {avg_loss:.4f} | "
              f"Val F1: {val_f1:.4f} | "
              f"Time: {epoch_time:.1f}s")

        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'val_f1': val_f1,
            'val_metrics': val_metrics,
        })

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_dir = os.path.join(output_dir, f'{model_key}_best')
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"    ✓ New best model saved (F1: {val_f1:.4f})")

    # Final evaluation on test set
    print(f"\nFinal Test Evaluation ({model_key}):")
    # Load best model for test eval
    best_model = AutoModelForTokenClassification.from_pretrained(
        os.path.join(output_dir, f'{model_key}_best')
    ).to(device)
    test_metrics = evaluate_ner(best_model, test_loader, device)

    for entity, metrics in test_metrics.items():
        if entity == 'overall':
            print(f"  OVERALL — Accuracy: {metrics['accuracy']:.4f}, Macro F1: {metrics['macro_f1']:.4f}")
        else:
            print(f"  {entity} — P: {metrics['precision']:.4f}, R: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")

    # Save results
    results = {
        'model_key': model_key,
        'model_name': model_name,
        'config': cfg,
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics,
        'training_history': training_history,
    }

    results_path = os.path.join(output_dir, f'{model_key}_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {results_path}")
    return results


# ============================================================
# ONNX Export
# ============================================================

def export_to_onnx(model_dir: str, output_path: str, quantize: bool = True):
    """Export trained model to ONNX format with optional INT8 quantization."""
    print(f"\nExporting to ONNX: {model_dir} → {output_path}")

    try:
        from optimum.onnxruntime import ORTModelForTokenClassification
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        from optimum.onnxruntime import ORTQuantizer

        # Export to ONNX
        ort_model = ORTModelForTokenClassification.from_pretrained(
            model_dir, export=True
        )
        ort_model.save_pretrained(output_path)
        print(f"  ✓ ONNX model exported")

        if quantize:
            # INT8 dynamic quantization
            quantizer = ORTQuantizer.from_pretrained(output_path)
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
            quantizer.quantize(save_dir=output_path + '_int8', quantization_config=qconfig)
            print(f"  ✓ INT8 quantized model saved to {output_path}_int8")

    except ImportError:
        print("  ⚠ optimum not installed. Install with: pip install optimum[onnxruntime]")
        print("  Falling back to manual ONNX export...")

        # Manual fallback
        from transformers import AutoTokenizer, AutoModelForTokenClassification

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        model.eval()

        dummy = tokenizer("اختبار", return_tensors="pt")
        os.makedirs(output_path, exist_ok=True)
        onnx_path = os.path.join(output_path, "model.onnx")

        torch.onnx.export(
            model,
            (dummy['input_ids'], dummy['attention_mask']),
            onnx_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'sequence'},
                'attention_mask': {0: 'batch', 1: 'sequence'},
                'logits': {0: 'batch', 1: 'sequence'},
            },
            opset_version=14,
        )
        tokenizer.save_pretrained(output_path)
        print(f"  ✓ Manual ONNX export complete: {onnx_path}")


# ============================================================
# Comparison
# ============================================================

def compare_models(results: list):
    """Print a comparison table of model results."""
    print(f"\n{'='*70}")
    print(f"MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'Val F1':<10} {'ORIGIN F1':<12} {'DEST F1':<12} {'Accuracy':<10}")
    print(f"{'-'*70}")

    for r in results:
        name = r['model_key'].upper()
        val_f1 = r['best_val_f1']
        origin_f1 = r['test_metrics'].get('ORIGIN', {}).get('f1', 0)
        dest_f1 = r['test_metrics'].get('DEST', {}).get('f1', 0)
        accuracy = r['test_metrics'].get('overall', {}).get('accuracy', 0)
        print(f"{name:<15} {val_f1:<10.4f} {origin_f1:<12.4f} {dest_f1:<12.4f} {accuracy:<10.4f}")

    # Declare winner
    best = max(results, key=lambda r: r['best_val_f1'])
    print(f"\n🏆 Winner: {best['model_key'].upper()} (Val F1: {best['best_val_f1']:.4f})")
    return best['model_key']


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Train NER models for Arabic route extraction')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with train/val/test .txt files')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for models')
    parser.add_argument('--models', type=str, nargs='+', default=['egybert', 'marbert'],
                        help='Models to train (shorthand: egybert, marbert, camelbert or full HF ID)')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--export-onnx', action='store_true', help='Export best model to ONNX')
    parser.add_argument('--quantize', action='store_true', help='Apply INT8 quantization')

    args = parser.parse_args()

    config = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
    }

    os.makedirs(args.output_dir, exist_ok=True)

    # Train each model
    all_results = []
    for model_key in args.models:
        result = train_model(model_key, args.data_dir, args.output_dir, config)
        all_results.append(result)

    # Compare
    if len(all_results) > 1:
        winner_key = compare_models(all_results)
    else:
        winner_key = all_results[0]['model_key']

    # Export winner to ONNX
    if args.export_onnx:
        best_model_dir = os.path.join(args.output_dir, f'{winner_key}_best')
        onnx_dir = os.path.join(args.output_dir, f'{winner_key}_onnx')
        export_to_onnx(best_model_dir, onnx_dir, quantize=args.quantize)

    print(f"\n✅ Training complete. Winner: {winner_key.upper()}")
    print(f"   Best model: {os.path.join(args.output_dir, f'{winner_key}_best')}")


if __name__ == '__main__':
    main()

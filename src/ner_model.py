"""
NER Model Inference Wrapper for Route Extraction.

Supports:
- PyTorch model inference
- ONNX Runtime inference (faster, quantized)
- Automatic model loading with fallback
"""

import os
import numpy as np
from typing import List, Tuple, Optional

# ============================================================
# Label Configuration
# ============================================================
LABEL_LIST = ['O', 'B-ORIGIN', 'I-ORIGIN', 'B-DEST', 'I-DEST']
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}


class NERModel:
    """
    NER model wrapper supporting PyTorch and ONNX inference.
    
    Usage:
        model = NERModel("path/to/model")
        entities = model.predict("من مدينة نصر الى المعادي")
        # Returns: {'origin': 'مدينة نصر', 'origin_score': 0.95, 
        #           'destination': 'المعادي', 'destination_score': 0.92}
    """

    def __init__(self, model_path: str, use_onnx: bool = False):
        """
        Load NER model.
        
        Args:
            model_path: Path to saved model directory
            use_onnx: If True, load ONNX model (faster inference)
        """
        self.model_path = model_path
        self.use_onnx = use_onnx
        self.model = None
        self.tokenizer = None
        self.ort_session = None

        if use_onnx:
            self._load_onnx(model_path)
        else:
            self._load_pytorch(model_path)

    def _load_pytorch(self, model_path: str):
        """Load PyTorch model."""
        import torch
        from transformers import AutoTokenizer, AutoModelForTokenClassification

        print(f"Loading NER model (PyTorch): {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        self.device = torch.device('cpu')  # CPU for production
        self.model.to(self.device)
        print("NER model loaded successfully (PyTorch)")

    def _load_onnx(self, model_path: str):
        """Load ONNX model for faster inference."""
        from transformers import AutoTokenizer

        print(f"Loading NER model (ONNX): {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Try optimum first, fall back to onnxruntime
        try:
            from optimum.onnxruntime import ORTModelForTokenClassification
            self.ort_model = ORTModelForTokenClassification.from_pretrained(model_path)
            self._onnx_method = 'optimum'
        except ImportError:
            import onnxruntime as ort
            onnx_path = os.path.join(model_path, 'model.onnx')
            self.ort_session = ort.InferenceSession(
                onnx_path,
                providers=['CPUExecutionProvider'],
            )
            self._onnx_method = 'onnxruntime'

        print(f"NER model loaded successfully (ONNX via {self._onnx_method})")

    def predict(self, text: str) -> dict:
        """
        Run NER prediction on text.
        
        Returns dict with:
        - origin: extracted origin text (or None)
        - origin_score: confidence score (0-1)
        - destination: extracted destination text (or None)
        - destination_score: confidence score (0-1)
        """
        if not text or not text.strip():
            return {'origin': None, 'origin_score': 0, 'destination': None, 'destination_score': 0}

        tokens = text.split()

        if self.use_onnx:
            predicted_labels, scores = self._predict_onnx(tokens)
        else:
            predicted_labels, scores = self._predict_pytorch(tokens)

        # Extract entity spans from BIO labels
        origin, origin_score = self._extract_entity(tokens, predicted_labels, scores, 'ORIGIN')
        destination, dest_score = self._extract_entity(tokens, predicted_labels, scores, 'DEST')

        return {
            'origin': origin,
            'origin_score': origin_score,
            'destination': destination,
            'destination_score': dest_score,
        }

    def _predict_pytorch(self, tokens: List[str]) -> Tuple[List[str], List[float]]:
        """Run PyTorch inference."""
        import torch

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors='pt',
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding['input_ids'].to(self.device),
                attention_mask=encoding['attention_mask'].to(self.device),
            )

        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)

        # Align predictions back to original tokens
        word_ids = encoding.word_ids()
        labels = []
        scores = []
        prev_word_id = None

        for idx, word_id in enumerate(word_ids):
            if word_id is not None and word_id != prev_word_id:
                pred_id = predictions[idx].item()
                labels.append(ID2LABEL[pred_id])
                scores.append(probs[idx][pred_id].item())
            prev_word_id = word_id

        return labels[:len(tokens)], scores[:len(tokens)]

    def _predict_onnx(self, tokens: List[str]) -> Tuple[List[str], List[float]]:
        """Run ONNX inference."""
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors='np' if self._onnx_method == 'onnxruntime' else 'pt',
        )

        if self._onnx_method == 'optimum':
            inputs = {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
            }
            if 'token_type_ids' in encoding:
                inputs['token_type_ids'] = encoding['token_type_ids']
            outputs = self.ort_model(**inputs)
            logits = outputs.logits[0].detach().numpy()
        else:
            ort_inputs = {
                'input_ids': encoding['input_ids'].astype(np.int64),
                'attention_mask': encoding['attention_mask'].astype(np.int64),
            }
            if 'token_type_ids' in encoding:
                ort_inputs['token_type_ids'] = encoding['token_type_ids'].astype(np.int64)
            logits = self.ort_session.run(None, ort_inputs)[0][0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        predictions = np.argmax(logits, axis=-1)

        # Align predictions back to original tokens
        word_ids = encoding.word_ids()
        labels = []
        scores = []
        prev_word_id = None

        for idx, word_id in enumerate(word_ids):
            if word_id is not None and word_id != prev_word_id:
                pred_id = int(predictions[idx])
                labels.append(ID2LABEL[pred_id])
                scores.append(float(probs[idx][pred_id]))
            prev_word_id = word_id

        return labels[:len(tokens)], scores[:len(tokens)]

    def _extract_entity(self, tokens: List[str], labels: List[str],
                        scores: List[float], entity_type: str) -> Tuple[Optional[str], float]:
        """Extract entity span from BIO-tagged tokens."""
        entity_tokens = []
        entity_scores = []
        in_entity = False

        for token, label, score in zip(tokens, labels, scores):
            if label == f'B-{entity_type}':
                entity_tokens = [token]
                entity_scores = [score]
                in_entity = True
            elif label == f'I-{entity_type}' and in_entity:
                entity_tokens.append(token)
                entity_scores.append(score)
            else:
                if in_entity:
                    break  # End of entity
                in_entity = False

        if entity_tokens:
            entity_text = ' '.join(entity_tokens)
            avg_score = sum(entity_scores) / len(entity_scores)
            return entity_text, avg_score

        return None, 0.0

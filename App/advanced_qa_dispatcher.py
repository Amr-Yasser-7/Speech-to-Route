from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import re

# =============================================================================
# MODEL SELECTION: AraElectra for Extractive Question Answering
# =============================================================================
WHISPER_QA_MODEL = "ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA"

try:
    print(f"Loading Advanced QA Transformer... ({WHISPER_QA_MODEL})")
    tokenizer = AutoTokenizer.from_pretrained(WHISPER_QA_MODEL)
    model = AutoModelForQuestionAnswering.from_pretrained(WHISPER_QA_MODEL)
    print("QA Transformer loaded successfully.")
except Exception as e:
    print(f"Failed to load QA Transformer: {e}")
    model = None
    tokenizer = None

def get_qa_answer(question, context):
    """Manual QA implementation to bypass pipeline issues."""
    if not model or not tokenizer: return {"answer": "", "score": 0}
    
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    # Simple score based on logit value
    score = float(torch.max(torch.softmax(answer_start_scores, dim=-1)))
    
    return {"answer": answer, "score": score}

def clean_extracted_answer(text: str) -> str:
    """Removes common Egyptian Arabic conversational noise from the AI's answer."""
    if not text or "[CLS]" in text or "[SEP]" in text: return ""
    
    noise = [
        "لو سمحت", "من فضلك", "يا ريس", "يا باشا", "يا حج", "يا كابتن",
        "عايز اروح", "محتاج اوصل", "بسرعة", "دلوقتي", "ممكن", "خدني"
    ]
    
    cleaned = text.strip()
    cleaned = re.sub(r'^[لب]ـ?', '', cleaned)
    
    for n in noise:
        cleaned = cleaned.replace(n, "")
    
    cleaned = re.sub(r'[\s,،.!-]+', ' ', cleaned).strip()
    return cleaned

def advanced_extract_route(text: str) -> dict:
    """Advanced Extractive QA Dispatcher."""
    if not text or not model:
        return {"origin": None, "origin_score": 0, "destination": None, "destination_score": 0}

    text_norm = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")

    ans_origin = get_qa_answer("من أي مكان ستبدأ الرحلة؟", text_norm)
    ans_dest = get_qa_answer("إلى أين تريد الذهاب؟", text_norm)
    
    result = {
        "origin": clean_extracted_answer(ans_origin['answer']),
        "origin_score": ans_origin['score'],
        "destination": clean_extracted_answer(ans_dest['answer']),
        "destination_score": ans_dest['score']
    }

    # Post-validation: If scores are extremely low, or result is trivial
    if result["origin_score"] < 0.05 or len(str(result["origin"])) < 2:
        result["origin"] = None
    if result["destination_score"] < 0.05 or len(str(result["destination"])) < 2:
        result["destination"] = None

    return result


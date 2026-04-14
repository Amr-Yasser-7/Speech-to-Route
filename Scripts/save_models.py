import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModelForQuestionAnswering

# Define local paths
whisper_path = "./models/whisper_egy"
qa_path = "./models/qa_brain"

# Create directories
os.makedirs(whisper_path, exist_ok=True)
os.makedirs(qa_path, exist_ok=True)

print("Starting model backup to your project folder...")

# 1. Save Whisper Model
print("Saving Whisper model...")
WHISPER_ID = "itshamdi404/Egy_Arabic_whisper-small"
proc = WhisperProcessor.from_pretrained(WHISPER_ID)
model = WhisperForConditionalGeneration.from_pretrained(WHISPER_ID)
proc.save_pretrained(whisper_path)
model.save_pretrained(whisper_path)

# 2. Save QA Brain
print("Saving QA Brain...")
QA_ID = "ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA"
tok = AutoTokenizer.from_pretrained(QA_ID)
qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_ID)
tok.save_pretrained(qa_path)
qa_model.save_pretrained(qa_path)

# 3. Save NER Model (CAMeL-Lab)
print("Saving NER Model...")
from transformers import pipeline, AutoModelForTokenClassification
NER_ID = "CAMeL-Lab/bert-base-arabic-camelbert-da-ner"
ner_path = "./models/ner_brain"
os.makedirs(ner_path, exist_ok=True)
ner_tok = AutoTokenizer.from_pretrained(NER_ID)
ner_model = AutoModelForTokenClassification.from_pretrained(NER_ID)
ner_tok.save_pretrained(ner_path)
ner_model.save_pretrained(ner_path)

print(f"\n✅ Models saved successfully in: {os.path.abspath('./models')}")
print("You can now push the 'models' folder to GitHub (if you use Git LFS).")

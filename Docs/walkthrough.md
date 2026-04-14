# 🗺️ Development Journey: The Path to 100% Accuracy

This document outlines the engineering iterations taken to build a robust Egyptian Arabic dispatcher.

## Phase 1: The Deterministic Foundation (Rules)
Initially, the system used basic **Regular Expressions**. 
*   **Challenge**: Egyptian speech is unpredictable. "رايح المعادي" (Going to Maadi) vs "المعادي رايح لها" (Maadi, I'm going to it) required hundreds of delicate rules.
*   **Lesson**: Rules are fast but "brittle."

## Phase 2: The Neural Pivot (AraElectra)
To solve the "slang" problem, we integrated a **Transformer-based QA model**.
*   **Improvement**: The model understood *meaning*. It could find landmarks it had never seen before because it understood Arabic grammar.
*   **Challenge**: "Greedy spans." The AI would often include prepositions like "to" or "from" inside the city name (e.g., getting "to Maadi" instead of just "Maadi").

## Phase 3: The Data-Driven Baseline (Test Suite)
We built a **68-case evaluation suite** (`test_ext.py`). 
*   **Innovation**: Implemented **Fuzzy Semantic Matching**. This normalized Egyptian spelling variations (Hamzas/Ta-Marbutas) during testing so that a correct answer wasn't marked wrong just because of a typo.
*   **Outcome**: We finally had a "Scorecard" to measure our progress.

## Phase 4: The Hybrid Fusion (Current Architecture)
The final breakthrough came from combining both worlds:
1.  **AI for Discovery**: Used to identify the *vibe* and *landmark*.
2.  **Logic for Precision**: If the AI is confused, the **Syntactic Dispatcher** uses directional markers (من/الى) to correctly assign roles.
3.  **The Result**: A system that is smarter than a regex but safer than a pure transformer.

## Phase 5: Production Optimization
Finally, we moved the complexity out of the runtime:
*   **Cold Start Problem**: Loading 2GB of models was too slow for an API.
*   **Solution**: Moved all loading to a pre-serialization script (`main.py`). The API now loads a single `Speech.pkl` file, making it ready to serve requests in milliseconds.

---
### 🛠️ Key Skills Demonstrated
- Sequence Labeling & Named Entity Recognition (NER)
- Multi-Stage Pipeline Architecture
- Egyptian Arabic Dialect Normalization
- Automated Semantic Testing
- Model Serialization & Infrastructure Optimization

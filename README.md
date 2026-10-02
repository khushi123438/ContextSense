# 🧠 ContextSense

ContextSense is an educational NLP application that demonstrates how a
Transformer model processes text using **DistilBERT**, contextual
representations, and self-attention.

## Features

- DistilBERT-based sentiment classification
- Hugging Face Transformers
- Tokenization visualization
- Self-attention extraction
- Multi-head attention averaging
- Interactive attention heatmap
- Token-level attention scores
- Streamlit interface
- PyTorch inference

## Architecture

```text
Input Text
    ↓
DistilBERT Tokenizer
    ↓
Token IDs
    ↓
DistilBERT Embeddings
    ↓
Transformer Self-Attention
    ↓
Transformer Layers
    ↓
Contextual Representations
    ↓
Classification Head
    ↓
Positive / Negative + Confidence
```

## Attention Visualization

ContextSense extracts attention weights from the final DistilBERT
Transformer layer and averages the attention across all attention heads.

```text
DistilBERT
   ↓
Final Transformer Layer
   ↓
Attention Heads
   ↓
Average Across Heads
   ↓
Attention Matrix
   ↓
Interactive Heatmap
```

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- DistilBERT
- Streamlit
- Pandas
- Plotly

## Model

The project uses:

`distilbert-base-uncased-finetuned-sst-2-english`

This is a DistilBERT model fine-tuned for binary sentiment classification
on the SST-2 task.

## Run Locally

### 1. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

The first run downloads the pretrained DistilBERT model from Hugging Face.

## Project Structure

```text
ContextSense/
│
├── app.py
├── requirements.txt
├── README.md
│
└── src/
    ├── __init__.py
    ├── model.py
    ├── analyzer.py
    └── visualization.py
```

## Note

The attention visualization is intended for learning and interpretability.
Attention weights should not automatically be treated as a definitive
explanation of why a model made a prediction.

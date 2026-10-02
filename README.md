# 🧠 ContextSense

### Context-Aware Text Understanding using Transformers

ContextSense is a mini NLP project built to explore how Transformer-based models understand text through **tokenization, contextual representations and self-attention**.

Instead of treating a Transformer as a black box, this project provides a simple interface to observe parts of the processing pipeline behind a pretrained Transformer model.

---

## 🚀 Features

* Transformer-based sentiment analysis
* Text tokenization visualization
* Contextual text understanding
* Self-attention visualization
* Attention heatmap
* Token-level attention scores
* Sentiment confidence score
* Interactive Streamlit interface
* Simple explanation of the Transformer pipeline

---

## 🧠 Transformer Pipeline

```text
User Input
    ↓
Tokenization
    ↓
Token Embeddings
    ↓
Transformer Layers
    ↓
Self-Attention
    ↓
Contextual Representation
    ↓
Classification
    ↓
Sentiment + Confidence
```

---

## 🔍 How It Works

### 1. Tokenization

The input sentence is divided into tokens using the tokenizer associated with the pretrained Transformer model.

Example:

```text
"The movie was amazing!"

        ↓

["the", "movie", "was", "amazing", "!"]
```

---

### 2. Embeddings

Each token is converted into a numerical representation that the Transformer can process.

---

### 3. Self-Attention

The Transformer calculates relationships between tokens.

For example:

```text
"The movie was surprisingly good"
```

The model can use relationships between words such as:

```text
movie ↔ good
surprisingly ↔ good
```

to build contextual representations.

---

### 4. Contextual Representation

Unlike static word representations, Transformer-based representations can change depending on the surrounding context.

For example:

```text
I deposited money at the bank.

The boat reached the river bank.
```

The word **bank** appears in different contexts, so its representation can differ.

---

### 5. Classification

The contextual representation is passed through the classification layer to predict the sentiment.

---

## 📊 Attention Visualization

ContextSense extracts attention weights from the final Transformer layer and averages the attention across its attention heads.

The resulting matrix is visualized as a heatmap.

This provides an intuitive way to explore token-to-token attention patterns.

> Note: Attention visualization should be treated as an interpretability aid rather than a definitive explanation of why the model made a prediction.

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* Streamlit
* Pandas
* NumPy
* Plotly

---

## 🤖 Model

This project uses:

```text
distilbert-base-uncased-finetuned-sst-2-english
```

DistilBERT is a smaller Transformer-based model derived from BERT and fine-tuned for sentiment classification.

---

## 📁 Project Structure

```text
ContextSense/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── sample_sentiment.csv
│
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── analyzer.py
│   └── visualization.py
│
└── assets/
    └── architecture.png
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ContextSense.git
```

Move into the project:

```bash
cd ContextSense
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🧪 Example

Input:

```text
The movie was surprisingly good!
```

The application provides:

```text
Sentiment: POSITIVE

Confidence: ~99%

Tokens:
[CLS] the movie was surprisingly good ! [SEP]
```

It also displays the Transformer attention heatmap and token-level attention scores.

---

## 📚 Concepts Explored

Through this project, I explored:

* Tokenization
* Token Embeddings
* Contextual Embeddings
* Self-Attention
* Multi-Head Attention
* Transformer Layers
* Attention Weights
* Sequence Classification
* Transfer Learning
* Transformer-based NLP

---

## 🔮 Future Improvements

* Implement Multi-Head Self-Attention from scratch
* Add manual Q, K and V visualization
* Compare BERT and DistilBERT
* Add multiple NLP tasks
* Add attention comparison across different layers
* Add static vs contextual embedding comparison
* Support custom datasets
* Add model comparison
* Deploy the application

---

## 🎯 Learning Objective

The main goal of ContextSense is not simply to perform sentiment classification.

It is to understand what happens inside a Transformer-based NLP pipeline and make concepts such as **tokenization, contextual representations and self-attention** easier to visualize.

---

## 👩‍💻 Author

**Khushi Pandey**

B.Tech CSE (AI)

---

## ⭐ If you found this project useful

Feel free to explore, modify and extend the project to experiment with Transformer-based NLP systems.

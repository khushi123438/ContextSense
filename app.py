import streamlit as st
import pandas as pd

from src.model import TransformerModel
from src.analyzer import get_attention_matrix, get_token_scores
from src.visualization import create_attention_heatmap

st.set_page_config(
    page_title="ContextSense",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}
.subtitle {
    font-size: 18px;
    color: #777;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return TransformerModel()

transformer = load_model()

st.markdown(
    '<div class="main-title">🧠 ContextSense</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="subtitle">
Explore how DistilBERT understands text using contextual representations
and self-attention.
</div>
""", unsafe_allow_html=True)

st.sidebar.title("About")
st.sidebar.write(
    "ContextSense is an educational NLP application that visualizes "
    "how a DistilBERT Transformer processes text."
)

st.sidebar.markdown("""
### Pipeline

Tokenization  
↓  
Embeddings  
↓  
Self-Attention  
↓  
Transformer Layers  
↓  
Contextual Representation  
↓  
Sentiment Classification
""")

st.sidebar.markdown("""
### Model

**DistilBERT**  
Pre-trained Transformer model  
Fine-tuned for sentiment classification
""")

st.subheader("Enter Text")

text = st.text_area(
    "Input sentence",
    value="The movie was surprisingly good!",
    height=120
)

analyze = st.button("🔍 Analyze Text", use_container_width=True)

if analyze:
    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    with st.spinner("Running DistilBERT analysis..."):
        result = transformer.predict(text)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Sentiment", result["sentiment"])

    with col2:
        st.metric("Confidence", f"{result['confidence']:.2f}%")

    tokens = transformer.get_tokens(result["inputs"])

    st.subheader("🔤 Tokenization")
    st.write(tokens)

    attention_matrix = get_attention_matrix(result["outputs"])

    st.subheader("👀 Self-Attention Visualization")
    st.write(
        "The heatmap shows how strongly tokens attend to one another "
        "in the selected final DistilBERT layer."
    )

    figure = create_attention_heatmap(attention_matrix, tokens)

    st.plotly_chart(
        figure,
        use_container_width=True
    )

    st.subheader("⭐ Token Attention Scores")

    token_scores = get_token_scores(attention_matrix, tokens)
    dataframe = pd.DataFrame(token_scores)

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("🧩 What happened internally?")

    st.markdown("""
**1. Tokenization**  
The input sentence was split into subword tokens.

**2. Embeddings**  
Tokens were converted into numerical representations.

**3. Self-Attention**  
DistilBERT calculated relationships between tokens.

**4. Contextual Representation**  
Each token representation was influenced by surrounding context.

**5. Classification**  
The final representation was used to predict positive or negative sentiment.
""")

st.divider()
st.caption(
    "ContextSense • DistilBERT + PyTorch + Hugging Face Transformers + Streamlit"
)

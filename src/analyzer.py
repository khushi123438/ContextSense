import torch


def get_attention_matrix(outputs):
    """
    Extracts and averages the attention matrix across all heads
    from the final Transformer layer.

    Returns:
        Tensor with shape [sequence_length, sequence_length]
    """

    if not hasattr(outputs, "attentions") or outputs.attentions is None:
        raise ValueError("Attention outputs are not available.")

    # DistilBERT attentions:
    # tuple[num_layers] -> [batch, num_heads, seq_len, seq_len]

    final_layer_attention = outputs.attentions[-1][0]

    # Average attention across heads
    attention_matrix = final_layer_attention.mean(dim=0)

    return attention_matrix.detach().cpu()


def get_token_scores(attention_matrix, tokens):
    """
    Computes a simple token-level importance score by averaging
    how much attention each token receives from other tokens.
    """

    matrix = attention_matrix.numpy()

    # Average attention received by each token
    scores = matrix.mean(axis=0)

    results = []

    for token, score in zip(tokens, scores):
        results.append({
            "Token": token,
            "Attention Score": round(float(score), 6)
        })

    results.sort(
        key=lambda item: item["Attention Score"],
        reverse=True
    )

    return results

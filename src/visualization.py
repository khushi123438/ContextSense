import plotly.graph_objects as go


def create_attention_heatmap(attention_matrix, tokens):
    """
    Creates an interactive Plotly heatmap for token-to-token attention.
    """

    matrix = attention_matrix.numpy()

    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=tokens,
            y=tokens,
            colorscale="Viridis",
            colorbar=dict(
                title="Attention"
            ),
            hovertemplate=(
                "From: %{y}<br>"
                "To: %{x}<br>"
                "Attention: %{z:.4f}"
                "<extra></extra>"
            )
        )
    )

    figure.update_layout(
        title="DistilBERT Self-Attention — Final Layer",
        xaxis_title="Tokens Being Attended To",
        yaxis_title="Tokens Providing Attention",
        height=650,
        margin=dict(
            l=40,
            r=40,
            t=70,
            b=120
        )
    )

    return figure

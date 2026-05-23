"""Streamlit dashboard for Amazon Electronics sentiment analysis.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
LABELED_CSV = BASE_DIR / "data" / "processed" / "amazon_reviews_2023_electronics_labeled.csv"
COMPARISON_CSV = BASE_DIR / "outputs" / "model_comparison" / "model_comparison.csv"
RANKING_CSV = BASE_DIR / "outputs" / "amazon_reviews_2023" / "amazon_reviews_2023_product_ranking.csv"
BERT_MODEL_DIR = BASE_DIR / "models" / "bert"

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL_ORDER = ["Negative", "Neutral", "Positive"]
EXAMPLE_REVIEWS = {
    "positive": (
        "This speaker exceeded my expectations. The sound is crisp, the battery lasts all day, "
        "and setup was incredibly easy."
    ),
    "negative": (
        "Terrible headphones. The right side stopped working after two days and the battery life is awful."
    ),
    "ambiguous": (
        "The tablet is okay for basic tasks, but performance feels average and the display could be better."
    ),
}


st.set_page_config(page_title="Dashboard cảm xúc Amazon Electronics", layout="wide")


def inject_styles() -> None:
    """Add a small visual layer so the dashboard feels more polished."""
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top, rgba(217, 249, 157, 0.45), transparent 28%),
                linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(226, 232, 240, 0.95);
            padding: 1rem;
            border-radius: 1rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
        }
        div[data-testid="stDataFrame"],
        div[data-testid="stPlotlyChart"] {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(226, 232, 240, 0.95);
            border-radius: 1rem;
            padding: 0.35rem;
        }
        .dashboard-note {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(226, 232, 240, 0.95);
            border-radius: 1rem;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            color: #334155;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file once and return an empty frame if it is missing."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_bert_resources(model_dir: Path) -> tuple[Any | None, Any | None, str | None]:
    """Load tokenizer and model once, returning a warning message on failure."""
    if not model_dir.exists():
        return None, None, f"Không tìm thấy checkpoint BERT tại `{model_dir}`."

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        return tokenizer, model, None
    except Exception as exc:  # pragma: no cover - defensive UI path
        return None, None, f"Không thể load checkpoint BERT: {exc}"


def safe_percentage(series: pd.Series, positive_value: int | str) -> float:
    """Compute a percentage safely for summary metrics."""
    if series.empty:
        return 0.0
    return float((series == positive_value).mean() * 100.0)


def infer_with_bert(review_text: str) -> tuple[str, list[float]]:
    """Run BERT inference when a checkpoint is available."""
    tokenizer, model, warning_message = load_bert_resources(BERT_MODEL_DIR)
    if tokenizer is None or model is None:
        raise RuntimeError(warning_message or "BERT model is not available.")

    encoded = tokenizer(
        review_text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**encoded)
        probabilities = torch.softmax(outputs.logits, dim=-1).squeeze(0).tolist()

    predicted_index = int(torch.tensor(probabilities).argmax().item())
    return LABEL_MAP[predicted_index], [float(value) for value in probabilities]


def infer_with_fallback(review_text: str) -> tuple[str, list[float]]:
    """Use a transparent keyword-based fallback so the UI remains demoable."""
    text = review_text.lower()
    positive_keywords = [
        "excellent",
        "great",
        "amazing",
        "perfect",
        "love",
        "good",
        "satisfied",
        "easy",
        "fast",
        "reliable",
    ]
    negative_keywords = [
        "bad",
        "terrible",
        "awful",
        "broken",
        "poor",
        "hate",
        "disappointing",
        "slow",
        "worse",
        "problem",
    ]

    positive_score = sum(keyword in text for keyword in positive_keywords)
    negative_score = sum(keyword in text for keyword in negative_keywords)
    neutral_score = 1.0

    raw_scores = torch.tensor(
        [
            1.0 + float(negative_score) * 1.6,
            neutral_score + (0.3 if positive_score == negative_score else 0.0),
            1.0 + float(positive_score) * 1.6,
        ],
        dtype=torch.float32,
    )
    probabilities = torch.softmax(raw_scores, dim=0).tolist()
    predicted_index = int(torch.argmax(raw_scores).item())
    return LABEL_MAP[predicted_index], [float(value) for value in probabilities]


def build_probability_chart(probabilities: list[float]) -> Any:
    """Create a horizontal probability bar chart."""
    chart_df = pd.DataFrame(
        {
            "Nhãn": LABEL_ORDER,
            "Xác suất": probabilities,
        }
    )
    return px.bar(
        chart_df,
        x="Xác suất",
        y="Nhãn",
        orientation="h",
        color="Nhãn",
        color_discrete_map={
            "Negative": "#ef4444",
            "Neutral": "#f59e0b",
            "Positive": "#22c55e",
        },
        text="Xác suất",
    ).update_traces(texttemplate="%{text:.2%}", textposition="outside").update_layout(
        height=300,
        showlegend=False,
        xaxis_tickformat=".0%",
        margin=dict(l=10, r=10, t=20, b=10),
    )


def render_missing_file(path: Path, label: str) -> None:
    """Render a consistent warning for missing optional data files."""
    st.warning(f"Không tìm thấy {label}: `{path}`. Tab này sẽ hiển thị ở chế độ tối giản.")


def render_eda() -> None:
    """Render the EDA tab."""
    st.subheader("Tổng quan dữ liệu review")
    st.markdown(
        "<div class='dashboard-note'>Tab này cho bạn cái nhìn nhanh về cấu trúc dữ liệu, phân bố nhãn và độ dài review.</div>",
        unsafe_allow_html=True,
    )
    reviews_df = load_csv(LABELED_CSV)
    if reviews_df.empty:
        render_missing_file(LABELED_CSV, "file dữ liệu đã gán nhãn")
        return

    total_reviews = len(reviews_df)
    unique_products = reviews_df["parent_asin"].nunique() if "parent_asin" in reviews_df.columns else 0
    positive_pct = safe_percentage(reviews_df.get("label_id", pd.Series(dtype="int64")), 2)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Tổng số review", f"{total_reviews:,}")
    metric_col2.metric("Số parent_asin", f"{unique_products:,}")
    metric_col3.metric("% Positive", f"{positive_pct:.2f}%")

    chart_col1, chart_col2, chart_col3 = st.columns(3)

    with chart_col1:
        rating_df = (
            reviews_df["rating"]
            .value_counts()
            .sort_index()
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .rename_axis("Rating")
            .reset_index(name="Số lượng")
        )
        fig_rating = px.bar(rating_df, x="Rating", y="Số lượng", title="Phân bố rating")
        fig_rating.update_layout(height=360)
        st.plotly_chart(fig_rating, use_container_width=True)

    with chart_col2:
        label_name_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment_df = (
            reviews_df["label_id"]
            .map(label_name_map)
            .fillna(reviews_df.get("sentiment_label", pd.Series(dtype="object")))
            .value_counts()
            .reindex(LABEL_ORDER, fill_value=0)
            .rename_axis("Sentiment")
            .reset_index(name="Số lượng")
        )
        fig_sentiment = px.bar(
            sentiment_df,
            x="Sentiment",
            y="Số lượng",
            title="Phân bố nhãn cảm xúc",
            color="Sentiment",
            color_discrete_map={
                "Negative": "#ef4444",
                "Neutral": "#f59e0b",
                "Positive": "#22c55e",
            },
        )
        fig_sentiment.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_sentiment, use_container_width=True)

    with chart_col3:
        if "word_count" in reviews_df.columns:
            fig_length = px.histogram(
                reviews_df,
                x="word_count",
                nbins=40,
                title="Phân bố độ dài review",
            )
            fig_length.update_layout(height=360)
            st.plotly_chart(fig_length, use_container_width=True)
        else:
            st.warning("Thiếu cột `word_count`, không thể vẽ histogram độ dài review.")


def render_comparison() -> None:
    """Render the model comparison tab."""
    st.subheader("So sánh hiệu năng mô hình")
    st.markdown(
        "<div class='dashboard-note'>Macro F1 được ưu tiên vì bài toán có 3 lớp cảm xúc và cần cân bằng giữa các lớp.</div>",
        unsafe_allow_html=True,
    )
    comparison_df = load_csv(COMPARISON_CSV)
    if comparison_df.empty:
        render_missing_file(COMPARISON_CSV, "file so sánh mô hình")
        return

    display_columns = ["model_name", "accuracy", "precision", "recall", "macro_f1"]
    available_columns = [column for column in display_columns if column in comparison_df.columns]
    styled_df = comparison_df[available_columns].copy()
    styled = styled_df.style.format(
        {
            "accuracy": "{:.2%}",
            "precision": "{:.2%}",
            "recall": "{:.2%}",
            "macro_f1": "{:.4f}",
        }
    ).background_gradient(subset=["macro_f1"], cmap="YlGn")
    st.dataframe(styled, use_container_width=True)

    fig = px.bar(
        comparison_df.sort_values("macro_f1", ascending=True),
        x="macro_f1",
        y="model_name",
        orientation="h",
        title="So sánh macro F1 giữa các mô hình",
        text="macro_f1",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    best_row = comparison_df.sort_values("macro_f1", ascending=False).iloc[0]
    st.success(
        f"Mô hình tốt nhất theo macro_f1 là `{best_row['model_name']}` với điểm `{best_row['macro_f1']:.4f}`."
    )


def render_ranking() -> None:
    """Render the product ranking tab."""
    st.subheader("Bảng xếp hạng sản phẩm tiềm năng")
    st.markdown(
        "<div class='dashboard-note'>Tab này giúp ưu tiên các sản phẩm vừa có cảm xúc tích cực vừa có khối lượng review đủ lớn.</div>",
        unsafe_allow_html=True,
    )
    ranking_df = load_csv(RANKING_CSV)
    if ranking_df.empty:
        render_missing_file(RANKING_CSV, "file xếp hạng sản phẩm")
        return

    st.info(
        "Công thức: product_potential_score = 0.7 × normalized_sentiment_mean + "
        "0.3 × normalized_review_volume."
    )
    top_n = st.slider("Chọn số lượng sản phẩm top", min_value=5, max_value=20, value=10, step=1)

    top_df = ranking_df.sort_values("product_potential_score", ascending=False).head(top_n).copy()

    fig = px.bar(
        top_df,
        x="parent_asin",
        y="product_potential_score",
        color="product_potential_score",
        title=f"Top {top_n} sản phẩm theo product_potential_score",
        text="product_potential_score",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(height=420, xaxis_title="parent_asin", yaxis_title="product_potential_score")
    st.plotly_chart(fig, use_container_width=True)

    table_columns = [
        "parent_asin",
        "review_count",
        "sentiment_score_mean",
        "positive_ratio",
        "product_potential_score",
    ]
    available_columns = [column for column in table_columns if column in top_df.columns]
    st.dataframe(
        top_df[available_columns].style.format(
            {
                "sentiment_score_mean": "{:.4f}",
                "positive_ratio": "{:.2%}",
                "product_potential_score": "{:.4f}",
            }
        ),
        use_container_width=True,
    )


def handle_example_buttons() -> None:
    """Update the review text area from example buttons."""
    button_col1, button_col2, button_col3 = st.columns(3)
    if button_col1.button("Ví dụ tích cực", use_container_width=True):
        st.session_state["review_input"] = EXAMPLE_REVIEWS["positive"]
    if button_col2.button("Ví dụ tiêu cực", use_container_width=True):
        st.session_state["review_input"] = EXAMPLE_REVIEWS["negative"]
    if button_col3.button("Ví dụ mơ hồ", use_container_width=True):
        st.session_state["review_input"] = EXAMPLE_REVIEWS["ambiguous"]


def render_predict() -> None:
    """Render the real-time prediction tab."""
    st.subheader("Dự đoán cảm xúc theo thời gian thực")
    st.caption("Mô hình hiện chỉ xử lý review bằng TIẾNG ANH.")
    st.markdown(
        "<div class='dashboard-note'>Nếu checkpoint BERT load thành công, app sẽ dùng mô hình thật. "
        "Nếu không, app tự động chuyển sang chế độ <strong>DEMO FALLBACK</strong> để bạn vẫn demo được giao diện.</div>",
        unsafe_allow_html=True,
    )

    warning_message = None
    if not BERT_MODEL_DIR.exists():
        warning_message = f"Không tìm thấy model tại `{BERT_MODEL_DIR}`. Sẽ dùng DEMO FALLBACK."
    else:
        _, _, load_warning = load_bert_resources(BERT_MODEL_DIR)
        warning_message = load_warning

    if warning_message:
        st.warning(warning_message)

    if "review_input" not in st.session_state:
        st.session_state["review_input"] = ""

    handle_example_buttons()
    review_text = st.text_area(
        "Nhập nội dung review",
        key="review_input",
        placeholder="Paste một review tiếng Anh vào đây...",
        height=180,
    )

    if st.button("Phân tích", type="primary"):
        if not review_text.strip():
            st.warning("Vui lòng nhập review trước khi phân tích.")
            return

        used_fallback = False
        try:
            predicted_label, probabilities = infer_with_bert(review_text)
        except Exception:
            predicted_label, probabilities = infer_with_fallback(review_text)
            used_fallback = True

        result_col1, result_col2 = st.columns([0.35, 0.65])
        with result_col1:
            if used_fallback:
                st.info("Kết quả đang dùng DEMO FALLBACK.")
            st.metric("Nhãn dự đoán", predicted_label)
            st.metric("Độ tin cậy", f"{max(probabilities):.2%}")

        with result_col2:
            st.plotly_chart(build_probability_chart(probabilities), use_container_width=True)


def main() -> None:
    """Build and render the Streamlit dashboard."""
    inject_styles()
    st.title("Dashboard phân tích cảm xúc Amazon Electronics")
    tab_eda, tab_compare, tab_ranking, tab_predict = st.tabs(
        [
            "Phân tích dữ liệu (EDA)",
            "So sánh mô hình",
            "Xếp hạng sản phẩm tiềm năng",
            "Dự đoán cảm xúc (real-time)",
        ]
    )

    with tab_eda:
        render_eda()
    with tab_compare:
        render_comparison()
    with tab_ranking:
        render_ranking()
    with tab_predict:
        render_predict()


if __name__ == "__main__":
    main()

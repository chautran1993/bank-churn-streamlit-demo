# -*- coding: utf-8 -*-
"""Streamlit app for Vietnam bank customer churn prediction.

Run:
    streamlit run app.py --server.port 8501
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
FALLBACK_MODEL_PATH = BASE_DIR / "best_churn_model.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"

PROVINCES = [
    "TP. Hồ Chí Minh",
    "Hà Nội",
    "Đà Nẵng",
    "Cần Thơ",
    "Hải Phòng",
    "Đồng Nai",
    "Bình Dương",
    "Bà Rịa - Vũng Tàu",
    "An Giang",
    "Bắc Giang",
    "Bắc Ninh",
    "Bình Định",
    "Đắk Lắk",
    "Gia Lai",
    "Khánh Hòa",
    "Lâm Đồng",
    "Long An",
    "Nghệ An",
    "Quảng Ninh",
    "Thanh Hóa",
    "Thừa Thiên Huế",
]

OCCUPATIONS = [
    "Nhân viên văn phòng/Công chức",
    "Kinh doanh",
    "Tự do",
    "Nghỉ hưu",
    "Nội trợ/Sinh viên",
]

INCOME_OPTIONS = {
    "<5M": 4_000_000,
    "5-10M": 7_500_000,
    "10-20M": 15_000_000,
    "20-50M": 35_000_000,
    ">50M": 70_000_000,
}


st.set_page_config(
    page_title="Dự đoán churn ngân hàng",
    page_icon="🏦",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1280px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    }
    div[data-testid="stMetricValue"] {
        white-space: normal;
        overflow-wrap: anywhere;
        font-size: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_artifact() -> tuple[Any | None, dict[str, Any], str | None]:
    """Load trained model artifact and metadata once."""
    metadata: dict[str, Any] = {}
    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

    model_path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL_PATH
    if not model_path.exists():
        return None, metadata, "Please train and export the best model first."

    try:
        artifact = joblib.load(model_path)
        return artifact, metadata, None
    except Exception as exc:  # pragma: no cover - user-facing Streamlit branch
        return None, metadata, f"Không thể load model: {exc}"


def get_pipeline(artifact: Any) -> Any:
    """Support both a raw pipeline and a dict artifact from the notebook."""
    if isinstance(artifact, dict) and "pipeline" in artifact:
        return artifact["pipeline"]
    return artifact


def get_threshold(artifact: Any, metadata: dict[str, Any]) -> float:
    # UI should follow the reviewed notebook metadata. Some older exported
    # artifacts may still contain a previous threshold.
    if "best_threshold" in metadata:
        return float(metadata["best_threshold"])
    if isinstance(artifact, dict) and "best_threshold" in artifact:
        return float(artifact["best_threshold"])
    return 0.5


def get_feature_columns(artifact: Any, metadata: dict[str, Any]) -> list[str]:
    if isinstance(artifact, dict) and "feature_columns" in artifact:
        return list(artifact["feature_columns"])
    return list(metadata.get("feature_columns", []))


def predict_probability(pipeline: Any, input_df: pd.DataFrame) -> float:
    """Return churn probability from a trained sklearn/imblearn pipeline."""
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(input_df)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return float(proba[0, 1])
        return float(proba[0])

    if hasattr(pipeline, "decision_function"):
        score = float(pipeline.decision_function(input_df)[0])
        return float(1 / (1 + np.exp(-score)))

    prediction = int(pipeline.predict(input_df)[0])
    return float(prediction)


def risk_from_probability(probability: float) -> tuple[str, str]:
    if probability >= 0.60:
        return "Nguy cơ cao", "Churn"
    if probability >= 0.30:
        return "Nguy cơ trung bình", "Not Churn"
    return "Nguy cơ thấp", "Not Churn"


def recommendation_for_risk(risk_level: str) -> str:
    if risk_level == "Nguy cơ cao":
        return (
            "Khách hàng có nguy cơ rời bỏ cao. Khuyến nghị: gọi điện tư vấn "
            "trong 7 ngày, đề xuất ưu đãi lãi suất hoặc gói dịch vụ phù hợp."
        )
    if risk_level == "Nguy cơ trung bình":
        return (
            "Theo dõi sát hơn. Khuyến nghị: gửi email chăm sóc, khảo sát trải nghiệm "
            "và mời tham gia chương trình loyalty."
        )
    return "Khách hàng ổn định. Khuyến nghị: tiếp tục duy trì chất lượng dịch vụ hiện tại."


def build_input_form() -> pd.DataFrame:
    """Create sidebar form and return one customer record."""
    st.sidebar.title("Thông tin khách hàng")

    with st.sidebar.expander("1. Thông tin cá nhân", expanded=True):
        age = st.number_input("Tuổi", min_value=18, max_value=80, value=35, step=1)
        gender_label = st.selectbox("Giới tính", ["Nam", "Nữ"])
        origin_province = st.selectbox("Tỉnh thành", PROVINCES)
        occupation = st.selectbox("Nghề nghiệp", OCCUPATIONS)
        married_label = st.selectbox("Hôn nhân", ["Có", "Không"])

    with st.sidebar.expander("2. Thông tin tài chính", expanded=True):
        balance = st.number_input(
            "Số dư tài khoản",
            min_value=0,
            max_value=500_000_000,
            value=30_000_000,
            step=1_000_000,
            format="%d",
        )
        credit_sco = st.slider("Điểm tín dụng", min_value=300, max_value=850, value=650)
        income_label = st.selectbox("Thu nhập hàng tháng", list(INCOME_OPTIONS.keys()), index=2)
        tenure_ye = st.slider("Số năm gắn bó", min_value=0, max_value=15, value=3)
        nums_card = st.selectbox("Số thẻ", [1, 2, 3, 4], index=1)
        nums_service = st.selectbox("Số sản phẩm", [1, 2, 3, 4, 5], index=2)

    with st.sidebar.expander("3. Thông tin hành vi", expanded=True):
        active_label = st.selectbox("Thành viên tích cực", ["Có", "Không"])
        last_transaction_month = st.slider("Tháng giao dịch gần nhất", min_value=1, max_value=12, value=6)
        engagement_score = st.slider("Engagement score", min_value=1, max_value=100, value=50)
        loyalty_level = st.selectbox("Loyalty level", ["Bronze", "Silver", "Gold"])
        digital_behavior = st.selectbox("Digital behavior", ["offline", "mobile"])
        customer_segment = st.selectbox("Customer segment", ["Mass", "Emerging", "Affluent", "Priority"])

    st.sidebar.divider()
    st.sidebar.caption("Nhóm 3 - Demo Machine Learning")

    record = {
        "credit_sco": int(credit_sco),
        "gender": "male" if gender_label == "Nam" else "female",
        "age": int(age),
        "occupation": occupation,
        "balance": int(balance),
        "monthly_ir": int(INCOME_OPTIONS[income_label]),
        "origin_province": origin_province,
        "tenure_ye": int(tenure_ye),
        "married": 1 if married_label == "Có" else 0,
        "nums_card": int(nums_card),
        "nums_service": int(nums_service),
        "active_member": True if active_label == "Có" else False,
        "last_transaction_month": int(last_transaction_month),
        "customer_segment": customer_segment,
        "engagement_score": int(engagement_score),
        "loyalty_level": loyalty_level,
        "digital_behavior": digital_behavior,
    }
    return pd.DataFrame([record])


def show_model_info(metadata: dict[str, Any], threshold: float, using_real_model: bool) -> None:
    model_name = metadata.get("best_model_name", "Model đã huấn luyện")
    pipeline_name = metadata.get("best_pipeline", "Pipeline đã huấn luyện")
    status = "Model thật đã được tải" if using_real_model else "Chưa có model thật"
    st.info(f"{status}: {pipeline_name} · {model_name} · Threshold {threshold:.3f}")

    metrics = metadata.get("final_test_metrics") or metadata.get("validation_metrics") or {}
    if metrics:
        with st.expander("Thông tin model và metric", expanded=False):
            st.write(f"**Pipeline:** {metrics.get('Pipeline', pipeline_name)}")
            st.write(f"**Model:** {metrics.get('Model', model_name)}")
            st.write(f"**Threshold tối ưu trên validation:** {float(metrics.get('Threshold', threshold)):.3f}")
            metric_cols = st.columns(5)
            for col, key, label in zip(
                metric_cols,
                ["Accuracy", "Precision_Churn", "Recall_Churn", "F1_Churn", "ROC_AUC"],
                ["Accuracy", "Precision", "Recall", "F1 churn", "ROC-AUC"],
            ):
                value = metrics.get(key)
                if value is not None:
                    col.metric(label, f"{float(value):.3f}")


def show_result(input_df: pd.DataFrame, probability: float, threshold: float, using_real_model: bool) -> None:
    risk_level, default_label = risk_from_probability(probability)
    prediction = "Churn" if probability >= threshold else "Not Churn"
    recommendation = recommendation_for_risk(risk_level)

    if risk_level == "Nguy cơ cao":
        st.error(f"{risk_level.upper()} | Xác suất churn: {probability:.2%} | Dự đoán: {prediction}")
    elif risk_level == "Nguy cơ trung bình":
        st.warning(f"{risk_level.upper()} | Xác suất churn: {probability:.2%} | Dự đoán: {prediction}")
    else:
        st.success(f"{risk_level.upper()} | Xác suất churn: {probability:.2%} | Dự đoán: {prediction}")

    col1, col2, col3 = st.columns([1, 1, 1.4])
    col1.metric("Xác suất churn", f"{probability:.2%}", "Model thật" if using_real_model else "Demo")
    col2.metric("Dự đoán", prediction, f"Threshold {threshold:.3f}")
    col3.metric("Mức rủi ro", risk_level)

    st.markdown(f"**Khuyến nghị:** {recommendation}")
    st.progress(min(max(probability, 0.0), 1.0))

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Diễn giải nhanh")
        st.write(
            "- Xác suất churn là khả năng khách hàng rời bỏ dịch vụ theo model.\n"
            "- Dự đoán Churn/Not Churn được quyết định bằng threshold đã tối ưu khi train.\n"
            "- Mức rủi ro Low/Medium/High giúp ngân hàng ưu tiên hành động chăm sóc."
        )
    with right:
        st.subheader("Input Summary")
        summary = {
            "Điểm tín dụng": f"{input_df.loc[0, 'credit_sco']}",
            "Giới tính": input_df.loc[0, "gender"],
            "Tuổi": f"{input_df.loc[0, 'age']}",
            "Nghề nghiệp": input_df.loc[0, "occupation"],
            "Số dư tài khoản": f"{int(input_df.loc[0, 'balance']):,} VND",
            "Thu nhập hàng tháng": f"{int(input_df.loc[0, 'monthly_ir']):,} VND",
            "Tỉnh thành": input_df.loc[0, "origin_province"],
            "Số năm gắn bó": f"{input_df.loc[0, 'tenure_ye']}",
            "Số thẻ": f"{input_df.loc[0, 'nums_card']}",
            "Số sản phẩm": f"{input_df.loc[0, 'nums_service']}",
            "Thành viên tích cực": "Có" if bool(input_df.loc[0, "active_member"]) else "Không",
            "Engagement score": f"{input_df.loc[0, 'engagement_score']}",
        }
        st.dataframe(pd.DataFrame(summary.items(), columns=["Thông tin", "Giá trị"]), use_container_width=True)


def main() -> None:
    artifact, metadata, load_error = load_model_artifact()
    using_real_model = artifact is not None and load_error is None
    threshold = get_threshold(artifact, metadata) if using_real_model else float(metadata.get("best_threshold", 0.5))

    st.title("🏦 Dự đoán khả năng rời bỏ dịch vụ ngân hàng")
    st.caption("Demo Machine Learning hỗ trợ nhận diện khách hàng có nguy cơ churn và đề xuất hành động chăm sóc phù hợp.")

    input_df = build_input_form()
    show_model_info(metadata, threshold, using_real_model)

    predict_clicked = st.sidebar.button("Dự đoán ngay", use_container_width=True, type="primary")
    st.subheader("Kết quả dự đoán")

    if load_error:
        st.warning(load_error)

    if predict_clicked:
        if using_real_model:
            pipeline = get_pipeline(artifact)
            feature_columns = get_feature_columns(artifact, metadata)
            missing_cols = [col for col in feature_columns if col not in input_df.columns]
            if missing_cols:
                st.error(f"Input thiếu cột bắt buộc: {missing_cols}")
                return

            try:
                model_input = input_df[feature_columns] if feature_columns else input_df
                probability = predict_probability(pipeline, model_input)
            except Exception as exc:
                st.error(f"Lỗi khi dự đoán: {exc}")
                return
        else:
            probability = 0.42

        show_result(input_df, probability, threshold, using_real_model)
    else:
        st.write("Nhập thông tin khách hàng ở sidebar, sau đó bấm **Dự đoán ngay** để xem kết quả.")


if __name__ == "__main__":
    main()

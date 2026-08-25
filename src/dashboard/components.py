"""Reusable Streamlit components for the QwiSense dashboard."""

import streamlit as st


def render_section_header(title: str, subtitle: str = "") -> None:
    """Render a consistent dashboard section heading."""
    st.markdown(f"## {title}")

    if subtitle:
        st.caption(subtitle)


def render_status_card(label: str, value: str) -> None:
    """Render a compact status card."""
    st.markdown(
        f"""
        <div class="qwisense-status">
            <div class="qwisense-status-label">{label}</div>
            <div class="qwisense-status-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi(label: str, value: str) -> None:
    """Render a dashboard KPI card."""
    st.markdown(
        f"""
        <div class="qwisense-kpi">
            <div class="qwisense-kpi-label">{label}</div>
            <div class="qwisense-kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_activity_badge(activity: str, confidence: float | None = None) -> None:
    """Render the current detected activity and optional confidence."""
    confidence_text = (
        f"{confidence:.1%} confidence"
        if confidence is not None
        else "Confidence unavailable"
    )

    st.markdown(
        f"""
        <div class="qwisense-kpi">
            <div class="qwisense-kpi-label">Detected Activity</div>
            <div class="qwisense-kpi-value">{activity}</div>
            <div>{confidence_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_row(metrics: list[tuple[str, str]]) -> None:
    """Render multiple dashboard metrics in a consistent row."""
    columns = st.columns(len(metrics))

    for column, (label, value) in zip(columns, metrics):
        with column:
            render_kpi(label, value)


def render_alert(message: str, level: str = "info") -> None:
    """Render a dashboard status message."""
    levels = {
        "info": st.info,
        "success": st.success,
        "warning": st.warning,
        "error": st.error,
    }

    renderer = levels.get(level)

    if renderer is None:
        raise ValueError(
            "level must be one of: info, success, warning, error"
        )

    renderer(message)


def render_section(title: str, subtitle: str = "") -> None:
    """Render a consistent dashboard section."""
    st.divider()
    st.markdown(f"### {title}")

    if subtitle:
        st.caption(subtitle)


def render_section(title: str, subtitle: str = "") -> None:
    """Render a consistent dashboard section."""
    st.divider()
    st.markdown(f"### {title}")

    if subtitle:
        st.caption(subtitle)


def render_empty_state(title: str, message: str) -> None:
    """Render an empty-state message."""
    st.markdown(f"### {title}")
    st.info(message)


def render_loading_state(message: str = "Loading QwiSense data...") -> None:
    """Render a dashboard loading indicator."""
    with st.spinner(message):
        st.empty()


def render_prediction_card(
    activity: str,
    confidence: float | None = None,
) -> None:
    """Render a prediction-focused dashboard card."""
    st.markdown("### 🎯 Prediction")
    st.metric("Activity", activity)

    if confidence is not None:
        st.progress(max(0.0, min(1.0, confidence)))
        st.caption(f"Confidence: {confidence:.1%}")
    else:
        st.caption("Confidence unavailable")


def render_model_badge(model_name: str) -> None:
    """Render the active model name."""
    st.caption(f"Active model: **{model_name}**")


def render_data_status(samples: int, source: str) -> None:
    """Render dataset status information."""
    st.markdown("### 📦 Dataset")
    cols = st.columns(2)

    with cols[0]:
        st.metric("Samples", samples)

    with cols[1]:
        st.metric("Source", source)


def render_pipeline_status(
    preprocessing: str = "Ready",
    inference: str = "Ready",
) -> None:
    """Render preprocessing and inference status."""
    st.markdown("### ⚙️ Pipeline Status")

    cols = st.columns(2)

    with cols[0]:
        st.metric("Preprocessing", preprocessing)

    with cols[1]:
        st.metric("Inference", inference)

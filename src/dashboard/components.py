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

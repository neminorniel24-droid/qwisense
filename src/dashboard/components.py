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


def render_activity_classes(classes: dict[int, str]) -> None:
    """Render supported activity classes."""
    st.markdown("### 🏷️ Activity Classes")

    for class_id, name in classes.items():
        st.write(f"**{class_id}** — {name}")


def render_confidence_metric(confidence: float | None) -> None:
    """Render confidence as a percentage metric."""
    if confidence is None:
        st.metric("Confidence", "—")
        return

    bounded = max(0.0, min(1.0, confidence))
    st.metric("Confidence", f"{bounded:.1%}")


def render_dashboard_summary(
    activity: str,
    confidence: float | None,
    samples: int,
    model: str,
) -> None:
    """Render the primary dashboard summary."""
    st.markdown("## 📊 Sensing Overview")

    cols = st.columns(4)

    with cols[0]:
        st.metric("Activity", activity)

    with cols[1]:
        value = "—" if confidence is None else f"{confidence:.1%}"
        st.metric("Confidence", value)

    with cols[2]:
        st.metric("Samples", samples)

    with cols[3]:
        st.metric("Model", model)


def render_model_selector(
    models: list[str],
    default: int = 0,
) -> str:
    """Render a model selector and return the selected model."""
    if not models:
        raise ValueError("models must contain at least one model.")

    if not 0 <= default < len(models):
        raise ValueError("default must reference a valid model.")

    return st.selectbox(
        "Model",
        models,
        index=default,
    )


def render_activity_selector(activities: list[str]) -> str:
    """Render an activity filter."""
    if not activities:
        raise ValueError("activities must contain at least one activity.")

    return st.selectbox(
        "Activity",
        activities,
    )


def render_sample_count_control(
    minimum: int = 1,
    maximum: int = 1000,
    default: int = 100,
) -> int:
    """Render a sample-count control."""
    if minimum > maximum:
        raise ValueError("minimum must not exceed maximum.")

    if not minimum <= default <= maximum:
        raise ValueError("default must be within the allowed range.")

    return st.slider(
        "Sample count",
        min_value=minimum,
        max_value=maximum,
        value=default,
    )


def render_inference_button(label: str = "Run inference") -> bool:
    """Render the primary inference action."""
    return st.button(
        label,
        type="primary",
        use_container_width=True,
    )


def render_refresh_button(label: str = "Refresh data") -> bool:
    """Render a dashboard refresh action."""
    return st.button(
        f"↻ {label}",
        use_container_width=True,
    )


def render_last_updated(timestamp: str) -> None:
    """Render the last-updated timestamp."""
    st.caption(f"Last updated: {timestamp}")


def render_progress(label: str, value: float) -> None:
    """Render a bounded progress indicator."""
    if not 0.0 <= value <= 1.0:
        raise ValueError("value must be between 0 and 1.")

    st.caption(label)
    st.progress(value)


def render_warning_banner(message: str) -> None:
    """Render a prominent dashboard warning."""
    st.warning(f"⚠️ {message}")


def render_success_banner(message: str) -> None:
    """Render a prominent dashboard success message."""
    st.success(f"✓ {message}")


def render_technical_details(details: dict[str, str]) -> None:
    """Render technical dashboard details in an expandable panel."""
    with st.expander("🔬 Technical Details"):
        for key, value in details.items():
            st.write(f"**{key}:** {value}")


def render_confidence_badge(confidence: float | None) -> None:
    """Render a compact confidence badge."""
    if confidence is None:
        st.caption("Confidence: —")
        return

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1.")

    st.caption(f"Confidence: **{confidence:.1%}**")


def render_dataset_selector(sources: list[str], default: int = 0) -> str:
    """Render a dataset source selector."""
    if not sources:
        raise ValueError("sources must contain at least one item.")
    if not 0 <= default < len(sources):
        raise ValueError("default must reference a valid source.")

    return st.selectbox("Dataset", sources, index=default)


def render_window_size_control(
    minimum: int = 50,
    maximum: int = 1000,
    default: int = 200,
) -> int:
    """Render a CSI window-size control."""
    if minimum > maximum:
        raise ValueError("minimum must not exceed maximum.")
    if not minimum <= default <= maximum:
        raise ValueError("default must be within the allowed range.")

    return st.number_input(
        "Window samples",
        min_value=minimum,
        max_value=maximum,
        value=default,
        step=10,
    )


def render_sampling_rate_control(
    minimum: float = 1.0,
    maximum: float = 1000.0,
    default: float = 100.0,
) -> float:
    """Render a sampling-rate control."""
    if minimum <= 0 or minimum > maximum:
        raise ValueError("sampling-rate bounds are invalid.")
    if not minimum <= default <= maximum:
        raise ValueError("default must be within the allowed range.")

    return st.number_input(
        "Sampling rate (Hz)",
        min_value=minimum,
        max_value=maximum,
        value=default,
    )


def render_preprocessing_status(status: str = "Ready") -> None:
    """Render preprocessing pipeline status."""
    st.metric("Preprocessing", status)


def render_model_status(model: str, status: str = "Ready") -> None:
    """Render active model status."""
    st.metric(f"{model} Status", status)


def render_sample_info(
    samples: int,
    subcarriers: int = 52,
    window_samples: int = 200,
) -> None:
    """Render CSI sample information."""
    cols = st.columns(3)

    with cols[0]:
        st.metric("Samples", samples)

    with cols[1]:
        st.metric("Subcarriers", subcarriers)

    with cols[2]:
        st.metric("Window", window_samples)


def validate_model_name(model_name: str, available_models: list[str]) -> str:
    """Validate a selected model name."""
    if not available_models:
        raise ValueError("available_models must not be empty.")

    if model_name not in available_models:
        raise ValueError(f"Unknown model: {model_name}")

    return model_name


def activity_label(class_id: int) -> str:
    """Convert a QwiSense activity class ID to its display label."""
    labels = {
        0: "Empty",
        1: "Present",
        2: "Walking",
        3: "Fall",
    }

    if class_id not in labels:
        raise ValueError(f"Unknown activity class: {class_id}")

    return labels[class_id]


def render_version_badge(version: str) -> None:
    """Render the dashboard version."""
    st.caption(f"QwiSense Dashboard v{version}")


def render_signal_quality(score: float | None) -> None:
    """Render a CSI signal-quality indicator."""
    if score is None:
        st.metric("Signal Quality", "—")
        return

    if not 0.0 <= score <= 1.0:
        raise ValueError("score must be between 0 and 1.")

    st.metric("Signal Quality", f"{score:.1%}")


def render_sensor_status(status: str) -> None:
    """Render the current sensor connection status."""
    st.metric("Sensor", status)


def render_feature_count(count: int) -> None:
    """Render the number of extracted features."""
    if count < 0:
        raise ValueError("count must not be negative.")

    st.metric("Features", count)


def render_latency(latency_ms: float | None) -> None:
    """Render inference latency."""
    if latency_ms is None:
        st.metric("Latency", "—")
        return

    if latency_ms < 0:
        raise ValueError("latency_ms must not be negative.")

    st.metric("Latency", f"{latency_ms:.1f} ms")


def render_fall_alert(is_fall: bool) -> None:
    """Render a fall-detection status indicator."""
    if is_fall:
        st.error("🚨 Fall activity detected")
    else:
        st.success("No fall detected")


def render_activity_distribution(
    distribution: dict[str, int],
) -> None:
    """Render activity counts as a dashboard table."""
    st.markdown("### Activity Distribution")

    if not distribution:
        st.info("No activity data available.")
        return

    for activity, count in distribution.items():
        st.write(f"**{activity}** — {count}")


def render_confidence_threshold(
    default: float = 0.70,
) -> float:
    """Render the minimum confidence threshold."""
    if not 0.0 <= default <= 1.0:
        raise ValueError("default must be between 0 and 1.")

    return st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=default,
        step=0.01,
    )


def render_inference_mode_selector() -> str:
    """Select between single-sample and batch inference."""
    return st.radio(
        "Inference mode",
        ["Single sample", "Batch"],
        horizontal=True,
    )


def render_model_comparison_header() -> None:
    """Render the model-comparison section header."""
    st.markdown("### ⚖️ Model Comparison")
    st.caption("Compare classical and quantum model performance.")


def render_dashboard_help() -> None:
    """Render a concise dashboard usage guide."""
    with st.expander("❓ How to use QwiSense"):
        st.markdown(
            """
            1. Select a dataset.
            2. Configure preprocessing parameters.
            3. Choose a model.
            4. Run inference.
            5. Review activity and confidence.
            """
        )


def build_export_filename(prefix: str, extension: str) -> str:
    """Build a safe dashboard export filename."""
    clean_prefix = prefix.strip().replace(" ", "_")
    clean_extension = extension.strip().lstrip(".")

    if not clean_prefix:
        raise ValueError("prefix must not be empty.")

    if not clean_extension:
        raise ValueError("extension must not be empty.")

    return f"{clean_prefix}.{clean_extension}"


def render_download_button(
    label: str,
    data: str | bytes,
    filename: str,
    mime: str,
) -> bool:
    """Render a reusable dashboard download button."""
    return st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime,
        use_container_width=True,
    )


def dataframe_to_csv(dataframe) -> bytes:
    """Convert a dataframe to UTF-8 CSV bytes."""
    if dataframe is None:
        raise ValueError("dataframe must not be None.")

    return dataframe.to_csv(index=False).encode("utf-8")


def result_to_json(result: dict) -> str:
    """Serialize dashboard results to readable JSON."""
    import json

    if not isinstance(result, dict):
        raise TypeError("result must be a dictionary.")

    return json.dumps(result, indent=2, default=str)


def render_accessible_note(text: str) -> None:
    """Render supplementary accessible dashboard guidance."""
    if not text.strip():
        raise ValueError("text must not be empty.")

    st.caption(text)


def render_prediction_history(history: list[dict]) -> None:
    """Render recent prediction history."""
    st.markdown("### 🕘 Recent Predictions")

    if not history:
        st.info("No predictions yet.")
        return

    st.dataframe(history, use_container_width=True)


def render_recent_events(events: list[str]) -> None:
    """Render recent sensing events."""
    st.markdown("### 🕘 Recent Events")

    if not events:
        st.info("No recent events.")
        return

    for event in events:
        st.write(f"• {event}")


def render_model_metrics(metrics: dict[str, dict[str, float]]) -> None:
    """Render model evaluation metrics."""
    st.markdown("### 📋 Model Metrics")

    if not metrics:
        st.info("No model metrics available.")
        return

    st.dataframe(metrics, use_container_width=True)


def render_dataset_info(info: dict[str, str]) -> None:
    """Render dataset metadata."""
    st.markdown("### 📦 Dataset Information")

    if not info:
        st.info("No dataset metadata available.")
        return

    for key, value in info.items():
        st.write(f"**{key}:** {value}")


def render_navigation_note(current_section: str) -> None:
    """Show the active dashboard section."""
    if not current_section.strip():
        raise ValueError("current_section must not be empty.")

    st.caption(f"Current section: **{current_section}**")


def render_signal_overview(
    subcarriers: int = 52,
    window_samples: int = 200,
    sampling_rate: float = 100.0,
) -> None:
    """Render the main CSI signal overview."""
    st.markdown("### 📡 Signal Overview")

    cols = st.columns(3)

    with cols[0]:
        st.metric("Subcarriers", subcarriers)

    with cols[1]:
        st.metric("Window", f"{window_samples} samples")

    with cols[2]:
        st.metric("Sampling Rate", f"{sampling_rate:g} Hz")


def render_quantum_model_card(
    model_name: str = "VQC",
    qubits: int | None = None,
) -> None:
    """Render quantum-model information."""
    st.markdown("### ⚛️ Quantum Model")

    st.markdown(f"**Model:** {model_name}")

    if qubits is not None:
        st.caption(f"Qubits: {qubits}")
    else:
        st.caption("Quantum circuit configuration unavailable.")


def render_classical_model_card(
    model_name: str = "Random Forest",
) -> None:
    """Render classical-model information."""
    st.markdown("### 🤖 Classical Model")
    st.markdown(f"**Model:** {model_name}")
    st.caption("Classical baseline for activity recognition.")


def render_comparison_metrics(
    classical_accuracy: float | None,
    quantum_accuracy: float | None,
) -> None:
    """Render classical versus quantum accuracy."""
    st.markdown("### ⚖️ Model Comparison")

    cols = st.columns(2)

    with cols[0]:
        value = (
            "—"
            if classical_accuracy is None
            else f"{classical_accuracy:.1%}"
        )
        st.metric("Classical Accuracy", value)

    with cols[1]:
        value = (
            "—"
            if quantum_accuracy is None
            else f"{quantum_accuracy:.1%}"
        )
        st.metric("Quantum Accuracy", value)


def render_signal_health(
    quality: float | None,
    status: str = "Ready",
) -> None:
    """Render signal health information."""
    st.markdown("### 💚 Signal Health")

    if quality is None:
        st.metric("Quality", "—")
    else:
        if not 0.0 <= quality <= 1.0:
            raise ValueError("quality must be between 0 and 1.")
        st.metric("Quality", f"{quality:.1%}")

    st.caption(f"Status: {status}")


def render_inference_summary(
    activity: str,
    confidence: float | None,
    latency_ms: float | None,
) -> None:
    """Render a compact inference summary."""
    st.markdown("### 🧠 Inference Summary")

    cols = st.columns(3)

    with cols[0]:
        st.metric("Activity", activity)

    with cols[1]:
        st.metric(
            "Confidence",
            "—" if confidence is None else f"{confidence:.1%}",
        )

    with cols[2]:
        st.metric(
            "Latency",
            "—" if latency_ms is None else f"{latency_ms:.1f} ms",
        )


def render_no_data_state() -> None:
    """Render the initial dashboard state when no CSI is loaded."""
    st.info(
        "📡 No CSI data loaded yet. "
        "Choose a dataset or provide signal data to begin."
    )


def render_error_state(message: str) -> None:
    """Render a consistent dashboard error state."""
    if not message.strip():
        raise ValueError("message must not be empty.")

    st.error(f"Something went wrong: {message}")


def render_control_group(title: str) -> None:
    """Render a consistent controls section."""
    st.markdown(f"### ⚙️ {title}")

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


def render_product_summary() -> None:
    """Render the QwiSense product summary."""
    st.markdown(
        """
        ### 📡 QwiSense

        **Quantum-enhanced WiFi human sensing**

        Monitor CSI signals, preprocess wireless measurements,
        compare machine-learning models, and inspect activity
        recognition results from a single dashboard.
        """
    )


def render_signal_chart_header(title: str = "CSI Signal") -> None:
    """Render a signal-chart heading."""
    st.markdown(f"### 📈 {title}")
    st.caption("Wireless channel-state information visualization.")


def render_signal_statistics(
    mean: float | None,
    minimum: float | None,
    maximum: float | None,
) -> None:
    """Render basic signal statistics."""
    st.markdown("### 📊 Signal Statistics")

    cols = st.columns(3)

    values = [
        ("Mean", mean),
        ("Minimum", minimum),
        ("Maximum", maximum),
    ]

    for column, (label, value) in zip(cols, values):
        with column:
            st.metric(
                label,
                "—" if value is None else f"{value:.3f}",
            )


def render_feature_summary(
    feature_count: int,
    feature_type: str = "Extracted features",
) -> None:
    """Render feature extraction summary."""
    if feature_count < 0:
        raise ValueError("feature_count must not be negative.")

    st.markdown("### 🧮 Feature Extraction")
    st.metric(feature_type, feature_count)


def render_preprocessing_config(
    normalize: bool = True,
    detrend: bool = False,
) -> tuple[bool, bool]:
    """Render preprocessing configuration controls."""
    st.markdown("### 🔧 Preprocessing")

    normalize_value = st.checkbox(
        "Normalize CSI",
        value=normalize,
    )

    detrend_value = st.checkbox(
        "Remove trend",
        value=detrend,
    )

    return normalize_value, detrend_value


def render_signal_mode_selector() -> str:
    """Select the CSI representation."""
    return st.radio(
        "Signal representation",
        ["Amplitude", "Phase", "Magnitude"],
        horizontal=True,
    )


def render_channel_selector(
    channels: list[int],
    default: int = 0,
) -> int:
    """Select a CSI channel."""
    if not channels:
        raise ValueError("channels must not be empty.")

    if not 0 <= default < len(channels):
        raise ValueError("default must reference a valid channel.")

    return st.selectbox(
        "CSI channel",
        channels,
        index=default,
    )


def render_subcarrier_selector(
    minimum: int = 1,
    maximum: int = 52,
) -> int:
    """Select a CSI subcarrier."""
    if minimum > maximum:
        raise ValueError("minimum must not exceed maximum.")

    return st.slider(
        "Subcarrier",
        min_value=minimum,
        max_value=maximum,
        value=minimum,
    )


def render_visualization_toggle(default: bool = True) -> bool:
    """Toggle signal visualization."""
    return st.toggle(
        "Show signal visualization",
        value=default,
    )


def render_chart_empty_state(chart_name: str = "Signal") -> None:
    """Render an empty state for unavailable chart data."""
    if not chart_name.strip():
        raise ValueError("chart_name must not be empty.")

    st.info(f"No {chart_name.lower()} data available yet.")


def render_signal_controls() -> tuple[str, int]:
    """Render the primary signal-analysis controls."""
    mode = render_signal_mode_selector()
    subcarrier = render_subcarrier_selector()

    return mode, subcarrier


def validate_signal_data(values) -> bool:
    """Validate that signal data is non-empty and numeric."""
    if values is None:
        raise ValueError("signal data must not be None.")

    if len(values) == 0:
        raise ValueError("signal data must not be empty.")

    try:
        [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("signal data must be numeric.") from exc

    return True


def render_signal_preview(values, title: str = "Signal Preview") -> None:
    """Render a lightweight signal preview."""
    validate_signal_data(values)

    st.markdown(f"### 📈 {title}")
    st.line_chart(list(values), use_container_width=True)


def render_amplitude_preview(values) -> None:
    """Render CSI amplitude data."""
    render_signal_preview(
        values,
        title="CSI Amplitude",
    )


def render_phase_preview(values) -> None:
    """Render CSI phase data."""
    render_signal_preview(
        values,
        title="CSI Phase",
    )


def render_activity_history(history: list[dict]) -> None:
    """Render recent activity predictions."""
    st.markdown("### 🕘 Activity History")

    if not history:
        st.info("No activity predictions available.")
        return

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True,
    )


def render_confusion_matrix_section(matrix) -> None:
    """Render confusion-matrix data."""
    st.markdown("### 🧩 Confusion Matrix")

    if matrix is None:
        st.info("No confusion matrix available.")
        return

    st.dataframe(
        matrix,
        use_container_width=True,
    )


def render_evaluation_metrics(
    accuracy: float | None = None,
    f1_weighted: float | None = None,
    f1_fall: float | None = None,
) -> None:
    """Render evaluation metrics."""
    st.markdown("### 📊 Evaluation Metrics")

    metrics = [
        ("Accuracy", accuracy),
        ("Weighted F1", f1_weighted),
        ("Fall F1", f1_fall),
    ]

    columns = st.columns(len(metrics))

    for column, (label, value) in zip(columns, metrics):
        with column:
            display = "—" if value is None else f"{value:.1%}"
            st.metric(label, display)


def render_model_comparison_table(models: list[dict]) -> None:
    """Render a model comparison table."""
    st.markdown("### ⚖️ Model Comparison")

    if not models:
        st.info("No model comparison data available.")
        return

    st.dataframe(
        models,
        use_container_width=True,
        hide_index=True,
    )


def render_confidence_distribution(values) -> None:
    """Render prediction-confidence distribution."""
    validate_signal_data(values)

    st.markdown("### 📈 Confidence Distribution")
    st.bar_chart(list(values), use_container_width=True)


def render_analytics_section() -> None:
    """Render the analytics section heading."""
    st.markdown("## 📊 Analytics")
    st.caption(
        "Inspect CSI signals, predictions, confidence, "
        "and model performance."
    )


def render_dashboard_theme() -> None:
    """Apply the QwiSense dashboard visual theme."""
    st.markdown(
        """
        <style>
        .qwisense-panel {
            padding: 1rem;
            border-radius: 14px;
            border: 1px solid rgba(128,128,128,.22);
            margin-bottom: 1rem;
        }

        .qwisense-muted {
            opacity: .7;
            font-size: .9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_panel(title: str, content: str) -> None:
    """Render a reusable dashboard panel."""
    if not title.strip():
        raise ValueError("title must not be empty.")

    st.markdown(
        f"""
        <div class="qwisense-panel">
            <strong>{title}</strong>
            <div class="qwisense-muted">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_confidence(
    classical: float | None,
    quantum: float | None,
) -> None:
    """Compare model confidence values."""
    st.markdown("### Confidence Comparison")

    cols = st.columns(2)

    with cols[0]:
        st.metric(
            "Classical",
            "—" if classical is None else f"{classical:.1%}",
        )

    with cols[1]:
        st.metric(
            "Quantum",
            "—" if quantum is None else f"{quantum:.1%}",
        )


def render_prediction_status(
    activity: str,
    confidence: float | None,
) -> None:
    """Render prediction status."""
    if confidence is not None and not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1.")

    if confidence is None:
        st.info(f"Prediction: {activity}")
    elif confidence >= 0.8:
        st.success(f"High-confidence prediction: {activity}")
    elif confidence >= 0.5:
        st.warning(f"Moderate-confidence prediction: {activity}")
    else:
        st.warning(f"Low-confidence prediction: {activity}")


def render_activity_indicator(activity: str) -> None:
    """Render an activity indicator with accessible text."""
    if not activity.strip():
        raise ValueError("activity must not be empty.")

    st.markdown(f"**Current activity:** {activity}")


def render_inference_controls() -> tuple[bool, bool]:
    """Render inference configuration controls."""
    live_mode = st.toggle("Live inference", value=False)
    auto_refresh = st.toggle("Auto refresh", value=False)

    return live_mode, auto_refresh


def validate_kpi_value(value: float | int) -> float:
    """Normalize a numeric KPI value."""
    if not isinstance(value, (int, float)):
        raise TypeError("KPI value must be numeric.")

    return float(value)


def format_percentage(value: float | None) -> str:
    """Format a normalized value as a percentage."""
    if value is None:
        return "—"

    if not 0 <= value <= 1:
        raise ValueError("percentage value must be between 0 and 1.")

    return f"{value:.1%}"


def format_latency(value: float | None) -> str:
    """Format inference latency."""
    if value is None:
        return "—"

    if value < 0:
        raise ValueError("latency must not be negative.")

    return f"{value:.1f} ms"


def render_component_guide() -> None:
    """Show the dashboard component categories."""
    with st.expander("🧩 Dashboard Components"):
        st.markdown(
            """
            **Signal**
            - CSI overview
            - Signal statistics
            - Amplitude and phase previews

            **Inference**
            - Activity prediction
            - Confidence
            - Latency

            **Models**
            - Classical baseline
            - Quantum VQC
            - Model comparison
            """
        )


def render_dashboard_header(title: str = "QwiSense") -> None:
    """Render the primary dashboard header."""
    if not title.strip():
        raise ValueError("title must not be empty.")

    st.title(f"📡 {title}")
    st.caption("Quantum-enhanced WiFi human sensing")


def render_section_navigation(sections: list[str]) -> str:
    """Render dashboard section navigation."""
    if not sections:
        raise ValueError("sections must not be empty.")

    return st.radio(
        "Dashboard section",
        sections,
        horizontal=True,
    )


def render_date_range_filter():
    """Render a dashboard date range filter."""
    import datetime as dt

    return st.date_input(
        "Date range",
        value=(dt.date.today(), dt.date.today()),
    )


def render_activity_filter(
    activities: list[str],
) -> list[str]:
    """Render a multi-select activity filter."""
    if not activities:
        raise ValueError("activities must not be empty.")

    return st.multiselect(
        "Activities",
        activities,
        default=activities,
    )


def render_model_filter(models: list[str]) -> list[str]:
    """Render a model multi-select filter."""
    if not models:
        raise ValueError("models must not be empty.")

    return st.multiselect(
        "Models",
        models,
        default=models,
    )


def render_activity_chart(distribution: dict[str, int]) -> None:
    """Render activity distribution as a chart."""
    if not distribution:
        st.info("No activity distribution available.")
        return

    st.bar_chart(distribution)


def render_confidence_chart(values) -> None:
    """Render prediction confidence values."""
    validate_signal_data(values)
    st.line_chart(list(values))


def render_prediction_count(count: int) -> None:
    """Render the number of predictions."""
    if count < 0:
        raise ValueError("count must not be negative.")

    st.metric("Predictions", count)


def render_fall_count(count: int) -> None:
    """Render detected fall count."""
    if count < 0:
        raise ValueError("count must not be negative.")

    st.metric("Falls Detected", count)


def render_walking_count(count: int) -> None:
    """Render detected walking count."""
    if count < 0:
        raise ValueError("count must not be negative.")

    st.metric("Walking Events", count)


def render_presence_count(count: int) -> None:
    """Render detected presence count."""
    if count < 0:
        raise ValueError("count must not be negative.")

    st.metric("Presence Events", count)


def render_empty_count(count: int) -> None:
    """Render empty-scene count."""
    if count < 0:
        raise ValueError("count must not be negative.")

    st.metric("Empty Samples", count)


def render_dataset_size(size: int) -> None:
    """Render dataset size."""
    if size < 0:
        raise ValueError("size must not be negative.")

    st.metric("Dataset Size", f"{size:,}")


def render_accuracy_card(accuracy: float | None) -> None:
    """Render model accuracy."""
    st.metric(
        "Accuracy",
        format_percentage(accuracy),
    )


def render_f1_card(f1: float | None) -> None:
    """Render weighted F1 score."""
    st.metric(
        "Weighted F1",
        format_percentage(f1),
    )


def render_fall_f1_card(f1: float | None) -> None:
    """Render fall-class F1 score."""
    st.metric(
        "Fall F1",
        format_percentage(f1),
    )


def render_model_performance_summary(
    accuracy: float | None,
    f1: float | None,
) -> None:
    """Render compact model performance."""
    cols = st.columns(2)

    with cols[0]:
        render_accuracy_card(accuracy)

    with cols[1]:
        render_f1_card(f1)


def render_system_health(
    sensor: str,
    preprocessing: str,
    model: str,
) -> None:
    """Render system health summary."""
    st.markdown("### 💚 System Health")

    cols = st.columns(3)

    with cols[0]:
        st.metric("Sensor", sensor)

    with cols[1]:
        st.metric("Preprocessing", preprocessing)

    with cols[2]:
        st.metric("Model", model)


def render_footer_metadata(version: str, environment: str) -> None:
    """Render dashboard metadata."""
    st.divider()
    st.caption(
        f"QwiSense v{version} • Environment: {environment}"
    )


def render_overview_dashboard(
    activity: str,
    confidence: float | None,
    samples: int,
    model: str,
) -> None:
    """Render the primary QwiSense overview."""
    render_dashboard_header()
    render_dashboard_summary(
        activity=activity,
        confidence=confidence,
        samples=samples,
        model=model,
    )
    render_prediction_status(activity, confidence)


def render_page_title(title: str, subtitle: str = "") -> None:
    """Render a consistent page title."""
    if not title.strip():
        raise ValueError("title must not be empty.")

    st.title(title)

    if subtitle:
        st.caption(subtitle)


def render_breadcrumb(items: list[str]) -> None:
    """Render dashboard breadcrumb navigation."""
    if not items:
        raise ValueError("items must not be empty.")

    st.caption(" / ".join(items))


def render_sidebar_info(version: str) -> None:
    """Render compact sidebar product information."""
    with st.sidebar:
        st.caption(f"QwiSense Dashboard v{version}")
        st.caption("WiFi human sensing")


def render_section_tabs(sections: list[str]) -> str:
    """Render compact section tabs."""
    if not sections:
        raise ValueError("sections must not be empty.")

    return st.radio(
        "View",
        sections,
        horizontal=True,
        label_visibility="collapsed",
    )


def render_info_callout(title: str, message: str) -> None:
    """Render an informational callout."""
    if not title.strip() or not message.strip():
        raise ValueError("title and message must not be empty.")

    st.info(f"**{title}** — {message}")


def render_quick_actions() -> tuple[bool, bool]:
    """Render common dashboard actions."""
    cols = st.columns(2)

    with cols[0]:
        run = st.button(
            "▶ Run inference",
            use_container_width=True,
        )

    with cols[1]:
        refresh = st.button(
            "↻ Refresh",
            use_container_width=True,
        )

    return run, refresh


def render_connection_indicator(connected: bool) -> None:
    """Render sensor connection state."""
    if connected:
        st.success("Sensor connected")
    else:
        st.warning("Sensor disconnected")


def render_processing_indicator(processing: bool) -> None:
    """Render preprocessing state."""
    if processing:
        st.info("Processing CSI data...")
    else:
        st.caption("Processing idle")


def render_inference_indicator(running: bool) -> None:
    """Render model inference state."""
    if running:
        st.info("Running inference...")
    else:
        st.caption("Inference idle")


def render_dashboard_state(
    sensor_connected: bool,
    processing: bool,
    inference_running: bool,
) -> None:
    """Render the current dashboard operating state."""
    st.markdown("### System State")

    cols = st.columns(3)

    with cols[0]:
        st.metric(
            "Sensor",
            "Connected" if sensor_connected else "Offline",
        )

    with cols[1]:
        st.metric(
            "Processing",
            "Active" if processing else "Idle",
        )

    with cols[2]:
        st.metric(
            "Inference",
            "Running" if inference_running else "Idle",
        )


def render_page_title(title: str, subtitle: str = "") -> None:
    """Render a consistent page title."""
    if not title.strip():
        raise ValueError("title must not be empty.")

    st.title(title)

    if subtitle:
        st.caption(subtitle)


def render_breadcrumb(items: list[str]) -> None:
    """Render dashboard breadcrumb navigation."""
    if not items:
        raise ValueError("items must not be empty.")

    st.caption(" / ".join(items))


def render_sidebar_info(version: str) -> None:
    """Render compact sidebar product information."""
    with st.sidebar:
        st.caption(f"QwiSense Dashboard v{version}")
        st.caption("WiFi human sensing")


def render_section_tabs(sections: list[str]) -> str:
    """Render compact section tabs."""
    if not sections:
        raise ValueError("sections must not be empty.")

    return st.radio(
        "View",
        sections,
        horizontal=True,
        label_visibility="collapsed",
    )


def render_info_callout(title: str, message: str) -> None:
    """Render an informational callout."""
    if not title.strip() or not message.strip():
        raise ValueError("title and message must not be empty.")

    st.info(f"**{title}** — {message}")


def render_quick_actions() -> tuple[bool, bool]:
    """Render common dashboard actions."""
    cols = st.columns(2)

    with cols[0]:
        run = st.button(
            "▶ Run inference",
            use_container_width=True,
        )

    with cols[1]:
        refresh = st.button(
            "↻ Refresh",
            use_container_width=True,
        )

    return run, refresh

from src.dashboard.components import render_kpi, render_section_header, render_status_card


def test_dashboard_components_are_importable():
    assert callable(render_kpi)
    assert callable(render_section_header)
    assert callable(render_status_card)


def test_activity_badge_component_is_importable():
    from src.dashboard.components import render_activity_badge

    assert callable(render_activity_badge)


def test_metric_row_component_is_importable():
    from src.dashboard.components import render_metric_row

    assert callable(render_metric_row)


def test_alert_component_is_importable():
    from src.dashboard.components import render_alert

    assert callable(render_alert)


def test_alert_rejects_unknown_level():
    import pytest

    from src.dashboard.components import render_alert

    with pytest.raises(ValueError, match="level must be"):
        render_alert("test", level="unknown")


def test_section_component_is_importable():
    from src.dashboard.components import render_section

    assert callable(render_section)


def test_section_component_is_importable():
    from src.dashboard.components import render_section

    assert callable(render_section)


def test_empty_state_component_is_importable():
    from src.dashboard.components import render_empty_state

    assert callable(render_empty_state)


def test_loading_state_component_is_importable():
    from src.dashboard.components import render_loading_state

    assert callable(render_loading_state)


def test_prediction_card_component_is_importable():
    from src.dashboard.components import render_prediction_card

    assert callable(render_prediction_card)


def test_model_badge_component_is_importable():
    from src.dashboard.components import render_model_badge

    assert callable(render_model_badge)


def test_data_status_component_is_importable():
    from src.dashboard.components import render_data_status

    assert callable(render_data_status)


def test_pipeline_status_component_is_importable():
    from src.dashboard.components import render_pipeline_status

    assert callable(render_pipeline_status)


def test_activity_classes_component_is_importable():
    from src.dashboard.components import render_activity_classes

    assert callable(render_activity_classes)


def test_confidence_metric_component_is_importable():
    from src.dashboard.components import render_confidence_metric

    assert callable(render_confidence_metric)


def test_dashboard_summary_component_is_importable():
    from src.dashboard.components import render_dashboard_summary

    assert callable(render_dashboard_summary)


def test_model_selector_component_is_importable():
    from src.dashboard.components import render_model_selector

    assert callable(render_model_selector)


def test_activity_selector_component_is_importable():
    from src.dashboard.components import render_activity_selector

    assert callable(render_activity_selector)


def test_sample_count_control_is_importable():
    from src.dashboard.components import render_sample_count_control

    assert callable(render_sample_count_control)


def test_inference_button_component_is_importable():
    from src.dashboard.components import render_inference_button

    assert callable(render_inference_button)


def test_refresh_button_component_is_importable():
    from src.dashboard.components import render_refresh_button

    assert callable(render_refresh_button)


def test_last_updated_component_is_importable():
    from src.dashboard.components import render_last_updated

    assert callable(render_last_updated)


def test_progress_component_is_importable():
    from src.dashboard.components import render_progress

    assert callable(render_progress)


def test_warning_banner_component_is_importable():
    from src.dashboard.components import render_warning_banner

    assert callable(render_warning_banner)


def test_success_banner_component_is_importable():
    from src.dashboard.components import render_success_banner

    assert callable(render_success_banner)


def test_technical_details_component_is_importable():
    from src.dashboard.components import render_technical_details

    assert callable(render_technical_details)


def test_confidence_badge_component_is_importable():
    from src.dashboard.components import render_confidence_badge
    assert callable(render_confidence_badge)


def test_dataset_selector_component_is_importable():
    from src.dashboard.components import render_dataset_selector
    assert callable(render_dataset_selector)


def test_window_size_control_is_importable():
    from src.dashboard.components import render_window_size_control
    assert callable(render_window_size_control)


def test_sampling_rate_control_is_importable():
    from src.dashboard.components import render_sampling_rate_control
    assert callable(render_sampling_rate_control)


def test_preprocessing_status_is_importable():
    from src.dashboard.components import render_preprocessing_status
    assert callable(render_preprocessing_status)


def test_model_status_is_importable():
    from src.dashboard.components import render_model_status
    assert callable(render_model_status)


def test_sample_info_component_is_importable():
    from src.dashboard.components import render_sample_info
    assert callable(render_sample_info)


def test_validate_model_name_accepts_known_model():
    from src.dashboard.components import validate_model_name

    assert validate_model_name("SVM", ["SVM", "Random Forest"]) == "SVM"


def test_validate_model_name_rejects_unknown_model():
    import pytest
    from src.dashboard.components import validate_model_name

    with pytest.raises(ValueError, match="Unknown model"):
        validate_model_name("VQC", ["SVM", "Random Forest"])


def test_activity_label_maps_known_classes():
    from src.dashboard.components import activity_label

    assert activity_label(0) == "Empty"
    assert activity_label(1) == "Present"
    assert activity_label(2) == "Walking"
    assert activity_label(3) == "Fall"


def test_activity_label_rejects_unknown_class():
    import pytest
    from src.dashboard.components import activity_label

    with pytest.raises(ValueError, match="Unknown activity class"):
        activity_label(99)


def test_version_badge_is_importable():
    from src.dashboard.components import render_version_badge
    assert callable(render_version_badge)


def test_signal_quality_component_is_importable():
    from src.dashboard.components import render_signal_quality
    assert callable(render_signal_quality)


def test_sensor_status_component_is_importable():
    from src.dashboard.components import render_sensor_status
    assert callable(render_sensor_status)


def test_feature_count_component_is_importable():
    from src.dashboard.components import render_feature_count
    assert callable(render_feature_count)


def test_latency_component_is_importable():
    from src.dashboard.components import render_latency
    assert callable(render_latency)


def test_fall_alert_component_is_importable():
    from src.dashboard.components import render_fall_alert
    assert callable(render_fall_alert)


def test_activity_distribution_component_is_importable():
    from src.dashboard.components import render_activity_distribution
    assert callable(render_activity_distribution)


def test_confidence_threshold_component_is_importable():
    from src.dashboard.components import render_confidence_threshold
    assert callable(render_confidence_threshold)


def test_inference_mode_selector_is_importable():
    from src.dashboard.components import render_inference_mode_selector
    assert callable(render_inference_mode_selector)


def test_model_comparison_header_is_importable():
    from src.dashboard.components import render_model_comparison_header
    assert callable(render_model_comparison_header)


def test_dashboard_help_is_importable():
    from src.dashboard.components import render_dashboard_help
    assert callable(render_dashboard_help)


def test_build_export_filename():
    from src.dashboard.components import build_export_filename

    assert build_export_filename("QwiSense Results", "csv") == "QwiSense_Results.csv"


def test_download_button_component_is_importable():
    from src.dashboard.components import render_download_button

    assert callable(render_download_button)


def test_dataframe_to_csv():
    import pandas as pd

    from src.dashboard.components import dataframe_to_csv

    frame = pd.DataFrame({"activity": ["Walking"], "confidence": [0.9]})
    result = dataframe_to_csv(frame)

    assert isinstance(result, bytes)
    assert b"activity" in result
    assert b"Walking" in result


def test_result_to_json():
    import json

    from src.dashboard.components import result_to_json

    encoded = result_to_json({"activity": "Walking", "confidence": 0.9})
    decoded = json.loads(encoded)

    assert decoded["activity"] == "Walking"
    assert decoded["confidence"] == 0.9


def test_accessible_note_component_is_importable():
    from src.dashboard.components import render_accessible_note

    assert callable(render_accessible_note)


def test_prediction_history_component_is_importable():
    from src.dashboard.components import render_prediction_history

    assert callable(render_prediction_history)


def test_recent_events_component_is_importable():
    from src.dashboard.components import render_recent_events

    assert callable(render_recent_events)


def test_model_metrics_component_is_importable():
    from src.dashboard.components import render_model_metrics

    assert callable(render_model_metrics)


def test_dataset_info_component_is_importable():
    from src.dashboard.components import render_dataset_info

    assert callable(render_dataset_info)


def test_navigation_note_is_importable():
    from src.dashboard.components import render_navigation_note

    assert callable(render_navigation_note)


def test_signal_overview_component_is_importable():
    from src.dashboard.components import render_signal_overview
    assert callable(render_signal_overview)


def test_quantum_model_card_is_importable():
    from src.dashboard.components import render_quantum_model_card
    assert callable(render_quantum_model_card)


def test_classical_model_card_is_importable():
    from src.dashboard.components import render_classical_model_card
    assert callable(render_classical_model_card)


def test_comparison_metrics_is_importable():
    from src.dashboard.components import render_comparison_metrics
    assert callable(render_comparison_metrics)


def test_signal_health_is_importable():
    from src.dashboard.components import render_signal_health
    assert callable(render_signal_health)


def test_inference_summary_is_importable():
    from src.dashboard.components import render_inference_summary
    assert callable(render_inference_summary)


def test_no_data_state_is_importable():
    from src.dashboard.components import render_no_data_state
    assert callable(render_no_data_state)


def test_error_state_is_importable():
    from src.dashboard.components import render_error_state
    assert callable(render_error_state)


def test_control_group_is_importable():
    from src.dashboard.components import render_control_group
    assert callable(render_control_group)


def test_product_summary_is_importable():
    from src.dashboard.components import render_product_summary
    assert callable(render_product_summary)


def test_signal_chart_header_is_importable():
    from src.dashboard.components import render_signal_chart_header
    assert callable(render_signal_chart_header)


def test_signal_statistics_is_importable():
    from src.dashboard.components import render_signal_statistics
    assert callable(render_signal_statistics)


def test_feature_summary_is_importable():
    from src.dashboard.components import render_feature_summary
    assert callable(render_feature_summary)


def test_preprocessing_config_is_importable():
    from src.dashboard.components import render_preprocessing_config
    assert callable(render_preprocessing_config)


def test_signal_mode_selector_is_importable():
    from src.dashboard.components import render_signal_mode_selector
    assert callable(render_signal_mode_selector)


def test_channel_selector_is_importable():
    from src.dashboard.components import render_channel_selector
    assert callable(render_channel_selector)


def test_subcarrier_selector_is_importable():
    from src.dashboard.components import render_subcarrier_selector
    assert callable(render_subcarrier_selector)


def test_visualization_toggle_is_importable():
    from src.dashboard.components import render_visualization_toggle
    assert callable(render_visualization_toggle)


def test_chart_empty_state_is_importable():
    from src.dashboard.components import render_chart_empty_state
    assert callable(render_chart_empty_state)


def test_signal_controls_are_importable():
    from src.dashboard.components import render_signal_controls
    assert callable(render_signal_controls)

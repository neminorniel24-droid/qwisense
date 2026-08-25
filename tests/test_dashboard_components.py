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

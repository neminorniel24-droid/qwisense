from src.dashboard.components import render_kpi, render_section_header, render_status_card


def test_dashboard_components_are_importable():
    assert callable(render_kpi)
    assert callable(render_section_header)
    assert callable(render_status_card)

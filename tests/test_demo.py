from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_demo_loads_and_generates_from_bundled_checkpoint():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert len(app.metric) == 3
    app.button[0].click().run()
    assert not app.exception
    assert app.code[0].value.startswith("Mira found a")


def test_attention_controls_render_another_layer_and_head():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.selectbox[0].set_value(1)
    app.selectbox[1].set_value(3)
    app.run()
    assert not app.exception

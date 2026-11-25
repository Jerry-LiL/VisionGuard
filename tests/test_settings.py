from visionguard.classes import parse_class_names
from visionguard.settings import load_dotenv


def test_parse_class_names():
    assert parse_class_names(None) == ("scratch", "crack", "spot")
    assert parse_class_names(" scratch, crack ") == ("scratch", "crack")


def test_dotenv_does_not_override_existing_variables(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "MODEL_PATH=models/from-file.onnx",
                "CONF_THRESHOLD=0.4",
                "EMPTY=",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MODEL_PATH", "models/already.onnx")
    monkeypatch.delenv("CONF_THRESHOLD", raising=False)
    load_dotenv(env_file)
    import os

    try:
        assert os.environ["MODEL_PATH"] == "models/already.onnx"
        assert os.environ["CONF_THRESHOLD"] == "0.4"
    finally:
        os.environ.pop("CONF_THRESHOLD", None)

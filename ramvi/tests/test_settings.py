import json
import shutil

import pytest

from ramvi_service import settings as st


def test_defaults_network_off(tmp_path):
    s = st.load(tmp_path)
    assert s == st.Settings() and s.network_access is False


def test_save_load_roundtrip_and_no_temp_left(tmp_path):
    s = st.Settings(language="uk", api_port=12000, network_access=True)
    st.save(tmp_path, s)
    assert st.load(tmp_path) == s
    assert [p.name for p in tmp_path.iterdir()] == [st.FILE_NAME]


@pytest.mark.parametrize(
    "bad", [{"language": "de"}, {"api_port": 80}, {"api_port": True}, {"network_access": "yes"}, {"models_dir": " "}]
)
def test_invalid_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        st.Settings(**bad).validate()
    (tmp_path / st.FILE_NAME).write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        st.load(tmp_path)


def test_unknown_keys_ignored(tmp_path):
    (tmp_path / st.FILE_NAME).write_text(json.dumps({"language": "en", "future_option": 1}))
    assert st.load(tmp_path).language == "en"


def test_paths_inside_app_survive_move(tmp_path):
    app = tmp_path / "D_drive" / "RAMVI"
    (app / "my_models").mkdir(parents=True)
    s = st.Settings().with_path(app, "models_dir", app / "my_models")
    assert s.models_dir == "my_models"
    st.save(app / "data", s)
    moved = tmp_path / "E_drive" / "RAMVI"
    shutil.move(str(app), str(moved))
    loaded = st.load(moved / "data")
    assert loaded.resolve(moved, "models_dir") == moved / "my_models"


def test_path_outside_app_stays_absolute(tmp_path):
    app, other = tmp_path / "app", tmp_path / "elsewhere"
    s = st.Settings().with_path(app, "models_dir", other)
    assert s.resolve(app, "models_dir") == other.resolve()

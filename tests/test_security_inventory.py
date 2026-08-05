from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import security_inventory_check as security_check  # noqa: E402


def test_no_tracked_env_file():
    tracked = {p.relative_to(ROOT).as_posix() for p in security_check.tracked_files()}
    assert ".env" not in tracked


def test_no_users_paths_in_active_source_config_or_notebooks():
    assert security_check.find_macos_users_path_offenders(security_check.tracked_files()) == []


def test_macos_users_path_detector_does_not_self_match_but_detects_real_path(tmp_path):
    clean_file = tmp_path / "clean.py"
    clean_file.write_text("prefix = '/' + 'Users' + '/'\n")
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("DATA_PATH = '" + "/" + "Users" + "/" + "example/private.csv'\n")

    assert security_check.find_macos_users_path_offenders([clean_file]) == []
    assert security_check.find_macos_users_path_offenders([bad_file]) == [bad_file.as_posix()]


def test_known_newsapi_credential_absent_without_printing_value():
    assert security_check.find_known_secret_offenders(security_check.tracked_files()) == []


def test_dataset_inventory_yaml_parses_and_schema_is_valid():
    inventory = security_check.validate_dataset_inventory(ROOT / "configs" / "datasets.yaml")
    datasets = inventory["datasets"]
    ids = [entry["id"] for entry in datasets]
    assert len(ids) == len(set(ids))
    assert security_check.REQUIRED_DATASET_IDS <= set(ids)
    for entry in datasets:
        assert isinstance(entry, dict)
        assert security_check.REQUIRED_DATASET_FIELDS <= set(entry)
        assert isinstance(entry["used_by"], list)
        assert isinstance(entry["required"], bool)
        assert isinstance(entry["synthetic_fixture_available"], bool)


def test_env_example_contains_placeholders_only():
    values = security_check.validate_env_example(ROOT / ".env.example")
    assert "NEWS_API_KEY" in values
    assert all(value == "" for value in values.values())

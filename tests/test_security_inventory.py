from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]
KNOWN_NEWSAPI_SHA256 = "d9d80879dc4f5ff3a92a2a2df056cbba65770b74dbf7f9f9ac60299616d2a53d"


def tracked_files():
    import subprocess
    output = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    return [ROOT / line for line in output.splitlines()]


def test_no_tracked_env_file():
    tracked = {p.relative_to(ROOT).as_posix() for p in tracked_files()}
    assert ".env" not in tracked


def test_no_users_paths_in_active_source_config_or_notebooks():
    checked_suffixes = {".py", ".md", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".ipynb"}
    offenders = []
    for path in tracked_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(".git/") or path.suffix not in checked_suffixes:
            continue
        text = path.read_text(errors="ignore")
        if re.search(r"/Users/[^\s\"']+", text):
            offenders.append(rel)
    assert offenders == []


def test_known_newsapi_credential_absent_without_printing_value():
    import hashlib
    offenders = []
    for path in tracked_files():
        if path.is_file() and path.suffix not in {".pdf"}:
            text = path.read_text(errors="ignore")
            for match in re.findall(r"[A-Za-z0-9_\-]{20,}", text):
                if hashlib.sha256(match.encode()).hexdigest() == KNOWN_NEWSAPI_SHA256:
                    offenders.append(path.relative_to(ROOT).as_posix())
                    break
    assert offenders == []


def test_dataset_inventory_yaml_parses():
    text = (ROOT / "configs" / "datasets.yaml").read_text()
    ids = set(re.findall(r"(?m)^\s*-\s+id:\s*([A-Za-z0-9_\-]+)\s*$", text))
    assert ids
    required_ids = {
        "esg_company_data",
        "use_table",
        "make_table",
        "oecd_icio_usa_io",
        "bea_real_value_added",
        "bea_real_intermediate_input",
        "bea_intermediate_input_price_indexes",
        "bea_real_gross_output",
        "bea_gross_output_price_indexes",
        "bea_gross_output",
        "emissions_ghgp_data",
        "environmental_tax_data",
        "optional_news_api_data",
        "optional_sqlite_news_database",
    }
    assert required_ids <= ids
    required_fields = ["id", "description", "expected_path", "provider", "format", "required", "used_by", "redistribution", "synthetic_fixture_available"]
    for field in required_fields:
        assert re.search(rf"(?m)^\s+{field}:", text)


def test_env_example_contains_placeholders_only():
    values = {}
    for line in (ROOT / ".env.example").read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    assert "NEWS_API_KEY" in values
    assert values["NEWS_API_KEY"] == ""
    assert all(value == "" for value in values.values())

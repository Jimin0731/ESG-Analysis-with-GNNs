from pathlib import Path
import pytest, yaml
from src.data.config import dataset_from_mapping, load_dataset_configs
from src.data.contracts import DataValidationError

def base(): return {'id':'x','provider':'p','expected_path':'data/x.csv','format':'csv','required':True}
def test_valid_typed_configuration(): assert dataset_from_mapping(base()).identity.id=='x'
def test_missing_required_configuration_fields():
    b=base(); del b['provider']
    with pytest.raises(DataValidationError): dataset_from_mapping(b)
def test_unsupported_formats():
    b=base(); b['format']='exe'
    with pytest.raises(DataValidationError): dataset_from_mapping(b)
def test_duplicate_dataset_ids(tmp_path):
    p=tmp_path/'d.yaml'; p.write_text(yaml.safe_dump({'datasets':[base(),base()]}))
    with pytest.raises(DataValidationError): load_dataset_configs(p)
def test_repository_relative_example_paths():
    raw=yaml.safe_load(Path('configs/pipeline.example.yaml').read_text())
    assert all(not str(v).startswith('/') for v in raw['inputs'].values())
def test_existing_dataset_inventory_parses(): assert len(load_dataset_configs('configs/datasets.yaml')) >= 1

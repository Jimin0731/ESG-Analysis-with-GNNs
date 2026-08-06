"""Static consolidation checks; archived notebook contents are never parsed or executed."""
from __future__ import annotations
import hashlib
import importlib
import pkgutil
import re
import os
import sys
from pathlib import Path
from unittest.mock import patch
import nbformat
import yaml
from scripts.launch_notebooks import build_notebook_command,build_notebook_environment,main as launch_notebooks
from scripts.run_notebook_smoke import ACTIVE_NOTEBOOKS

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'archive/research'
ARCHIVED={
'integrated_use_make/top 4 update ver.ipynb','integrated_use_make/Epoch 300 + IO data ver2.ipynb','integrated_use_make/Real GDP data + real env data ver3.ipynb','integrated_use_make/API added ver5.ipynb','integrated_use_make/Heterophily ver6.ipynb','integrated_use_make/Attention weight + visualization ver8.ipynb','integrated_use_make/Attention_weight_visualization_FULL.py','integrated_use_make/time series ver9.ipynb','integrated_use_make/ESG Analysis final ver.ipynb','us_icio_bea/Data Preprocess.ipynb','us_icio_bea/header file check.ipynb','us_icio_bea/differentiation top 10.ipynb','auxiliary/db_to_csv.ipynb','auxiliary/graphsage_ppi.py'}
ARCHIVE_ARTIFACTS=ARCHIVED|{'presentation/final_presentation.pdf'}

def manifest():
 rows=[]
 for line in (ARCHIVE/'SHA256SUMS').read_text().splitlines():
  digest,path=line.split('  ',1); rows.append((digest,path))
 return rows

def test_exact_archive_inventory_and_root_cleanup():
 actual={str(p.relative_to(ARCHIVE)) for p in ARCHIVE.rglob('*') if p.is_file() and p.name not in {'README.md','SHA256SUMS'}}
 assert actual==ARCHIVE_ARTIFACTS
 assert not list(ROOT.glob('*.ipynb'))
 assert not (ROOT/'reports').exists()
 assert all(not (ROOT/Path(p).name).exists() for p in ARCHIVED)

def test_checksum_manifest_is_complete_safe_unique_and_byte_exact():
 rows=manifest(); assert len(rows)==len(set(rows))==15
 assert {p for _,p in rows}==ARCHIVE_ARTIFACTS
 root=ARCHIVE.resolve()
 for expected,relative in rows:
  path=(ARCHIVE/relative).resolve()
  assert path.is_relative_to(root) and path.is_file()
  assert hashlib.sha256(path.read_bytes()).hexdigest()==expected

def test_exact_five_valid_clean_active_notebooks():
 assert ACTIVE_NOTEBOOKS==tuple(sorted(ACTIVE_NOTEBOOKS)) and len(ACTIVE_NOTEBOOKS)==5
 assert {p.name for p in (ROOT/'notebooks').glob('*.ipynb')}==set(ACTIVE_NOTEBOOKS)
 for name in ACTIVE_NOTEBOOKS:
  nb=nbformat.read(ROOT/'notebooks'/name,as_version=4); nbformat.validate(nb)
  assert nb.nbformat==4 and nb.metadata.kernelspec.name=='python3'
  assert {'code','markdown'} <= {c.cell_type for c in nb.cells}
  for cell in nb.cells:
   if cell.cell_type=='code': assert cell.execution_count is None and cell.outputs==[]

def test_active_notebooks_use_repo_apis_without_unsafe_scaffolding():
 forbidden=(r'sys\.path\.(?:append|insert)',r'from\s+archive',r'import\s+archive',r'https?://',r'\b(?:wget|curl)\b',r'/(?:Users|home)/',r'load_state_dict\s*\(\s*torch\.load',r'NEWS[_-]?API[_-]?KEY\s*=')
 for name in ACTIVE_NOTEBOOKS:
  text=(ROOT/'notebooks'/name).read_text()
  assert 'from src.' in text
  assert not any(re.search(pattern,text,re.I) for pattern in forbidden)

def test_notebook_readme_registers_all_active_examples():
 text=(ROOT/'notebooks/README.md').read_text()
 assert all(name in text for name in ACTIVE_NOTEBOOKS)

def test_documented_launcher_replaces_bare_jupyter_command():
 for path in (ROOT/'README.md',ROOT/'notebooks/README.md'):
  text=path.read_text()
  assert 'python scripts/launch_notebooks.py' in text
  assert 'jupyter notebook' not in text.lower()

def test_launcher_environment_prepends_root_without_mutating_input():
 base={'PYTHONPATH':'existing/path','UNCHANGED':'yes'}; original=dict(base)
 result=build_notebook_environment(ROOT,base)
 assert result['PYTHONPATH']==str(ROOT)+os.pathsep+'existing/path'
 assert result['UNCHANGED']=='yes' and base==original

def test_launcher_command_uses_current_python_interpreter():
 assert build_notebook_command()==(sys.executable,'-m','notebook')

def test_launcher_runs_from_root_with_prepared_environment_and_returns_code():
 completed=type('Completed',(),{'returncode':23})()
 with patch('scripts.launch_notebooks.build_notebook_environment',return_value={'PYTHONPATH':'prepared'}) as environment,patch('scripts.launch_notebooks.subprocess.run',return_value=completed) as run:
  assert launch_notebooks()==23
 environment.assert_called_once_with(ROOT)
 run.assert_called_once_with((sys.executable,'-m','notebook'),cwd=ROOT,env={'PYTHONPATH':'prepared'},shell=False,check=False)

def test_lineage_schema_paths_and_archive_coverage():
 payload=yaml.safe_load((ROOT/'docs/research_lineage.yaml').read_text())
 required={'component_id','current_paths','primary_archived_sources','secondary_archived_sources','migration_prs','status','intentional_behavior_changes','remaining_caveats'}
 covered=set()
 for entry in payload['components']:
  assert set(entry)==required
  assert entry['status'] in {'migrated','partially migrated','reference only','intentionally excluded'}
  for current in entry['current_paths']: assert (ROOT/current).exists()
  for archived in entry['primary_archived_sources']+entry['secondary_archived_sources']:
   assert archived.startswith('archive/research/') and (ROOT/archived).exists()
   covered.add(archived.removeprefix('archive/research/'))
 assert covered==ARCHIVED

def test_lineage_report_represents_major_active_packages():
 text=(ROOT/'docs/RESEARCH_LINEAGE.md').read_text().lower()
 for group in ('data','graph','feature','target','model','experiment','anomaly','interpret'):
  assert group in text

def test_archive_is_not_importable_and_active_imports_are_inert():
 discovered={m.name for m in pkgutil.iter_modules([str(ROOT)])}
 assert 'archive' not in discovered
 for package in ('src.data','src.graphs','src.features','src.targets','src.models','src.experiments','src.evaluation','src.visualization','examples'):
  importlib.import_module(package)

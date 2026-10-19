import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def cfg(tmp_path):
    """The quickstart config, run in a temp dir with the mock backend."""
    from abstract_atlas.config import Config

    raw = yaml.safe_load((ROOT / "configs" / "quickstart.yaml").read_text())
    raw["workdir"] = str(tmp_path / "run")
    raw["input"]["path"] = str(ROOT / "examples" / "synthetic_abstracts.csv")
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    return Config.load(path)

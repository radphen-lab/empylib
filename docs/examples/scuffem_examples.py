from __future__ import annotations

from pathlib import Path
import numpy as np
import tempfile

from .bootstrap import data_path, ensure_repo_on_path, working_directory

ensure_repo_on_path()

import empylib.scuffem as scf


def make_files_demo():
    lam = np.linspace(0.45, 1.10, 51)
    material_dict = {
        "demo_material": 2.20 + 0.02j + 0 * lam,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with working_directory(tmp_path):
            scf.make_spectral_files(lam, material_dict)
            omega_path = tmp_path / "OmegaList.dat"
            material_path = tmp_path / "demo_material.dat"
            omega_lines = omega_path.read_text(encoding="utf-8").splitlines()
            material_lines = material_path.read_text(encoding="utf-8").splitlines()

    return {
        "lam": lam,
        "omega_lines": omega_lines,
        "material_lines": material_lines,
    }


def read_results_demo():
    data_dir = data_path("tio2_D700nm")
    pft = scf.read_scatter_PFT(str(data_dir / "tio2_D700.AVSCAT.EMTPFT"))
    ave = scf.read_avescatter(str(data_dir / "tio2_D700.AVSCAT.EMTPFT"))
    return {"data_dir": data_dir, "pft": pft, "ave": ave}


def clean_data_demo():
    data_dir = data_path("tio2_D700nm")
    ave = scf.read_avescatter(str(data_dir / "tio2_D700.AVSCAT.EMTPFT"))
    cleaned = scf.clean_data(ave, inplace=False)
    return {"data_dir": data_dir, "cleaned": cleaned}


def run_all():
    return {
        "make_files": make_files_demo(),
        "read_results": read_results_demo(),
        "clean_data": clean_data_demo(),
    }

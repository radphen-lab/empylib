from __future__ import annotations

from .bootstrap import data_path, ensure_repo_on_path

ensure_repo_on_path()

import empylib.spectrometry as spec


def read_uvvis_demo():
    perkin_path = data_path("PerkinElmer_sample.asc")
    dataset = spec.read_uvvis(str(perkin_path))
    return {"path": perkin_path, "dataset": dataset}


def find_samples_demo():
    uvvis_dir = data_path("uvvis_sample_data")
    sample_names = spec.find_uvvis_samples(search_dirs=[str(uvvis_dir)])
    return {"uvvis_dir": uvvis_dir, "sample_names": sample_names}


def sample_demo():
    uvvis_dir = data_path("uvvis_sample_data")
    dataset = spec.sample_uvvis(
        sample="PMMApaint_CaCO3_15vv",
        search_dirs=[str(uvvis_dir)],
    )
    return {"uvvis_dir": uvvis_dir, "dataset": dataset}


def run_all():
    return {
        "read_uvvis": read_uvvis_demo(),
        "find_samples": find_samples_demo(),
        "sample": sample_demo(),
    }

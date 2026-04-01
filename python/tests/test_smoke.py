"""Smoke tests for the Phase-0 Python scaffold."""

from train import PROJECT_NAME
from train.datasets import module_purpose as datasets_purpose
from train.export import module_purpose as export_purpose
from train.losses import module_purpose as losses_purpose
from train.models import module_purpose as models_purpose
from train.trainers import module_purpose as trainers_purpose


def test_project_name_is_defined() -> None:
    assert PROJECT_NAME == "EngineKonzept"


def test_placeholder_packages_describe_their_scope() -> None:
    purposes = [
        datasets_purpose(),
        export_purpose(),
        losses_purpose(),
        models_purpose(),
        trainers_purpose(),
    ]
    assert all(purposes)


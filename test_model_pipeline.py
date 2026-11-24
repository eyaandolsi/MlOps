import pandas as pd
import pytest

from model_pipeline import align_columns, build_model, preprocess


def test_preprocess_age_et_cible():
    df = pd.DataFrame({
        "diabetesMed": ["Yes", "No"],
        "age": ["[0-10)", "[70-80)"],
        "race": [None, "Caucasian"],
        "max_glu_serum": [None, "Norm"],
        "A1Cresult": [None, ">8"],
        "admission_type_id": [1, 2],
        "discharge_disposition_id": [1, 1],
        "admission_source_id": [7, 7],
    })
    out, y = preprocess(df)
    assert list(out["age"]) == [5, 75]
    assert list(y) == [1, 0]
    assert out["race"].iloc[0] == "Inconnu"


def test_align_columns():
    X = pd.DataFrame({"a": [1], "b": [2]})
    res = align_columns(X, ["b", "c"])
    assert list(res.columns) == ["b", "c"]
    assert res["c"].iloc[0] == 0


def test_modele_inconnu():
    with pytest.raises(ValueError):
        build_model("inconnu")

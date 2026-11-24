"""Exemples de tests unitaires."""


def test_example():
    assert 1 + 1 == 2


def test_import_model_pipeline():
    import model_pipeline

    assert hasattr(model_pipeline, 'prepare_data')

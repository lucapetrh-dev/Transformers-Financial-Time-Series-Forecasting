from __future__ import annotations

from forecast.pipeline.transformer_official_backends import canonical_model_name


def test_transformer_hpo_canonical_name_mapping() -> None:
    assert canonical_model_name("itransformer") == "itransformer"
    assert canonical_model_name("patchtst") == "patchtst"

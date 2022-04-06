import compressai.zoo as cai_zoo
import compressai.zoo.image as cai_zoo_img

import deep_compression.zoo.compressai_base as cai_base


def setup_models():
    import deep_compression.models
    from deep_compression.models.compressai import (
        Cheng2020Anchor,
        Cheng2020Attention,
        FactorizedPrior,
        JointAutoregressiveHierarchicalPriors,
        MeanScaleHyperprior,
        ScaleHyperprior,
    )

    model_architectures = {
        "bmshj2018-factorized": FactorizedPrior,
        "bmshj2018-hyperprior": ScaleHyperprior,
        "mbt2018-mean": MeanScaleHyperprior,
        "mbt2018": JointAutoregressiveHierarchicalPriors,
        "cheng2020-anchor": Cheng2020Anchor,
        "cheng2020-attn": Cheng2020Attention,
    }

    for model_name, model_type in model_architectures.items():
        _register_model_copy(
            model_type=model_type,
            model_name=model_name,
            base_model_name=model_name,
        )


def _register_model_copy_compressai(
    model_type, model_name, model_cfg, model_create=None, model_url=None
):
    """Registers a model copy into CompressAI."""

    if model_create is None:

        def model_create_(*args, **kwargs):
            cai_zoo_img._load_model(model_name, *args, **kwargs)

        model_create = model_create_

    cai_zoo.models[model_name] = model_create
    cai_zoo_img.model_architectures[model_name] = model_type
    cai_zoo_img.cfgs[model_name] = model_cfg

    if model_url is not None:
        cai_zoo_img.model_urls[model_name] = model_url


def _register_model_copy(
    model_type, model_name, base_model_name, reuse_pretrained_weights=False
):
    """Registers a model that uses same base architecture."""
    _register_model_copy_compressai(
        model_type=model_type,
        model_name=model_name,
        model_create=cai_base.create_model_creator(
            model_name, base_model_name
        ),
        model_cfg=dict(cai_zoo_img.cfgs[base_model_name]),
        model_url=(
            cai_zoo_img.model_urls[base_model_name]
            if reuse_pretrained_weights
            else None
        ),
    )

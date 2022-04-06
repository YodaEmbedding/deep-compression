import compressai.zoo as cai_zoo
import compressai.zoo.image as cai_zoo_img

import deep_compression.zoo.compressai_base as cai_base
from deep_compression.models import (
    ChannelRemixerFactorizedPrior,
    DecorrFactorizedPrior,
    FactorizedPrior,
    ResidualPredFactorizedPrior,
)


def setup_models():
    _register_model_copy(
        model_type=FactorizedPrior,
        model_name="bmshj2018-factorized",
        base_model_name="bmshj2018-factorized",
    )
    _register_model(
        model_type=ChannelRemixerFactorizedPrior,
        model_name="bmshj2018-factorized-chan-remixer",
    )
    _register_model(
        model_type=DecorrFactorizedPrior,
        model_name="bmshj2018-factorized-batch-chan-decorr",
    )
    _register_model(
        model_type=ResidualPredFactorizedPrior,
        model_name="bmshj2018-factorized-residual-pred",
    )


def _register_model(model_type, model_name):
    """Registers a custom model."""
    cai_zoo_img.model_architectures[model_name] = model_type


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

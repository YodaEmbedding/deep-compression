import compressai.zoo as cai_zoo
import compressai.zoo.image as cai_zoo_img

import deep_compression.zoo.compressai_base as cai_base
from deep_compression.models import FactorizedPrior, FactorizedPriorDecorr


def setup_models():
    _register_model_copy(
        model_type=FactorizedPrior,
        model_name="bmshj2018-factorized",
        base_model_name="bmshj2018-factorized",
    )
    _register_model_copy(
        model_type=FactorizedPriorDecorr,
        model_name="bmshj2018-factorized-batch-chan-decorr",
        base_model_name="bmshj2018-factorized",
    )


def _register_model(
    model_type, model_name, model_cfg, model_factory=None, model_url=None
):
    """Registers a model."""

    if model_factory is None:

        def model_factory_(*args, **kwargs):
            cai_zoo_img._load_model(model_name, *args, **kwargs)

        model_factory = model_factory_

    cai_zoo.models[model_name] = model_factory
    cai_zoo_img.model_architectures[model_name] = model_type
    cai_zoo_img.cfgs[model_name] = model_cfg

    if model_url is not None:
        cai_zoo_img.model_urls[model_name] = model_url


def _register_model_copy(
    model_type, model_name, base_model_name, reuse_pretrained_weights=False
):
    """Registers a model that uses same base architecture."""
    _register_model(
        model_type=model_type,
        model_name=model_name,
        model_factory=cai_base.create_model_factory(
            model_name, base_model_name
        ),
        model_cfg=dict(cai_zoo_img.cfgs[base_model_name]),
        model_url=(
            cai_zoo_img.model_urls[base_model_name]
            if reuse_pretrained_weights
            else None
        ),
    )

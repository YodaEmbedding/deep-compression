import compressai.zoo

from .models import setup_models

setup_models()

models = compressai.zoo.models
model_architectures = compressai.zoo.image.model_architectures

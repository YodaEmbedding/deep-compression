import importlib
import sys

# import compressai.utils

import deep_compression

# from compressai.utils.eval_model.__main__ import main

# utils = {
#     "plot": compressai.utils.plot.__main__,
#     "bench": compressai.utils.bench.__main__,
#     "find_close": compressai.utils.find_close.__main__,
#     "eval_model": compressai.utils.eval_model.__main__,
#     "update_model": compressai.utils.update_model.__main__,
# }


if __name__ == "__main__":
    _, util_name, *argv = sys.argv
    if util_name == "update_and_eval_model":
        from . import update_and_eval_model

        main = update_and_eval_model.main
    else:
        module = importlib.import_module(
            f"compressai.utils.{util_name}.__main__"
        )
        main = module.main

    main(argv)

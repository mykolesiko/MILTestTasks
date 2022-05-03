"""
    main module for prediction util
"""

import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
from src.model import Model



def setup_parser(parser):
    """Setups parser"""
    # parser.set_defaults(callback=callback_analytics)

    parser.add_argument(
        "--data_dir",
        "-dd",
        required=True,
        default=None,
        help="path to data",
        metavar="FPATH",
    )

    parser.add_argument(
        "--model_dir",
        "-md",
        required=True,
        default=None,
        help="path to model",
        metavar="FPATH",
    )

    parser.add_argument(
        "--model_name",
        "-mn",
        required=True,
        default=None,
        help="model name",
        metavar="FPATH",
    )



if __name__ == "__main__":
    parser = ArgumentParser(
        prog="proc for training of model",
        description="proc for training of model",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    setup_parser(parser)
    args = parser.parse_args()
    # if not os.path.exists(arguments.output):
    #     os.mkdir(arguments.output)

    #setup_logging(LOGGER_YAML_DEFAULT, os.path.join(arguments.output, "predict.log"))
    # arguments.callback(arguments)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(args.model_dir, device)
    _, val_jaccard = model.predict(args)
    print(f"jaccard index for text {val_jaccard[0]}")
    print(f"jaccard index for figures {val_jaccard[1]}")




import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import logging

try:
    from .src.base_options import BaseOptions
    from .src import networks
except:
    from src.base_options import BaseOptions
    from src import networks


def main(args):
    options = BaseOptions()
    state = options.get_state(args=args, silence=True)

    # Init the model
    model = networks.get_main_model(state)
    # state.opt.main_model = model
    logging.info('Model Initialized!')

    model.train_self()
    logging.info('Model Trained!')

    if state.INLP:
        logging.info('Run INLP')
        from fairlib.src.networks.INLP import get_INLP_trade_offs
        get_INLP_trade_offs(model, state)

    logging.info('Finished!')


if __name__ == '__main__':
    main(args={})

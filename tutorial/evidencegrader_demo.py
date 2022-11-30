import fairlib


def train(args):
    fairlib.utils.seed_everything(2022)

    options = fairlib.BaseOptions()
    state = options.get_state(args=args, silence=True)
    print(state.num_groups, state.num_classes, state.dataset)
    print(state.hidden_size, state.n_hidden, state.activation_function)
    # Init Model
    model = fairlib.networks.get_main_model(state)
    model.train_self()

from utils.utils import load_check_point, train_model


def continue_train(path, epoch, loss='mse', optimizer='rmsprop', test_size=7, random_state=42, matrics=None,
                   plot=False):
    model = load_check_point(path)
    model._name = model.name + "_continued"
    train_model(model, epoch, loss=loss, optimizer=optimizer, test_size=test_size, random_state=random_state,
                matrics=matrics, plot=plot)

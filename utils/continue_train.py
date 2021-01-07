from utils.utils import load_check_point, train_model


# load the checkpoint and keep training from that model
def continue_train(path, epoch, data, loss='mse', optimizer='rmsprop', matrics=None, show=False):
    model = load_check_point(path)
    model._name = model.name + "_continued"
    train_model(model, epoch, data, loss=loss, optimizer=optimizer, matrics=matrics, show=show)

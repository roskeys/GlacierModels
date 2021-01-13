from utils.utils import load_check_point, train_model


# load the checkpoint and keep training from that model
def continue_train(path, name, epoch, data, loss='mse', optimizer='rmsprop', save_best_only=True, metrics=None, show=False):
    model = load_check_point(path)
    model._name = "Trans_" + model.name + "_" + name
    train_model(model, epoch, data, loss=loss, optimizer=optimizer, save_best_only=save_best_only, metrics=metrics,
                show=show)

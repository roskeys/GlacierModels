import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import re
import importlib
import argparse
from utils import train_model

parser = argparse.ArgumentParser(description='Glacier Model runner')
parser.add_argument("-p", type=str, default=None, help="Path to the model saved, or model to train or history to plot")
parser.add_argument("-e", type=int, default=200, help="Number of training epoch")
parser.add_argument("-l", type=str, default="mse", help="Training loss function")
parser.add_argument("-o", type=str, default="rmsprop", help="Training optimizer")
parser.add_argument("-t", type=int, default=7, help="Test size")
parser.add_argument("-r", type=int, default=42, help="Random state for train test split")
parser.add_argument("-m", type=list, default=['mse'], help="Training matric")
args = parser.parse_args()
if args.p and args.p[-3:] == ".py":
    file_path = args.p
    if os.path.exists(file_path) and os.path.isfile(file_path):
        if file_path[:2] == "./" or  file_path[:2] == ".\\":
            file_path = file_path[2:]
        file_path = file_path[:-3]
        path_list = os.path.split(file_path)
        if path_list[0] == '.':
            path_list = path_list[1:]
        module_path = ".".join(path_list)
        module = importlib.import_module(module_path)
        model = module.getModel(path_list[-1])
        train_model(model, epoch=args.e, loss=args.l, optimizer=args.o, test_size=args.t, random_state=args.r,
                    matrics=['mse'])
else:
    parser.print_help()

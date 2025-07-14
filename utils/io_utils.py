import os, json
import importlib

def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)

    return data

def import_module(string):
    module, cls = string.rsplit('.', 1)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    return import_module(config["type"])(config)

def instantiate_from_dict(config):
    return import_module(config["type"])(**config)

def instantiate_function(path):
    return import_module(path)

def get_local_rank():
    try:
        return int(os.environ["LOCAL_RANK"])
    except:
        print("LOCAL_RANK not found, set to 0")
        return 0

def get_class(config):
    return import_module(config["type"])
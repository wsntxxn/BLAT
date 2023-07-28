import importlib


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def init_obj_from_str(config, **kwargs):
    obj_args = config["args"].copy()
    obj_args.update(kwargs)
    for k in config:
        if k not in ["type", "args"] and isinstance(config[k], dict) and \
            k not in kwargs:
            obj_args[k] = init_obj_from_str(config[k])
    cls = get_obj_from_str(config["type"])
    obj = cls(**obj_args)
    return obj

import os
import torch
import inspect


def get_shape(T, print_non_tensor=False):
    if isinstance(T, torch.Tensor):
        return str(list(T.shape)) + f"({T.dtype})"
    elif isinstance(T, (list, tuple)):
        return ",".join([get_shape(t) for t in T])
    else:
        if print_non_tensor:
            return str(T)
        else:
            return ""


def getPrintHook(name, max_depth=10):
    top_level_frames = len(inspect.stack())

    # the hook signature
    def hook(model, input):
        st = inspect.stack()
        padding = " " * (len(st) - top_level_frames)  # each module call 2 frames
        if max_depth is not None and len(padding) > max_depth:
            return
        stack_list = inspect.stack()
        for i in range(2, len(stack_list)):
            frame, filename, line_number, function_name, lines, index = stack_list[i]
            if not filename.endswith("/module.py"):  # os.getcwd() in filename:
                break

        filename = filename.rsplit(os.getcwd() + "/", 1)[-1]
        filename = filename.rsplit("site-packages", 1)[-1]
        line = "".join(lines).strip()
        debug_str = (
            f"{padding}|_({i}){filename}:{line_number}>>>{line}<-{get_shape(input)}"
        )
        if "nn/modules/container.py" not in filename:
            print(debug_str)

    return hook


module_memory = {}


def register_print_hooks(module, register_at_step=1, pre_hook=True):
    hooks = []
    h = id(module)
    module_memory[h] = module_memory.get(h, 0) + 1
    if module_memory[h] != register_at_step:
        return hooks
    for name, layer in module.named_modules():
        if pre_hook:
            hooks.append(layer.register_forward_pre_hook(getPrintHook(name)))
    return hooks


def filter_dict(dict_to_filter, thing_with_kwargs, skip_keys=[]):
    sig = inspect.signature(thing_with_kwargs)
    filter_keys = [
        param.name
        for param in sig.parameters.values()
        if param.name not in skip_keys and (param.kind != param.VAR_KEYWORD)
        # if param.kind == param.POSITIONAL_OR_KEYWORD and param.name not in skip_keys
    ]
    filtered_dict = {
        filter_key: dict_to_filter[filter_key] for filter_key in filter_keys
    }
    return filtered_dict


def filtered_call(thing_with_kwargs, dict_to_filter, skip_keys=[]):
    return thing_with_kwargs(
        **filter_dict(dict_to_filter, thing_with_kwargs, skip_keys=skip_keys)
    )


def config_call(callable, config, skip_keys=[], **callable_kwargs):
    if hasattr(callable, "prefix"):
        config = config[callable.prefix]

    return callable(
        **filter_dict(
            config, callable, skip_keys=set(skip_keys) | set(callable_kwargs.keys())
        ),
        **callable_kwargs,
    )


def config_call_original(callable, config, pretrained=False, skip_keys=[], **callable_kwargs):
    if not pretrained:
        return callable(**callable_kwargs)
    if hasattr(callable, "prefix"):
        config = config[callable.prefix]

    return callable(
        **filter_dict(
            config, callable, skip_keys=set(skip_keys) | set(callable_kwargs.keys())
        ),
        **callable_kwargs,
    )


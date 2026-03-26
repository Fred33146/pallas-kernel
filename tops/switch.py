from tops.env import TOPS_NATIVE_ENV_NAME, TOPS_USE_NATIVE_MODULES, TOPS_USE_NATIVE_FUNCS
import logging, importlib

logger = logging.getLogger(__name__)

from pathlib import Path
import os

import inspect

def same_signature(f1, f2):
    return inspect.signature(f1) == inspect.signature(f2)


def is_env_true(env):
    env_lower_val = os.environ.get(env,"0").lower()
    return env_lower_val == "1" or env_lower_val == "true"

def get_env_val(env, sep=","):  
    env_lower_val = os.environ.get(env,'').lower()
    return env_lower_val.split(sep)
    

def use_native(func):
    if not is_env_true(TOPS_NATIVE_ENV_NAME):
        return False
    # TOPS_NATIVE=true, check if modules/funcs filters are set
    modules = get_env_val(TOPS_USE_NATIVE_MODULES)
    funcs = get_env_val(TOPS_USE_NATIVE_FUNCS)
    # No filters specified -> globally enable native
    if modules == [''] and funcs == ['']:
        return True
    module_name = func.__module__.lower()
    func_name = func.__name__.lower()
    return module_name in modules or func_name in funcs


def switch_func(func):
    module_path = func.__module__          # "tops.ops.simple_gla.chunk_h"
    func_name = func.__name__              # "chunk_fwd_h"

    if not module_path.startswith("tops."):
        return func
    alt_module_path = "tops.cpu." + module_path[len("tops."):]

    alt_module = importlib.import_module(alt_module_path)
    alt_func = getattr(alt_module, func_name, None)
    if alt_func is None:
        raise AttributeError(f"alter func is None, import module path {alt_module_path}, switch func failed.")
    if not same_signature(func, alt_func):
        raise TypeError(f"function signature between {func.__name__} and {alt_func.__name__} is different, switch func failed.")
    if use_native(func):
        return alt_func
    return func
import os

def getenv_bool(env_var_name):
    env_val = os.getenv(env_var_name)
    assert env_val in ("True", "true", "False", "false"), \
        f"{env_var_name} must be Boolean."
    
    return env_val in ("True", "true")

GPU = getenv_bool("RECEIPT_PROCESS_GPU")
DEBUG = getenv_bool("RECEIPT_PROCESS_DEBUG")
VERBOSE = getenv_bool("RECEIPT_PROCESS_VERBOSE")
craftnet=None
refinenet=None
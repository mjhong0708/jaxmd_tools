import pickle
import warnings
from datetime import datetime
from numbers import Number
from typing import Optional


import numpy as np
import termcolor


def log(msg: str, task: str = "MAIN", filename: Optional[str] = None):
    """Logs a message to the terminal.

    Args:
        msg: Message to log.
        task: Task to log.
        filename: Filename to log to. If None, only logs to stdout.
    """
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg_all = f"[{curr_time}] - {task}\t{msg}"
    msg_colored = termcolor.colored(f"[{curr_time}]", "green")
    msg_colored += f" - {termcolor.colored(task, 'blue')}"
    msg_colored += f"\t{msg}"
    print(msg_colored)
    if filename is not None:
        with open(filename, "a") as f:
            f.write(msg_all + "\n")


# MD simulation utils
def kT(T: float) -> float:
    """
    Converts temperature in Kelvin to kT (eV).
    """
    k_b = 8.617333262145e-5  # Boltzmann constant [eV/K]
    return k_b * T


def kT_inv(kT: float) -> float:
    """
    Converts kT (eV) to temperature in Kelvin.
    """
    k_b = 8.617333262145e-5  # Boltzmann constant [eV/K]
    return kT / k_b


def save_pickle(filename, *args):
    obj = tuple(args)
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj


# Array validation utils
def is_nonempty_array(obj):
    return isinstance(obj, np.ndarray) and obj.size > 0


def is_fractional_coordninates(R, box):
    """Checks if given positions are in fractional coordinates.
    Note that this function does not gaurentee that the positions are
    actually in fractional coordinates, but it prevents from accidentally
    using cartesian coordinates instead of fractional coordinates.

    Args:
        R: Positions to check.

    Returns:
        True if R is in fractional coordinates.
    """
    if isinstance(box, Number) or (box.ndim == 0):
        side_lengths = np.array([box] * 3)
    elif box.ndim == 1:
        side_lengths = box
    else:
        side_lengths = np.linalg.norm(box, axis=1)
    if np.all(side_lengths <= 1):
        warnings.warn("The box is too small, so the result may be inaccurate.")
        return True
    return not np.any(np.logical_or(R < 0, R > 1))

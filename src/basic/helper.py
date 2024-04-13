import gymnasium as gym
import platform
import torch
import os
import re


ROM_TXT = "rom_v5s.txt"


def acceleration() -> str:
    local_machine = platform.system()
    if local_machine == "Darwin":
        if torch.backends.mps.is_available():
            return "mps"
    elif local_machine == "Linux" or "Windows":
        if torch.cuda.is_available():
            return "cuda"
    return "cpu"


def rom_v5s() -> list[str]:
    all_ale_roms: list

    if ~os.path.isfile(ROM_TXT):
        registered_roms = gym.registry.keys()
        matched = lambda a: a if re.search(r"[a-z]+[^ram]-v5\b", a) else None
        roms = list(
            filter(lambda a: a is not None, [matched(x) for x in registered_roms])
        )
        with open("rom_v5s.txt", "w") as file:
            for rom in roms:
                if type(rom) == str:
                    file.write(f"{rom}\n")
        file.close()

    with open(ROM_TXT, "r") as file:
        all_ale_roms = file.read().split("\n")
    file.close()

    return all_ale_roms


def is_even(num: int) -> bool:
    # bitwise for the least significant bit checking.
    return True if num & 1 == 0 else False


def sliding_window():
    """
    Sliding-window method:
        An efficient variant of the 2**k-ary method.
    INPUT:
        An element x of G, a non negative integer n = (n[l-1], n[l-2], ..., n[0]),
    a parameter k > 0 and the precomputed values x**3, x**5, ..., x**(2**k - 1).
    OUTPUT:
        The element x**n belongs to G.
    Algorithm:
        y := 1; i := l - 1
        while i > -1 do
            if n_i = 0 then
                y := y**2' i := i - 1
            else
                s := max{i - k + 1, 0}
                while n_s = 0 do
                    s := s + 1
                for h := 1 to i - s + 1 do
                    y := y**2
                u := (n[i], n[i-1], ..., n[s])
                y := y * x**u
                i := s - 1
        return y
    """
    pass

def montgomery_ladder():
    """
    Montgomery's ladder technique
        Given the binary expansion of a positive, non-zero integer n
    = (n[k-1]...n[0]) with n[k-1] = 1, we can compute x**n as follows
        x1 = x; x2 = x**2
        for i = k - 2 to 0 do
            if n[i] = 0 then
                x2 = x1 * x2; x1 = x1**2
            else
                x1 = x1 * x2; x2 = x2**2
        return x1
    """
    pass

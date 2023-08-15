from ale_py import roms
import gymnasium
import subprocess


gymnasium.logger.set_level(50)
DL_ADDR = "http://www.atarimania.com/roms/Roms.rar"
ROM = roms.__all__[1:]


def env_making(env: str):
    try:
        e = gymnasium.make(id=env)
        del e
        return (env, True)
    except:
        return (env, False)

def envs_summary(envs) -> int:
    installed = 0
    for (e, init) in envs:
        installed += 1 if init == True else 0
        print("{:<20}: {:<5}".format(e, init))
    return installed

def main():
    envs_test = list(map(env_making, ROM)); total = len(envs_test)
    print(f"Registerd ROMs: {total}")
    installed = envs_summary(envs_test)
    print(f"Installed ROMs: {installed}")
    if installed <= 1:
        print("We need download ROMs...")
        subprocess.run(["wget", DL_ADDR])
    else:
        print("ROMs are ready!")

if __name__ == "__main__":
    main()

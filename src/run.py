from basic.helper import acceleration, rom_v5s
from basic.wrapper import En0Wrapper
from basic.preprocess import make_state

all_ale_roms = rom_v5s()

MAXIMUM_TESTING_EPOCHS = 250000
MAXIMUM_TRAINING_EPOCHS = 2500000


def main():
    global all_ale_roms

    device = acceleration()

    envv = En0Wrapper(all_ale_roms[0])

    print(envv)

    observation = envv.reset()
    state = make_state(observation, device)
    print(state)

    for _ in range(10):
        _, _, _, info = envv.step(envv.act())
        for k, v in info.items():
            print(f"{k}: {v}")

    """

    ob0 = np.zeros((len(ob), len(ob[0])), dtype=np.float32)

    local_rw = 0.0
    local_ob = ob0

    for epoch in range(MAXIMUM_EPOCHS):

        action = env.action_space.sample()
        ob, reward, t1, t2, _ = env.step(action)
        local_rw += reward.__float__()

        if t1:
            print(f"LiFe Counter After TERMINAT: {env.unwrapped.ale.lives()}")
        elif t2:
            print(f"LiFe Counter After TRUNCATE: {env.unwrapped.ale.lives()}")
        else:
            if epoch % 5 != 0:
                local_ob = np.fmax(ob0, ob)
            else:
                if epoch % 32 == 0:
                    print(f"{epoch}: {ob[0][0]}, {reward}")
                if t1 or t2:
                    _, _ = env.reset(seed=42)
                break

        local_ob = ob0

    env.close()
    """


if __name__ == "__main__":
    main()

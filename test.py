import multiprocessing as mp

import numpy as np


def target(out_queue):
    out_queue.put(("ok", np.random.randn(4871, 3)))
    return

def main():
    ctx = mp.get_context("spawn")  # safer / more predictable on many platforms
    out_queue = ctx.Queue()
    p = ctx.Process(target=target, args=(out_queue,))
    p.start()
    status, payload = out_queue.get(timeout=20)

    print(payload)
    p.join()
    print(f'{p.is_alive()}')

if __name__ == '__main__':
    main()

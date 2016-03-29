import multiprocessing

pool = None

def load():
    global pool
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool()


def hardresetpool():
    global pool
    pool.terminate()
    pool.join()
    pool = multiprocessing.Pool()

import time

class Timer:
    def __init__(self):
        self.tic = time.time()

    def tac(self):
        duration = time.time() - self.tic
        print('finished in {}s ...'.format(duration))
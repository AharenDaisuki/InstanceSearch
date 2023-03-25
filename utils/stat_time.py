import time

class Timer:
    def __init__(self, name=''):
        self.tic = time.time()
        self.name = name
        print('{} started...'.format(name))

    def tac(self):
        tmp = time.time()
        duration = tmp - self.tic
        print('{} finished in {}s ...'.format(self.name, duration))
        self.tic = tmp
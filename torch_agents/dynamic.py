from copy import deepcopy

class Dynamic(object):
    def __init__(self):
        return

    def __float__(self):
        raise NotImplementedError

    def __int__(self):
        return int(float(self))

class Flat(Dynamic):
    def __init__(self, steps, value):
        self.steps = steps
        self.value = value
        self.done = False
        return

    def __float__(self):
        if not self.done:
            self.steps -= 1
            self.done = (self.steps <= 0)
        return float(self.value)

class Linear(Dynamic):
    def __init__(self, steps, start_value, final_value):
        self.steps = steps
        self.slope = (final_value - start_value) / (steps - 1)
        self.value = start_value - self.slope
        self.done = False
        return

    def __float__(self):
        if not self.done:
            self.steps -= 1
            self.value += self.slope
            self.done = (self.steps <= 0)
        return float(self.value)

class Sawtooth(Dynamic):
    def __init__(self, steps, teeth, start_value, final_value):
        self.teeth = teeth
        self.width = steps//teeth
        self.start_value = start_value
        self.final_value = final_value
        self.start_new_tooth()
        self.done = False
        return

    def start_new_tooth(self):
        self.tooth = Linear(self.width, self.start_value, self.final_value)

    def __float__(self):
        if not self.done:
            self.value = float(self.tooth)
            if self.tooth.done:
                self.teeth -= 1
                if self.teeth <= 0:
                    self.done = True
                else:
                    self.start_new_tooth()

        return float(self.value)

class Sequence(Dynamic):
    def __init__(self, sequence):
        if len(sequence) == 0:
            raise RuntimeError

        self.sequence = sequence
        self.index = 0
        self.done = False
        self.value = None        
        return

    def __float__(self):
        if not self.done:
            self.value = float(self.sequence[self.index])
            if self.sequence[self.index].done:
                self.index += 1
            self.done = (self.index >= len(self.sequence))
        return float(self.value)


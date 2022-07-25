from copy import deepcopy

class Schedule(object):
    def __init__(self):
        self.abstract = True
        return

    def __float__(self):
        raise NotImplementedError

    def __int__(self):
        return int(float(self))

    def advance(self):
        raise NotImplementedError


class Flat(Schedule):
    def __init__(self, steps, value):
        super().__init__()
        self.steps = steps
        self.value = value
        self.done = False
        return

    def __float__(self):
        return float(self.value)

    def advance(self):
        if not self.done:
            self.steps -= 1
            self.done = (self.steps <= 0)

    def __repr__(self):
        return "Flat({self.steps:d}, {self.value:g})".format(self=self)


class Linear(Schedule):
    def __init__(self, steps, start_value, final_value, repeat=1):
        super().__init__()
        self.steps = steps
        self.start_value = start_value
        self.final_value = final_value
        self.repeat = repeat
        self.loops_remaining = self.repeat
        self.done = False
        self.start_loop()
        return

    def start_loop(self):
        self.loops_remaining -= 1
        if self.loops_remaining >= 0:
            self.slope = (self.final_value - self.start_value) / (self.steps - 1)
            self.value = self.start_value
            self.steps_remaining = self.steps
        else:
            # use final_value to correct for rounding error and handle a request with repeat=0
            self.value = self.final_value 
            self.done = True

    def __float__(self):
        return float(self.value)

    def advance(self):
        if not self.done:
            self.steps_remaining -= 1
            if self.steps_remaining <= 0:
                self.value=self.final_value
                self.start_loop()
            else:
                self.value += self.slope

    def __repr__(self):
        if self.repeat > 1:
            return "Linear({self.steps:d}, {self.start_value:g}, {self.final_value:g}, {self.repeat:d})".format(self=self)
        else:
            return "Linear({self.steps:d}, {self.start_value:g}, {self.final_value:g})".format(self=self)


class Sequence(Schedule):
    def __init__(self, contents, repeat=1):
        super().__init__()

        if len(contents) == 0:
            raise RuntimeError

        self.contents = contents
        self.repeat = repeat
        self.done = False
        self.value = None
        self.start_loop()
        return

    def start_loop(self):
        # Create an active copy of the requested abstract schedule
        self.sequence = [deepcopy(s) for s in self.contents]
        self.value = float(self.sequence[0])

    def __float__(self):
        return float(self.value)

    def advance(self):
        if not self.done:
            self.sequence[0].advance()
            if not self.sequence[0].done:
                self.value = float(self.sequence[0])
            else:
                self.sequence.pop(0)
                if len(self.sequence) > 0:
                    self.value = float(self.sequence[0])
                else:
                    self.repeat -= 1
                    if self.repeat > 0:
                        self.start_loop()
                    else:
                        self.done = True


    def __repr__(self):
        if self.repeat > 1:
            return "Sequence({self.contents:}, {self.repeat:d})".format(self=self)
        else:
            return "Sequence({self.contents:})".format(self=self)

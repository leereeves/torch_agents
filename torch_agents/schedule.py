from copy import deepcopy

class Schedule(object):
    def __init__(self):
        self.abstract = True
        return

    def protect_abstract(self):
        if self.abstract:
            raise RuntimeError # missing schedule.asfloat()

    def asfloat(self):
        copy = deepcopy(self)
        copy.abstract = False
        return copy

    def __float__(self):
        raise NotImplementedError

    def __int__(self):
        return int(float(self))

class Flat(Schedule):
    def __init__(self, steps, value):
        super().__init__()
        self.steps = steps
        self.value = value
        self.done = False
        return

    def __float__(self):
        self.protect_abstract()
        if not self.done:
            self.steps -= 1
            self.done = (self.steps <= 0)
        return float(self.value)

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
            self.next_value = self.start_value
            self.steps_remaining = self.steps
        else:
            # use final_value to correct for rounding error and handle a request with repeat=0
            self.next_value = self.final_value 
            self.done = True

    def __float__(self):
        self.protect_abstract()
        if not self.done:
            self.steps_remaining -= 1
            if self.steps_remaining <= 0:
                self.value=self.final_value
                self.start_loop()
            else:
                self.value = self.next_value
                self.next_value += self.slope
        return float(self.value)

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
        self.sequence = [s.asfloat() for s in self.contents]

    def __float__(self):
        self.protect_abstract()
        if not self.done:
            self.value = float(self.sequence[0])
            if self.sequence[0].done:
                self.sequence.pop(0)
            if len(self.sequence) == 0:
                self.repeat -= 1
                if self.repeat > 0:
                    self.start_loop()
                else:
                    self.done = True
        return float(self.value)


    def __repr__(self):
        if self.repeat > 1:
            return "Sequence({self.contents:}, {self.repeat:d})".format(self=self)
        else:
            return "Sequence({self.contents:})".format(self=self)

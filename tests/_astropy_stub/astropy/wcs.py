class NoConvergence(Exception):
    def __init__(self, *args, best_solution=None, divergent=None, slow_conv=None):
        super().__init__(*args)
        self.best_solution = best_solution
        self.divergent = divergent
        self.slow_conv = slow_conv

class FITSFixedWarning(Warning):
    pass

class WCS:
    def __init__(self, *args, **kwargs):
        self.has_celestial = False
    @property
    def celestial(self):
        return self

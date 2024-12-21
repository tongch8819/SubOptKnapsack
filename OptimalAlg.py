from base_task import BaseTask


class OptimalAlg:
    def __init__(self, model: BaseTask):
        self.model = model
        self.alpha = 0.8
        self.opt = None
        pass

    def setAlpha(self, alpha):
        self.alpha = alpha

    def setOpt(self, opt):
        self.opt = opt

    def build(self):
        pass

    def optimize(self):
        pass

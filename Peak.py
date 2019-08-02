class Peak():
    def __init__(self, position: float, area: float, fwhm: float, parameter: float, manual=False):
        self.position = position
        self.area = area
        self.fwhm = fwhm
        self.parameter = parameter
        self.manual = manual

    def __str__(self):
        return f'Peak ['f'pos: {self.position:.2f}, area: {self.area:.2f}, fwhm: {self.fwhm:.2f}, parameter: {self.parameter:.2f}]'

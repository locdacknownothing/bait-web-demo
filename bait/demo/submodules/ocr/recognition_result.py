class RegRes(object):
    def __init__(self, xyxy, text, conf):
        self.xyxy = xyxy
        self.text = text
        self.conf = conf
        self.height = xyxy[3] - xyxy[1]

        self.__validate_attribute__()

    def __str__(self):
        return f"{self.__class__.__name__}(xyxy={self.xyxy!r}, text={self.text!r}, conf={self.conf!r})"

    def __repr__(self):
        return self.__str__()

    def __validate_attribute__(self):
        pass

    def tolist(self):
        return [self.xyxy, self.text, self.conf]

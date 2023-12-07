class Result:
    def __init__(self, data: object or None = None, error: Exception or None = None) -> None:
        self.data = data
        self.error = error
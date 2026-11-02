class StandardScaler:
    with_mean: bool
    with_std: bool

    def __init__(self, *, with_mean: bool = True, with_std: bool = True) -> None: ...

class MinMaxScaler:
    def __init__(self) -> None: ...

class RobustScaler:
    def __init__(self) -> None: ...

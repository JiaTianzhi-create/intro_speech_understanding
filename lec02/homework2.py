def arithmetic(x, y):
    if isinstance(x, str) and isinstance(y, str):
        return x + y
    elif isinstance(x, str) and isinstance(y, float):
        return x * int(y)
    elif isinstance(x, float) and isinstance(y, str):
        return str(x) + y
    elif isinstance(x, float) and isinstance(y, float):
        return x * y

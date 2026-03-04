def time_block_split(
    n: int, train_frac: float, val_frac: float
) -> tuple[slice, slice, slice]:
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val
    train_slice = slice(0, n_train)
    val_slice = slice(n_train, n_train + n_val)
    test_slice = slice(n_train + n_val, n)
    return train_slice, val_slice, test_slice

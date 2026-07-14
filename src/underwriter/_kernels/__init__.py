"""Private numerical core: polars-in / polars-out pure functions.

Every parity-critical calculation lives here exactly once. Kernels do no input
validation and no pandas mirroring -- that is the ``pipeline`` layer's job. They
are the package's numerical truth, tested directly and pinned to the R package's
output.
"""

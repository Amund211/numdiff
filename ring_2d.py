import numpy as np


class Ring2D(np.ndarray):
    """
    A ringbuffer of columns in a 2D numpy array

    shape: pass as you would for a regular ndarray
    context: desired amount of levels of context
    """

    def __new__(cls, *args, shape, context, **kwargs):
        assert len(shape) == 2, "Ring2D only supports 2 dimensions"

        # The size of the internal array
        physical_size = context
        # The size emulated by this interface
        logical_size = shape[1]

        obj = super().__new__(cls, *args, shape=(shape[0], physical_size), **kwargs)

        obj.ring_2d_physical_size = physical_size
        obj.ring_2d_logical_size = logical_size
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            # Called from constructor => properties are set
            return

        self.ring_2d_physical_size = getattr(obj, "ring_2d_physical_size", None)
        self.ring_2d_logical_size = getattr(obj, "ring_2d_logical_size", None)

    def _map_index(self, index):
        """Find the corresponding index to the internal array"""
        if isinstance(index, tuple) and len(index) == 2:
            assert isinstance(
                index[1], int
            ), "Only simple indexing is permitted on the ring index"

            return (
                index[0],
                # Handle negative indicies by first taking mod logical_size
                (index[1] % self.ring_2d_logical_size) % self.ring_2d_physical_size,
            )

        return index

    def __getitem__(self, index):
        return super().__getitem__(self._map_index(index))

    def __setitem__(self, index, value):
        return super().__setitem__(self._map_index(index), value)

import numpy as np
import h5py


def cell_to_point_data(cell_data: np.ndarray) -> np.ndarray:
    nx, ny, nz = cell_data.shape

    padded = np.pad(cell_data, ((1, 1), (1, 1), (1, 1)), mode='edge')

    point_data = (
        padded[0:nx+1, 0:ny+1, 0:nz+1] +
        padded[0:nx+1, 0:ny+1, 1:nz+2] +
        padded[0:nx+1, 1:ny+2, 0:nz+1] +
        padded[0:nx+1, 1:ny+2, 1:nz+2] +
        padded[1:nx+2, 0:ny+1, 0:nz+1] +
        padded[1:nx+2, 0:ny+1, 1:nz+2] +
        padded[1:nx+2, 1:ny+2, 0:nz+1] +
        padded[1:nx+2, 1:ny+2, 1:nz+2]
    ) * 0.125

    return point_data


class VTKHDFImageDataWriter:
    """Temporal VTKHDF ImageData writer."""

    def __init__(self, path, whole_extent, origin, spacing, cell_data_specs=None, point_data_specs=None, direction=None):
        self.path = path
        self.whole_extent = np.array(whole_extent, dtype=np.int64)
        self.origin = np.array(origin, dtype=np.float64)
        self.spacing = np.array(spacing, dtype=np.float64)
        self.direction = np.array(direction, dtype=np.float64) if direction is not None else np.eye(3).flatten()

        self.cell_data_specs = cell_data_specs if cell_data_specs is not None else {}
        self.point_data_specs = point_data_specs if point_data_specs is not None else {}

        self.dims = (
            self.whole_extent[1] - self.whole_extent[0],
            self.whole_extent[3] - self.whole_extent[2],
            self.whole_extent[5] - self.whole_extent[4],
        )
        self.point_dims = (self.dims[0] + 1, self.dims[1] + 1, self.dims[2] + 1)

        self.f = h5py.File(path, 'w')
        self.vtkhdf = None
        self.steps = None
        self.time_values = None
        self.cell_data_group = None
        self.point_data_group = None
        self.cell_datasets = {}
        self.point_datasets = {}
        self.n_steps = 0

        self._setup_file()

    def _setup_file(self):
        self.vtkhdf = self.f.create_group('VTKHDF')

        self.vtkhdf.attrs.create('Type', 'ImageData', dtype=h5py.string_dtype(encoding='ascii'))
        self.vtkhdf.attrs['Version'] = np.array([2, 0], dtype=np.int64)

        self.vtkhdf.attrs['WholeExtent'] = self.whole_extent
        self.vtkhdf.attrs['Origin'] = self.origin
        self.vtkhdf.attrs['Spacing'] = self.spacing
        self.vtkhdf.attrs['Direction'] = self.direction

        self.steps = self.vtkhdf.create_group('Steps')
        self.steps.attrs['NSteps'] = np.int64(0)

        self.time_values = self.steps.create_dataset(
            'Values', shape=(0,), maxshape=(None,), chunks=True, dtype=np.float64
        )

        if self.cell_data_specs:
            self.cell_data_group = self.vtkhdf.create_group('CellData')

            first_field = list(self.cell_data_specs.keys())[0]
            self.cell_data_group.attrs.create('Scalars', first_field, 
                                               dtype=h5py.string_dtype(encoding='ascii'))

            nx, ny, nz = self.dims
            for name, dtype in self.cell_data_specs.items():
                self.cell_datasets[name] = self.cell_data_group.create_dataset(
                    name,
                    shape=(0, nz, ny, nx),
                    maxshape=(None, nz, ny, nx),
                    chunks=(1, nz, ny, nx),
                    dtype=dtype
                )

        if self.point_data_specs:
            self.point_data_group = self.vtkhdf.create_group('PointData')

            first_field = list(self.point_data_specs.keys())[0]
            self.point_data_group.attrs.create('Scalars', first_field, 
                                                dtype=h5py.string_dtype(encoding='ascii'))

            px, py, pz = self.point_dims
            for name, dtype in self.point_data_specs.items():
                self.point_datasets[name] = self.point_data_group.create_dataset(
                    name,
                    shape=(0, pz, py, px),
                    maxshape=(None, pz, py, px),
                    chunks=(1, pz, py, px),
                    dtype=dtype
                )

    def add_timestep(self, time, cell_data=None, point_data=None):
        """Append one step of (nx, ny, nz) arrays."""
        self.n_steps += 1
        self.steps.attrs['NSteps'] = np.int64(self.n_steps)

        self.time_values.resize((self.n_steps,))
        self.time_values[self.n_steps - 1] = time

        if cell_data is not None:
            for name, ds in self.cell_datasets.items():
                if name in cell_data:
                    data = cell_data[name]
                    data_transposed = np.transpose(data, (2, 1, 0))

                    ds.resize((self.n_steps, *ds.shape[1:]))
                    ds[self.n_steps - 1] = data_transposed

        if point_data is not None:
            for name, ds in self.point_datasets.items():
                if name in point_data:
                    data = point_data[name]
                    data_transposed = np.transpose(data, (2, 1, 0))

                    ds.resize((self.n_steps, *ds.shape[1:]))
                    ds[self.n_steps - 1] = data_transposed

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


VTKHDFIOImageDataWriter = VTKHDFImageDataWriter

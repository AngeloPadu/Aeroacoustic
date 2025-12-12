import numpy as np

class GeometryCreator:
    def __init__(self, Cx):
        self.Cx = Cx

    def m2i(self, x_m):
        return int(round(x_m / self.Cx))

    def add_wall_y(self, solid_mask, y_m):
        Ny, Nx = solid_mask.shape
        j = self.m2i(y_m)
        j = max(0, min(Ny-1, j))
        solid_mask[j, :] = 1

    def add_block(self, solid_mask, x_start_m, width_m, height_m, bottom=True):
        Ny, Nx = solid_mask.shape
        i0 = self.m2i(x_start_m)
        iw = max(1, self.m2i(width_m))
        ih = max(1, self.m2i(height_m))

        i0 = max(0, min(Nx-1, i0))
        i1 = max(0, min(Nx, i0+iw))

        if bottom:
            j0 = 0
            j1 = max(0, min(Ny, ih))
        else:
            j0 = max(0, Ny-ih)
            j1 = Ny

        solid_mask[j0:j1, i0:i1] = 1

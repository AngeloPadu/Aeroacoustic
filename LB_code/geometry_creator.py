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

    def add_block(self, solid_mask, x_start_m, width_m, height_m, tripper, position="bottom"):
        Ny, Nx = solid_mask.shape
    
        i0 = self.m2i(x_start_m)
        iw = max(1, self.m2i(width_m))
        ih = max(1, self.m2i(height_m))
    
        i0 = max(0, min(Nx-1, i0))
        i1 = max(0, min(Nx, i0 + iw))
    
        # bottom
        
            
        if position in ("bottom", "both"):
            j0 = 0
            j1 = max(0, min(Ny, ih))
            
            if tripper == "triangle":
                Y, X = np.meshgrid(np.arange(Ny), np.arange(Nx), indexing='ij')

                triangle = (Y <= ih - (2*ih/iw) * np.abs((X+iw) - i0))
                    
                solid_mask[triangle] = 1
                
                triangle = (Y >= (np.max(Y)-ih) + (2*ih/iw) * np.abs((X+iw) - i0))
                    
                solid_mask[triangle] = 1

            else:
                solid_mask[j0:j1, i0:i1] = 1
    
        # top
        if position in ("top", "both"):
            j0 = max(0, Ny - ih)
            j1 = Ny
            
            if tripper == "triangle":
                
                Y, X = np.meshgrid(np.arange(Ny), np.arange(Nx), indexing='ij')
                triangle = ((X-j1)-(X-j0))*(Y-i1)
                    
            else: 
                solid_mask[j0:j1, i0:i1] = 1

    
    def add_cylinder(self,solid_mask,center_x,center_y,radius):
        Ny, Nx = solid_mask.shape
        # cx_c = self.m2i(center_x)
        # cy_c = self.m2i(center_y)
        
        R = self.m2i(radius)
        
        Y, X = np.meshgrid(np.arange(Ny), np.arange(Nx), indexing='ij')
        
        cylinder = (X - center_x)**2 + (Y - center_y)**2 <= R**2
        
        solid_mask[cylinder] = 1
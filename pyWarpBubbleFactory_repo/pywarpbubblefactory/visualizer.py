import os
import numpy as np
import matplotlib.pyplot as plt
from .results import ECResult
from .ansatze.base import Ansatz

class Visualizer:
    def __init__(self, save_dir=None):
        self.save_dir = save_dir
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def _get_z_slice(self, data_array):
        """
        Takes a 3D or 4D array and slices through the center of the Z-axis.
        Assumes shape (..., Nx, Ny, Nz).
        """
        dims = data_array.shape
        if len(dims) == 4: # (Nt, Nx, Ny, Nz)
            nt, nx, ny, nz = dims
            z_center = nz // 2
            return data_array[0, :, :, z_center]
        elif len(dims) == 3: # (Nx, Ny, Nz)
            nx, ny, nz = dims
            z_center = nz // 2
            return data_array[:, :, z_center]
        else:
            return data_array # Base case if already 2D

    def _plot_heatmap(self, data_2d, title, filename, cmap='RdBu_r', center_zero=True):
        plt.figure(figsize=(8, 6))
        
        if center_zero:
            # Symmetrical color limits so white is zero
            limit = np.max(np.abs(data_2d))
            # If everything is zero, set a tiny limit
            if limit == 0: limit = 1e-10
            vmin, vmax = -limit, limit
        else:
            vmin, vmax = np.min(data_2d), np.max(data_2d)
            
        plt.imshow(data_2d.T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        plt.colorbar(label='Value (Geometrized Units)')
        plt.title(title)
        plt.xlabel("X Grid")
        plt.ylabel("Y Grid")
        
        plt.tight_layout()
        
        if self.save_dir:
            path = os.path.join(self.save_dir, filename)
            plt.savefig(path, dpi=300)
            print(f"📊 Saved Plot: {path}")
            plt.close()
        else:
            plt.show()

    def render_analysis(self, result: ECResult):
        """
        Parses the complete ADM 3+1 Physics tensors and saves the physical observations.
        Matches the standard plotting suite of the original WarpFactory.
        """
        print("\n[Visualizer] Generating 2D Tensor Slices (Z=0 plane)...")
        
        # 1. Eulerian Energy Density (rho = T^00)
        rho_2d = self._get_z_slice(result.rho_map)
        self._plot_heatmap(rho_2d, r"Eulerian Energy Density ($\rho = T^{00}$)", "density_T00.png", cmap='viridis', center_zero=False)
        
        # 2. Warp Condition Penalties (WEC and NEC)
        wec_2d = self._get_z_slice(result.WEC_map)
        nec_2d = self._get_z_slice(result.NEC_map)
        
        # For energy conditions, negative values are the explicit violations.
        self._plot_heatmap(wec_2d, r"Weak Energy Condition (WEC)", "wec_map.png", cmap='coolwarm', center_zero=True)
        self._plot_heatmap(nec_2d, r"Null Energy Condition (NEC)", "nec_map.png", cmap='coolwarm', center_zero=True)
        
        # 3. Structural Observer Components from T_euler
        if result.T_euler is not None:
            # result.T_euler shape (4, 4, Nt, Nx, Ny, Nz)
            T01_2d = self._get_z_slice(result.T_euler[0, 1])
            self._plot_heatmap(T01_2d, r"Momentum Flux Density ($J_x = T^{01}$)", "momentum_T01.png", cmap='PuOr', center_zero=True)
            
            T11_2d = self._get_z_slice(result.T_euler[1, 1])
            self._plot_heatmap(T11_2d, r"Stress / Pressure ($P_{xx} = T^{11}$)", "stress_T11.png", cmap='Spectral_r', center_zero=False)
        else:
            print("[Visualizer] Warning: Full T_euler tensor not found in results. Skipping structural components.")

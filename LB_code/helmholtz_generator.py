#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 16:52:49 2026

@author: angelo
"""

"""
helmholtz_resonator.py
Utilities to add a Helmholtz resonator to an LBM channel
"""

import numpy as np

# ============================================================
# GEOMETRY BUILDER
# ============================================================

def add_helmholtz_resonator(
    solid_mask,
    channel_height,
    neck_width,
    neck_length,
    cavity_width,
    cavity_height,
):
    """
    Modify solid_mask to add a Helmholtz resonator at the right end.

    Parameters
    ----------
    solid_mask : 2D uint8 array
        Existing mask (1 = solid, 0 = fluid)
    channel_height : int
        Height of the channel
    neck_width : int
        Width of the neck opening
    neck_length : int
        Length of the neck
    cavity_width : int
        Width of the cavity
    cavity_height : int
        Height of the cavity
    """

    Ny, Nx = solid_mask.shape

    # Start at right boundary
    x_start = Nx - neck_length - cavity_width
    
    # Channel center
    y_center = channel_height // 2

    # Neck vertical position
    neck_y0 = y_center - neck_width // 2
    neck_y1 = neck_y0 + neck_width
    
    # -------------------------
    # Build neck
    # -------------------------
    for x in range(Nx - neck_length - cavity_height, Nx - cavity_height):
        solid_mask[:neck_y0, x] = 1
        solid_mask[neck_y1:, x] = 1

    # -------------------------
    # Build cavity
    # -------------------------
    cav_x0 = Nx - cavity_height
    cav_x1 = Nx 
    

    cav_y0 = y_center - cavity_width // 2
    cav_y1 = cav_y0 + cavity_width
    
    
    # Fill walls around cavity
    solid_mask[:cav_y0, cav_x0:cav_x1] = 1
    solid_mask[cav_y1:, cav_x0:cav_x1] = 1

    # Close right wall
    solid_mask[:, Nx-1] = 1

    return solid_mask


# ============================================================
# PHYSICS UTILITIES
# ============================================================

def helmholtz_frequency(cs, neck_width, cavity_width, cavity_height,
                        neck_length, dx=1.0):
    """
    Estimate Helmholtz resonance frequency in lattice units.

    Parameters
    ----------
    cs : float
        Lattice sound speed
    neck_width : int
    cavity_width : int
    cavity_height : int
    neck_length : int
    dx : float
        lattice spacing

    Returns
    -------
    f0 : float
    """

    A = neck_width * dx
    V = cavity_width * cavity_height * dx**2
    L_eff = neck_length * dx + 0.8 * dx  # end correction

    f0 = (cs / (2*np.pi)) * np.sqrt(A / (V * L_eff))
    return f0

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 15:04:05 2026

@author: angelo
"""

import numpy as np


def field_saver(X,Y,grid,rho,rho0,ux,uy, case_name, snap_id):
    
    p = rho - rho0
    
    points = np.c_[X.ravel(order='C'),
                   Y.ravel(order='C'),
                   np.zeros(X.size)]
    
    
    grid.point_data['rho'] = rho.ravel(order='C')
    grid.point_data['ux'] = ux.ravel(order='C') 
    grid.point_data['uy'] = uy.ravel(order='C')
    grid.point_data['pressure'] = p.ravel(order='C')
    
    filename = f"t{case_name}_{snap_id:04d}.vts"
    grid.save(filename)
    
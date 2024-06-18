# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 04:29:53 2024

@author: mazzac3
"""

from DJAssistant import DanceDJ
import matplotlib.pyplot as plt
import numpy as np

vals = DanceDJ().generate_sinusoidal_profile(
        tempo_bounds=(70, 120),
        n_cycles=5, 
        horizontal_shift=0, 
        n_songs=25,
        n_points=100
    )

plt.plot(vals[:,1])
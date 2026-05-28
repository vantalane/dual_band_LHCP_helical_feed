#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
 Helical Antenna Tutorial

 Tested with
  - python 3.13
  - openEMS v0.0.36+

 (c) 2015-2025 Thorsten Liebig <thorsten.liebig@gmx.de>
      04-Jan-2026: modified to use matplotlib.pyplot instead of pylab
 (c) 2026 Mete Han Keskin <mete@kestech.net>
 SPDX-License-Identifier: GPL-3.0-or-later
"""

### Import Libraries
import os, tempfile
import numpy as np
import matplotlib.pyplot as plt  # pip install matplotlib

from CSXCAD import CSXCAD

from openEMS import openEMS
from openEMS.physical_constants import *


### Setup the simulation
Sim_Path = os.path.join(tempfile.gettempdir(), 'Helical_Anten')

unit = 1e-3 # all length in mm

f0 = 2.2e9 # center frequency, frequency of interest!
lambda0 = round(C0/f0/unit) # wavelength in mm
fc = 0.5e9 # 20 dB corner frequency

Helix_radius = 43.4/2 		# --> diameter is ~ lambda/pi, then d/2 = r
Helix_turns = 4.242  			# --> expected gain is G ~ 4 * 10 = 40 (16dBi)
Helix_pitch = 29.18  			# mm
wire_diameter = 2.7  		# mm
Helix_mesh_res = wire_diameter/2

gnd_shape     = 'square'  # 'square' or 'circle'
gnd_half_size = 65        # half-side (square) or radius (circle), mm
gnd_thickness = 1.5       # mm

# feeding
feed_heigth = 4
feed_R = 50    #feed impedance

# size of the simulation box
SimBox = np.array([1, 1, 1.5])*2.0*lambda0

### Setup FDTD parameter & excitation function
FDTD = openEMS(EndCriteria=1e-4)
FDTD.SetGaussExcite( f0, fc )
FDTD.SetBoundaryCond( ['MUR', 'MUR', 'MUR', 'MUR', 'MUR', 'PML_8'] )

### Setup Geometry & Mesh
CSX = CSXCAD.ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)

max_res = np.floor(C0 / (f0+fc) / unit / 20) # cell size: lambda/20

# create helix mesh
mesh.AddLine('x', [-Helix_radius, 0, Helix_radius])
mesh.SmoothMeshLines('x', Helix_mesh_res)
# add ground plate edges and air-box
mesh.AddLine('x', [-gnd_half_size, gnd_half_size, -SimBox[0]/2, SimBox[0]/2])
# create a smooth mesh between specified fixed mesh lines
mesh.SmoothMeshLines('x', max_res, ratio=1.4)

# copy x-mesh to y-direction
mesh.SetLines('y', mesh.GetLines('x'))

# create helix mesh in z-direction
mesh.AddLine('z', [-gnd_thickness, 0, feed_heigth, Helix_turns*Helix_pitch+feed_heigth])
mesh.SmoothMeshLines('z', Helix_mesh_res)

# add the air-box
mesh.AddLine('z', [-SimBox[2]/2, max(mesh.GetLines('z'))+SimBox[2]/2 ])
# create a smooth mesh between specified fixed mesh lines
mesh.SmoothMeshLines('z', max_res, ratio=1.4)

### Create the Geometry
## * Create the metal helix using the wire primitive.
## * Create a metal ground plane as a box.
helix_metal = CSX.AddMaterial('helix', kappa=5.8e7)  # copper conductivity

num_pts = round(Helix_turns * 20) + 1  # ~20 segments per turn
ang = np.linspace(0, Helix_turns * 2 * np.pi, num_pts)
Helix_x = Helix_radius * np.cos(ang)
Helix_y = Helix_radius * np.sin(ang)
Helix_z = ang / (2 * np.pi) * Helix_pitch + feed_heigth

p = np.array([Helix_x, Helix_y, Helix_z])
helix_metal.AddWire(p, wire_diameter / 2)

# aluminium ground reflector
gnd = CSX.AddMaterial('gnd', kappa=3.77e7)  # aluminium conductivity

if gnd_shape == 'square':
    gnd.AddBox([-gnd_half_size, -gnd_half_size, -gnd_thickness],
               [ gnd_half_size,  gnd_half_size,  0])
elif gnd_shape == 'circle':
    gnd.AddCylinder([0, 0, -gnd_thickness], [0, 0, 0], gnd_half_size)
else:
    raise ValueError(f"gnd_shape must be 'square' or 'circle', got '{gnd_shape}'")

# apply the excitation & resist as a current source
start = [Helix_radius, 0, 0]
stop  = [Helix_radius, 0, feed_heigth]
port = FDTD.AddLumpedPort(1 ,feed_R, start, stop, 'z', 1.0, priority=5)

# nf2ff calc
nf2ff = FDTD.CreateNF2FFBox(opt_resolution=[lambda0/15]*3)

# E-field time dump through XZ plane (y=0) — open Et*.vtr files in ParaView as a sequence
Et = CSX.AddDump('Et', dump_type=0, file_type=1)
Et.AddBox([-SimBox[0]/2, 0, -SimBox[2]/2],
          [ SimBox[0]/2, 0,  Helix_turns*Helix_pitch + feed_heigth + SimBox[2]/2])

### Preview geometry in AppCSXCAD (blocks until window is closed)
CSX_file = os.path.join(Sim_Path, 'helix.xml')
if not os.path.exists(Sim_Path):
    os.mkdir(Sim_Path)
CSX.Write2XML(CSX_file)
from CSXCAD import AppCSXCAD_BIN
os.system(AppCSXCAD_BIN + ' "{}"'.format(CSX_file))

### Run the simulation
if not os.path.exists(Sim_Path):
    os.mkdir(Sim_Path)
FDTD.Run(Sim_Path, cleanup=True)

### Postprocessing & plotting
freq = np.linspace( f0-fc, f0+fc, 501 )
port.CalcPort(Sim_Path, freq)

Zin = port.uf_tot / port.if_tot
s11 = port.uf_ref / port.uf_inc

## Plot the feed point impedance

fig, axis = plt.subplots(num="Zin", tight_layout=True)
axis.plot(freq/1e6, np.real(Zin), 'k-',  linewidth=2, label='$\\Re(Z_{in})$')
axis.plot(freq/1e6, np.imag(Zin), 'r--', linewidth=2, label='$\\Im(Z_{in})$')
axis.grid()
axis.set_xmargin(0)
axis.set_xlabel('frequency (MHz)')
axis.set_ylabel('Zin (Ohm)')
axis.set_title("feed point impedance")
axis.legend()


## Plot reflection coefficient S11
fig, axis = plt.subplots(num="S11", tight_layout=True)
axis.plot(freq/1e6, 20*np.log10(abs(s11)), 'k-',  linewidth=2)
axis.grid()
axis.set_xmargin(0)
axis.set_xlabel('frequency (MHz)')
axis.set_ylabel('S11 (dB)')
axis.set_title('reflection coefficient $S_{11}$' )


### Create the NFFF contour
## * calculate the far field at phi=0 degrees and at phi=90 degrees
theta = np.arange(0.,180.,1.)
phi = np.arange(-180,180,2)
print( 'calculating the 3D far field...' )

nf2ff_res = nf2ff.CalcNF2FF(Sim_Path, f0, theta, phi, read_cached=False, verbose=True )

Dmax_dB = 10*np.log10(nf2ff_res.Dmax[0])
E_norm = 20.0*np.log10(nf2ff_res.E_norm[0]/np.max(nf2ff_res.E_norm[0])) + 10*np.log10(nf2ff_res.Dmax[0])

theta_HPBW = theta[ np.where(np.squeeze(E_norm[:,phi==0])<Dmax_dB-3)[0][0] ]

## * Display power and directivity
print('radiated power: Prad = {:.5g} W'.format(nf2ff_res.Prad[0]))
print('directivity: Dmax = {:.2f} dBi'.format(Dmax_dB))
print('efficiency: nu_rad = {:.1f} %'.format(100*nf2ff_res.Prad[0]/np.interp(f0, freq, port.P_acc)))
print('theta_HPBW = {:.1f} °'.format(theta_HPBW))

## Export 3D radiation pattern to VTK for ParaView
E_norm_lin = nf2ff_res.E_norm[0]
E_norm_max = np.max(E_norm_lin)
Dmax_lin   = nf2ff_res.Dmax[0]

theta_rad_g = np.deg2rad(theta)
phi_rad_g   = np.deg2rad(phi)
T, P = np.meshgrid(theta_rad_g, phi_rad_g, indexing='ij')

# directivity per direction (linear); radius of the balloon surface
D_lin   = Dmax_lin * (E_norm_lin   / E_norm_max)**2
D_cprh  = Dmax_lin * (np.abs(nf2ff_res.E_cprh[0]) / E_norm_max)**2
D_cplh  = Dmax_lin * (np.abs(nf2ff_res.E_cplh[0]) / E_norm_max)**2

R = D_lin / Dmax_lin  # normalise so peak = 1
X = R * np.sin(T) * np.cos(P)
Y = R * np.sin(T) * np.sin(P)
Z = R * np.cos(T)

ntheta_v, nphi_v = len(theta), len(phi)
vtk_path = os.path.join(Sim_Path, 'pattern_3d.vtk')
with open(vtk_path, 'w') as fv:
    fv.write("# vtk DataFile Version 2.0\n")
    fv.write(f"3D Radiation Pattern {nf2ff_res.freq[0]/1e9:.3f} GHz\n")
    fv.write("ASCII\n")
    fv.write("DATASET STRUCTURED_GRID\n")
    fv.write(f"DIMENSIONS {ntheta_v} {nphi_v} 1\n")
    fv.write(f"POINTS {ntheta_v * nphi_v} float\n")
    for j in range(nphi_v):
        for i in range(ntheta_v):
            fv.write(f"{X[i,j]:.6f} {Y[i,j]:.6f} {Z[i,j]:.6f}\n")
    npts = ntheta_v * nphi_v
    fv.write(f"\nPOINT_DATA {npts}\n")
    for label, data in [('directivity_total_dBi', D_lin),
                        ('directivity_CPRH_dBi',  D_cprh),
                        ('directivity_CPLH_dBi',  D_cplh)]:
        fv.write(f"SCALARS {label} float 1\nLOOKUP_TABLE default\n")
        for j in range(nphi_v):
            for i in range(ntheta_v):
                fv.write(f"{10*np.log10(max(data[i,j], 1e-30)):.4f}\n")
print(f"3D pattern written to: {vtk_path}")

E_norm = 20.0*np.log10(nf2ff_res.E_norm[0]/np.max(nf2ff_res.E_norm[0])) + 10*np.log10(nf2ff_res.Dmax[0])
E_CPRH = 20.0*np.log10(np.abs(nf2ff_res.E_cprh[0])/np.max(nf2ff_res.E_norm[0])) + 10*np.log10(nf2ff_res.Dmax[0])
E_CPLH = 20.0*np.log10(np.abs(nf2ff_res.E_cplh[0])/np.max(nf2ff_res.E_norm[0])) + 10*np.log10(nf2ff_res.Dmax[0])

## * Plot the pattern

fig, axis = plt.subplots(num="Pattern", tight_layout=True)
axis.plot(theta, E_norm[:,phi==0], 'k-',  linewidth=2, label='|E total|')
axis.plot(theta, E_CPRH[:,phi==0], 'r--',  linewidth=2, label='|E CPRH|')
axis.plot(theta, E_CPLH[:,phi==0], 'g-.',  linewidth=2, label='|E CPLH|')
axis.grid()
axis.set_xmargin(0)
axis.set_xlabel('theta (deg)')
axis.set_ylabel('directivity (dBi)')
axis.set_title('Frequency: {:.2f} GHz'.format(nf2ff_res.freq[0]/1e9))
axis.legend()


## * Polar radiation pattern (phi=0 cut, mirrored for full 360°)
fig, axis = plt.subplots(num="Pattern_Polar", subplot_kw={'projection': 'polar'}, tight_layout=True)
axis.set_theta_zero_location('N')   # 0° (main beam) at top
axis.set_theta_direction(-1)        # clockwise, matching theta convention

theta_rad = np.deg2rad(theta)

e_total = np.squeeze(E_norm[:, phi==0])
e_cprh  = np.squeeze(E_CPRH[:, phi==0])
e_cplh  = np.squeeze(E_CPLH[:, phi==0])

# Mirror phi=0 cut to fill the left half and close the circle
angles_full  = np.concatenate([theta_rad, 2*np.pi - theta_rad[::-1]])
e_total_full = np.concatenate([e_total, e_total[::-1]])
e_cprh_full  = np.concatenate([e_cprh,  e_cprh[::-1]])
e_cplh_full  = np.concatenate([e_cplh,  e_cplh[::-1]])

# Clip to a 40 dB dynamic range so the plot stays readable
min_level = Dmax_dB - 40
axis.plot(angles_full, np.maximum(e_total_full, min_level), 'k-',  linewidth=2, label='|E total|')
axis.plot(angles_full, np.maximum(e_cprh_full,  min_level), 'r--', linewidth=2, label='|E CPRH|')
axis.plot(angles_full, np.maximum(e_cplh_full,  min_level), 'g-.', linewidth=2, label='|E CPLH|')

axis.set_rlim([min_level, Dmax_dB + 2])
axis.set_title('Radiation Pattern (phi=0° cut) — {:.2f} GHz'.format(nf2ff_res.freq[0]/1e9), pad=15)
axis.legend(loc='lower right')


# show all plots
plt.show()

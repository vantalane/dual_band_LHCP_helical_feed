#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
 Dual-Band Helical Antenna Simulation — cached / iterative variant
 Concentric L-band (1.7 GHz) and S-band (2.2 GHz) helices on a shared reflector.

 Based on the OpenEMS Helical Antenna Tutorial by Thorsten Liebig.

 Tested with
  - python 3.13
  - openEMS v0.0.36+

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


### Simulation control
# Set to True to delete cached data and re-run even if results exist
force_rerun = True

# fast_mode = True  : smaller box (1.5λ), lambda/15 mesh, EndCriteria 1e-3, sweep only the active band
# fast_mode = False : full box  (2.0λ), lambda/20 mesh, EndCriteria 1e-4, sweep both bands
fast_mode = True

unit = 1e-3 # all length in mm

# Choose driven antenna: 'L' = 1.7 GHz L-band, 'S' = 2.2 GHz S-band
selected_antenna = 'S'

# What to do with unused antenna feed:
#   'short' = connect helix start to GND (reflector) with a metal stub
#   'float' = leave helix wire end open / unterminated
unused_antenna = 'short'
# unused_antenna = 'float'


# Each config gets its own directory so results are never clobbered
_mode_tag = 'fast' if fast_mode else 'full'
Sim_Path = os.path.join(tempfile.gettempdir(),
                        f'Helical_Anten_doublebnd_{selected_antenna}_{unused_antenna}_{_mode_tag}')

f0 = 1.7e9 # center frequency of L band
f1 = 2.2e9 # center frequency of S band
fc = 0.5e9 # Gaussian pulse bandwidth (20 dB corner)

lambda0 = round(C0/f0/unit) # wavelength in mm at L-band
lambda1 = round(C0/f1/unit) # wavelength in mm at S-band

# Active antenna centre frequency / wavelength
if selected_antenna == 'L':
    f_center      = f0
    lambda_center = lambda0
elif selected_antenna == 'S':
    f_center      = f1
    lambda_center = lambda1
else:
    raise ValueError(f"selected_antenna must be 'L' or 'S', got '{selected_antenna}'")

# --- L-band helix parameters ---
Helix_radius = 28    # mm  (~lambda0 / pi / 2)
Helix_turns  = 5.5
Helix_pitch  = 36    # mm
wire_diameter = 2.7  # mm

# --- S-band helix parameters (concentric, smaller) ---
helix_radius_s_band  = 21.68  # mm
helix_turns_s_band   = 4.242
helix_pitch_s_band   = 29.18  # mm
wire_diameter_s_band = 2.7    # mm

# Mesh resolution set by the finer of the two wire diameters
Helix_mesh_res = min(wire_diameter, wire_diameter_s_band) / 2

gnd_shape     = 'square'  # 'square' or 'circle'
gnd_half_size = 65        # half-side (square) or radius (circle), mm
gnd_thickness = 1.5       # mm

feed_heigth = 4   # feed stub height: ground surface (z=0) to helix start, mm
feed_R      = 50  # feed impedance, Ohm

# Helix top z-coordinates
helix_L_top = Helix_turns       * Helix_pitch      + feed_heigth
helix_S_top = helix_turns_s_band * helix_pitch_s_band + feed_heigth

if fast_mode:
    # Smaller box focused on active antenna wavelength; trades ~5–10% accuracy for ~3–5x speed
    SimBox        = np.array([1, 1, 1.2]) * 1.5 * lambda_center
    end_criteria  = 1e-3   # fields must decay to 0.1 % of peak (vs 0.01 % full)
    mesh_lambda_d = 15     # lambda / 15 spatial resolution
else:
    SimBox        = np.array([1, 1, 1.5]) * 2.0 * lambda0
    end_criteria  = 1e-4
    mesh_lambda_d = 20

### Setup FDTD parameter & excitation function
FDTD = openEMS(EndCriteria=end_criteria)
FDTD.SetGaussExcite(f_center, fc)
FDTD.SetBoundaryCond(['MUR', 'MUR', 'MUR', 'MUR', 'MUR', 'PML_8'])

### Setup Geometry & Mesh
CSX = CSXCAD.ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)

max_res = np.floor(C0 / (f_center + fc) / unit / mesh_lambda_d)

# --- x-mesh: include both helix radii ---
mesh.AddLine('x', [-Helix_radius, -helix_radius_s_band, 0,
                    helix_radius_s_band,  Helix_radius])
mesh.SmoothMeshLines('x', Helix_mesh_res)
# ground plate edges and air-box
mesh.AddLine('x', [-gnd_half_size, gnd_half_size, -SimBox[0]/2, SimBox[0]/2])
mesh.SmoothMeshLines('x', max_res, ratio=1.4)

# copy x-mesh to y-direction
mesh.SetLines('y', mesh.GetLines('x'))

# --- z-mesh: include both helix tops ---
mesh.AddLine('z', [-gnd_thickness, 0, feed_heigth, helix_S_top, helix_L_top])
mesh.SmoothMeshLines('z', Helix_mesh_res)
# air-box
mesh.AddLine('z', [-SimBox[2]/2, max(mesh.GetLines('z')) + SimBox[2]/2])
mesh.SmoothMeshLines('z', max_res, ratio=1.4)

### Create the Geometry

# --- L-band helix (copper) ---
helix_L_metal = CSX.AddMaterial('helix_L', kappa=5.8e7)
num_pts_L = round(Helix_turns * 20) + 1
ang_L     = np.linspace(0, Helix_turns * 2 * np.pi, num_pts_L)
Helix_L_x = Helix_radius * np.cos(ang_L)
Helix_L_y = Helix_radius * np.sin(ang_L)
Helix_L_z = ang_L / (2 * np.pi) * Helix_pitch + feed_heigth
helix_L_metal.AddWire(np.array([Helix_L_x, Helix_L_y, Helix_L_z]),
                      wire_diameter / 2)

# --- S-band helix (copper), concentric inside L-band helix ---
helix_S_metal = CSX.AddMaterial('helix_S', kappa=5.8e7)
num_pts_S = round(helix_turns_s_band * 20) + 1
# Start at π so the S-band feed sits on the opposite side from the L-band feed
ang_S     = np.linspace(np.pi, helix_turns_s_band * 2 * np.pi + np.pi, num_pts_S)
Helix_S_x = helix_radius_s_band * np.cos(ang_S)
Helix_S_y = helix_radius_s_band * np.sin(ang_S)
Helix_S_z = (ang_S - np.pi) / (2 * np.pi) * helix_pitch_s_band + feed_heigth
helix_S_metal.AddWire(np.array([Helix_S_x, Helix_S_y, Helix_S_z]),
                      wire_diameter_s_band / 2)

# --- Ground reflector (aluminium) ---
gnd = CSX.AddMaterial('gnd', kappa=3.77e7)
if gnd_shape == 'square':
    gnd.AddBox([-gnd_half_size, -gnd_half_size, -gnd_thickness],
               [ gnd_half_size,  gnd_half_size,  0])
elif gnd_shape == 'circle':
    gnd.AddCylinder([0, 0, -gnd_thickness], [0, 0, 0], gnd_half_size)
else:
    raise ValueError(f"gnd_shape must be 'square' or 'circle', got '{gnd_shape}'")

# --- Feed port positions for each antenna ---
start_L = [Helix_radius,       0, 0]
stop_L  = [Helix_radius,       0, feed_heigth]
start_S = [-helix_radius_s_band, 0, 0]
stop_S  = [-helix_radius_s_band, 0, feed_heigth]

# Active port: driven with 50 Ohm excitation
# Inactive port: either shorted to GND via a metal stub, or left floating
if selected_antenna == 'L':
    port = FDTD.AddLumpedPort(1, feed_R, start_L, stop_L, 'z', 1.0, priority=5)
    inactive_start, inactive_stop = start_S, stop_S
    inactive_wire_radius = wire_diameter_s_band / 2
else:  # 'S'
    port = FDTD.AddLumpedPort(1, feed_R, start_S, stop_S, 'z', 1.0, priority=5)
    inactive_start, inactive_stop = start_L, stop_L
    inactive_wire_radius = wire_diameter / 2

if unused_antenna == 'short':
    # Straight copper stub: connects helix start point down to ground plane surface
    feed_short = CSX.AddMaterial('feed_short', kappa=5.8e7)
    feed_short.AddWire(
        np.array([[inactive_start[0], inactive_stop[0]],
                  [inactive_start[1], inactive_stop[1]],
                  [inactive_start[2], inactive_stop[2]]]),
        inactive_wire_radius
    )
# 'float': inactive helix wire end is simply open — nothing added

# nf2ff calc
nf2ff = FDTD.CreateNF2FFBox(opt_resolution=[lambda_center/15]*3)

# E-field time dump through XZ plane (y=0) — open Et*.vtr files in ParaView as a sequence
Et = CSX.AddDump('Et', dump_type=0, file_type=1)
Et.AddBox([-SimBox[0]/2, 0, -SimBox[2]/2],
          [ SimBox[0]/2, 0,  helix_L_top + SimBox[2]/2])

### Write geometry and optionally preview in AppCSXCAD
os.makedirs(Sim_Path, exist_ok=True)
CSX_file = os.path.join(Sim_Path, 'helix.xml')
CSX.Write2XML(CSX_file)
from CSXCAD import AppCSXCAD_BIN
os.system(AppCSXCAD_BIN + ' "{}"'.format(CSX_file))

# exit()gi

### Run the simulation (skipped if cached results exist and force_rerun is False)
sim_done = os.path.isfile(os.path.join(Sim_Path, 'port_ut1.h5'))
if force_rerun or not sim_done:
    if force_rerun and sim_done:
        print('force_rerun=True — discarding cached results and re-simulating.')
    FDTD.Run(Sim_Path, cleanup=force_rerun)
else:
    print(f'Skipping simulation — loading cached results from:\n  {Sim_Path}')

### Postprocessing & plotting
if fast_mode:
    # Sweep only the active band — data beyond f_center±fc is unreliable anyway
    freq = np.linspace(f_center - fc, f_center + fc, 401)
else:
    # Full sweep across both bands
    freq = np.linspace(f0 - fc, f1 + fc, 501)
port.CalcPort(Sim_Path, freq)

Zin = port.uf_tot / port.if_tot
s11 = port.uf_ref / port.uf_inc

## Plot the feed point impedance
fig, axis = plt.subplots(num="Zin", tight_layout=True)
axis.plot(freq/1e6, np.real(Zin), 'k-',  linewidth=2, label='$\\Re(Z_{in})$')
axis.plot(freq/1e6, np.imag(Zin), 'r--', linewidth=2, label='$\\Im(Z_{in})$')
axis.axvline(f0/1e6, color='b', linestyle=':', linewidth=1, label=f'L-band {f0/1e9:.1f} GHz')
axis.axvline(f1/1e6, color='g', linestyle=':', linewidth=1, label=f'S-band {f1/1e9:.1f} GHz')
axis.grid()
axis.set_xmargin(0)
axis.set_xlabel('frequency (MHz)')
axis.set_ylabel('Zin (Ohm)')
axis.set_title(f"feed point impedance — driven: {selected_antenna}-band, unused: {unused_antenna}")
axis.legend()


## Plot reflection coefficient S11
fig, axis = plt.subplots(num="S11", tight_layout=True)
axis.plot(freq/1e6, 20*np.log10(abs(s11)), 'k-', linewidth=2)
axis.axvline(f0/1e6, color='b', linestyle=':', linewidth=1, label=f'L-band {f0/1e9:.1f} GHz')
axis.axvline(f1/1e6, color='g', linestyle=':', linewidth=1, label=f'S-band {f1/1e9:.1f} GHz')
axis.grid()
axis.set_xmargin(0)
axis.set_xlabel('frequency (MHz)')
axis.set_ylabel('S11 (dB)')
axis.set_title(f'reflection coefficient $S_{{11}}$ — driven: {selected_antenna}-band, unused: {unused_antenna}')
axis.legend()


### Create the NFFF contour
## * calculate the far field at phi=0 degrees and at phi=90 degrees
theta = np.arange(0., 180., 1.)
phi   = np.arange(-180, 180, 2)
print('calculating the 3D far field...')

nf2ff_res = nf2ff.CalcNF2FF(Sim_Path, f_center, theta, phi, read_cached=True, verbose=True)

Dmax_dB = 10*np.log10(nf2ff_res.Dmax[0])
E_norm  = 20.0*np.log10(nf2ff_res.E_norm[0]/np.max(nf2ff_res.E_norm[0])) + 10*np.log10(nf2ff_res.Dmax[0])

theta_HPBW = theta[np.where(np.squeeze(E_norm[:,phi==0]) < Dmax_dB - 3)[0][0]]

## * Display power and directivity
print(f'driven antenna : {selected_antenna}-band  ({f_center/1e9:.2f} GHz)')
print(f'unused antenna : {unused_antenna}')
print('radiated power: Prad = {:.5g} W'.format(nf2ff_res.Prad[0]))
print('directivity: Dmax = {:.2f} dBi'.format(Dmax_dB))
print('efficiency: nu_rad = {:.1f} %'.format(100*nf2ff_res.Prad[0]/np.interp(f_center, freq, port.P_acc)))
print('theta_HPBW = {:.1f} °'.format(theta_HPBW))

## Export 3D radiation pattern to VTK for ParaView
E_norm_lin = nf2ff_res.E_norm[0]
E_norm_max = np.max(E_norm_lin)
Dmax_lin   = nf2ff_res.Dmax[0]

theta_rad_g = np.deg2rad(theta)
phi_rad_g   = np.deg2rad(phi)
T, P = np.meshgrid(theta_rad_g, phi_rad_g, indexing='ij')

# directivity per direction (linear); radius of the balloon surface
D_lin  = Dmax_lin * (E_norm_lin              / E_norm_max)**2
D_cprh = Dmax_lin * (np.abs(nf2ff_res.E_cprh[0]) / E_norm_max)**2
D_cplh = Dmax_lin * (np.abs(nf2ff_res.E_cplh[0]) / E_norm_max)**2

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
axis.plot(theta, E_CPRH[:,phi==0], 'r--', linewidth=2, label='|E CPRH|')
axis.plot(theta, E_CPLH[:,phi==0], 'g-.', linewidth=2, label='|E CPLH|')
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

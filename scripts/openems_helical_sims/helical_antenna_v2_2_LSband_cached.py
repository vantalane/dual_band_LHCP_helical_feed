#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
 Dual-Band Helical Antenna Simulation — cached / iterative variant
 Concentric L-band (1.7 GHz) and S-band (2.2 GHz) helices on a shared reflector.

 Each antenna/unused combination gets its own sim directory so switching
 selected_antenna or unused_antenna never clobbers previous results.
 Set force_rerun=True to discard cached data and re-simulate from scratch.

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
force_rerun = False

unit = 1e-3  # all length in mm

# Choose driven antenna: 'L' = 1.7 GHz L-band, 'S' = 2.2 GHz S-band
selected_antenna = 'L'

# What to do with the unused (inactive) antenna:
#   'short' = connect its feed to GND via a copper stub
#   'float' = helix wire end left open / unterminated
#   'none'  = inactive helix not added at all (single-antenna sanity check)
#	'terminate' = inactive helix terminated to reflector plate with a known lumped res

unused_antenna = 'short'
termination_res = 50 # Resistive 

# Simulation duration and difficulty stuff
end_criteria = 1e-3   #  when [1e-3 fields decay to 0.1 % of peak] when 1e-4, thats 0.01%
max_res_div  = 15     # cell size: lambda / 15 (was 20 before)


# change the end_criteria parameter to 1e-3 or lower to get acceptable compute time with L antenna

# L + float : [endcrit 1e-4 > long compute, more than 20-30 mins] 
# L + short : [endcrit 1e-4 > long compute, more than 20 mins]  [endcrit 1e-3 Long more than 20 minutes]
# L + terminate (50ohm) : [endcrit 1e-3 short compute],[endcrit 1e-4 > medium +10 mins ]
# L + none : normal/short compute 

# S + float&short&none :[endcrit 1e-4 > short ]
# S + terminate : [endcrit 1e-4 > ]

# Research papers say that the wires gotta be THIN to prevent excessive coupling and pattern distortion

if unused_antenna not in ('short', 'float', 'none', 'terminate'):
    raise ValueError(f"unused_antenna must be 'short', 'float', 'none', or 'terminate', got '{unused_antenna}'")

# Each config gets its own sim directory
Sim_Path = os.path.join(tempfile.gettempdir(),
                        f'Helical_Anten_doublebnd_{selected_antenna}_{unused_antenna}')

### Frequency parameters
f0 = 1.7e9  # L-band centre frequency
f1 = 2.2e9  # S-band centre frequency
fc = 0.5e9  # Gaussian pulse bandwidth (20 dB corner)

lambda0 = round(C0 / f0 / unit)  # wavelength in mm at L-band
lambda1 = round(C0 / f1 / unit)  # wavelength in mm at S-band

if selected_antenna == 'L':
    f_center      = f0
    lambda_center = lambda0
elif selected_antenna == 'S':
    f_center      = f1
    lambda_center = lambda1
else:
    raise ValueError(f"selected_antenna must be 'L' or 'S', got '{selected_antenna}'")

### Antenna parameters

# --- L-band helix ---
Helix_radius = 28    # mm  (~lambda0 / pi / 2)
Helix_turns  = 5.5
Helix_pitch  = 36    # mm
wire_diameter = 2.7  # mm

# --- S-band helix (concentric, smaller; feed rotated 180° from L-band) ---
helix_radius_s_band  = 21.68  # mm
helix_turns_s_band   = 4.242
helix_pitch_s_band   = 29.18  # mm
wire_diameter_s_band = 2.7    # mm

# Helix top z-coordinates
helix_L_top = Helix_turns        * Helix_pitch       + 4  # feed_heigth = 4
helix_S_top = helix_turns_s_band * helix_pitch_s_band + 4

### Simulation domain
feed_heigth = 4   # feed stub height: ground surface (z=0) to helix start, mm
feed_R      = 50  # feed impedance, Ohm

gnd_shape     = 'square'
gnd_half_size = 65   # half-side (square) or radius (circle), mm
gnd_thickness = 1.5  # mm

# Mesh resolution driven by wire diameter
Helix_mesh_res = min(wire_diameter, wire_diameter_s_band) / 2

# Simulation box focused on active antenna wavelength
SimBox       = np.array([1, 1, 1.2]) * 1.5 * lambda_center

### Setup FDTD
FDTD = openEMS(EndCriteria=end_criteria)
FDTD.SetGaussExcite(f_center, fc)
FDTD.SetBoundaryCond(['MUR', 'MUR', 'MUR', 'MUR', 'MUR', 'PML_8'])

### Mesh
CSX = CSXCAD.ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)

max_res = np.floor(C0 / (f_center + fc) / unit / max_res_div)

# x: both helix radii as anchors
mesh.AddLine('x', [-Helix_radius, -helix_radius_s_band, 0,
                    helix_radius_s_band,  Helix_radius])
mesh.SmoothMeshLines('x', Helix_mesh_res)
mesh.AddLine('x', [-gnd_half_size, gnd_half_size, -SimBox[0]/2, SimBox[0]/2])
mesh.SmoothMeshLines('x', max_res, ratio=1.4)

mesh.SetLines('y', mesh.GetLines('x'))

# z: ground, feed stub, both helix tops
mesh.AddLine('z', [-gnd_thickness, 0, feed_heigth, helix_S_top, helix_L_top])
mesh.SmoothMeshLines('z', Helix_mesh_res)
mesh.AddLine('z', [-SimBox[2]/2, max(mesh.GetLines('z')) + SimBox[2]/2])
mesh.SmoothMeshLines('z', max_res, ratio=1.4)

### Geometry

# L-band helix — always added unless it is the inactive antenna and unused_antenna='none'
if not (selected_antenna == 'S' and unused_antenna == 'none'):
    helix_L_metal = CSX.AddMaterial('helix_L', kappa=5.8e7)
    num_pts_L = round(Helix_turns * 20) + 1
    ang_L     = np.linspace(0, Helix_turns * 2 * np.pi, num_pts_L)
    helix_L_metal.AddWire(
        np.array([ Helix_radius * np.cos(ang_L),
                  -Helix_radius * np.sin(ang_L),
                  ang_L / (2 * np.pi) * Helix_pitch + feed_heigth]),
        wire_diameter / 2)

# S-band helix — always added unless it is the inactive antenna and unused_antenna='none'
# Feed starts at π (opposite side from L-band feed)
if not (selected_antenna == 'L' and unused_antenna == 'none'):
    helix_S_metal = CSX.AddMaterial('helix_S', kappa=5.8e7)
    num_pts_S = round(helix_turns_s_band * 20) + 1
    ang_S     = np.linspace(np.pi, helix_turns_s_band * 2 * np.pi + np.pi, num_pts_S)
    helix_S_metal.AddWire(
        np.array([ helix_radius_s_band * np.cos(ang_S),
                  -helix_radius_s_band * np.sin(ang_S),
                  (ang_S - np.pi) / (2 * np.pi) * helix_pitch_s_band + feed_heigth]),
        wire_diameter_s_band / 2)

# Ground reflector (aluminium)
gnd = CSX.AddMaterial('gnd', kappa=3.77e7)
if gnd_shape == 'square':
    gnd.AddBox([-gnd_half_size, -gnd_half_size, -gnd_thickness],
               [ gnd_half_size,  gnd_half_size,  0])
elif gnd_shape == 'circle':
    gnd.AddCylinder([0, 0, -gnd_thickness], [0, 0, 0], gnd_half_size)
else:
    raise ValueError(f"gnd_shape must be 'square' or 'circle', got '{gnd_shape}'")

### Ports
start_L = [ Helix_radius,        0, 0]
stop_L  = [ Helix_radius,        0, feed_heigth]
start_S = [-helix_radius_s_band, 0, 0]
stop_S  = [-helix_radius_s_band, 0, feed_heigth]

if selected_antenna == 'L':
    port = FDTD.AddLumpedPort(1, feed_R, start_L, stop_L, 'z', 1.0, priority=5)
    inactive_start, inactive_stop = start_S, stop_S
    inactive_wire_radius = wire_diameter_s_band / 2
else:  # 'S'
    port = FDTD.AddLumpedPort(1, feed_R, start_S, stop_S, 'z', 1.0, priority=5)
    inactive_start, inactive_stop = start_L, stop_L
    inactive_wire_radius = wire_diameter / 2

if unused_antenna == 'short':
    # Copper stub connecting inactive helix start down to ground plane
    feed_short = CSX.AddMaterial('feed_short', kappa=5.8e7)
    feed_short.AddWire(
        np.array([[inactive_start[0], inactive_stop[0]],
                  [inactive_start[1], inactive_stop[1]],
                  [inactive_start[2], inactive_stop[2]]]),
        inactive_wire_radius)
elif unused_antenna == 'terminate':
    # Passive lumped port (excite=0): acts as a resistive load at the feed point
    FDTD.AddLumpedPort(2, termination_res, inactive_start, inactive_stop, 'z', 0.0, priority=5)
    
    
# 'float': inactive wire end left open — nothing added
# 'none':  inactive helix not present at all — nothing added

### NF2FF box and E-field dump
nf2ff = FDTD.CreateNF2FFBox(opt_resolution=[lambda_center/15]*3)

Et = CSX.AddDump('Et', dump_type=0, file_type=1)
Et.AddBox([-SimBox[0]/2, 0, -SimBox[2]/2],
          [ SimBox[0]/2, 0,  helix_L_top + SimBox[2]/2])

### Write geometry / preview
os.makedirs(Sim_Path, exist_ok=True)
CSX_file = os.path.join(Sim_Path, 'helix.xml')
CSX.Write2XML(CSX_file)
from CSXCAD import AppCSXCAD_BIN
os.system(AppCSXCAD_BIN + ' "{}"'.format(CSX_file))

# for debugging and drawing helix for sim
# print('exiting because debug')
# exit()

### Run (skipped when cached results exist and force_rerun is False)
sim_done = os.path.isfile(os.path.join(Sim_Path, 'port_ut1.h5'))
if force_rerun or not sim_done:
    if force_rerun and sim_done:
        print('force_rerun=True — discarding cached results and re-simulating.')
    FDTD.Run(Sim_Path, cleanup=force_rerun)
else:
    print(f'Skipping simulation — loading cached results from:\n  {Sim_Path}')

### Post-processing
freq = np.linspace(f_center - fc, f_center + fc, 401)
port.CalcPort(Sim_Path, freq)

Zin = port.uf_tot / port.if_tot
s11 = port.uf_ref / port.uf_inc

## Feed point impedance
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

## Reflection coefficient S11
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

### Far-field
theta = np.arange(0., 180., 1.)
phi   = np.arange(-180, 180, 2)
print('calculating the 3D far field...')

nf2ff_res = nf2ff.CalcNF2FF(Sim_Path, f_center, theta, phi, read_cached=True, verbose=True)

Dmax_dB = 10*np.log10(nf2ff_res.Dmax[0])
E_norm  = 20.0*np.log10(nf2ff_res.E_norm[0]/np.max(nf2ff_res.E_norm[0])) + 10*np.log10(nf2ff_res.Dmax[0])

theta_HPBW = theta[np.where(np.squeeze(E_norm[:,phi==0]) < Dmax_dB - 3)[0][0]]

print(f'driven antenna : {selected_antenna}-band  ({f_center/1e9:.2f} GHz)')
print(f'unused antenna : {unused_antenna}')
print('radiated power: Prad = {:.5g} W'.format(nf2ff_res.Prad[0]))
print('directivity: Dmax = {:.2f} dBi'.format(Dmax_dB))
print('efficiency: nu_rad = {:.1f} %'.format(100*nf2ff_res.Prad[0]/np.interp(f_center, freq, port.P_acc)))
print('theta_HPBW = {:.1f} °'.format(theta_HPBW))

## 3D radiation pattern → VTK
E_norm_lin = nf2ff_res.E_norm[0]
E_norm_max = np.max(E_norm_lin)
Dmax_lin   = nf2ff_res.Dmax[0]

T, P = np.meshgrid(np.deg2rad(theta), np.deg2rad(phi), indexing='ij')
D_lin  = Dmax_lin * (E_norm_lin                   / E_norm_max)**2
D_cprh = Dmax_lin * (np.abs(nf2ff_res.E_cprh[0]) / E_norm_max)**2
D_cplh = Dmax_lin * (np.abs(nf2ff_res.E_cplh[0]) / E_norm_max)**2

R = D_lin / Dmax_lin
X = R * np.sin(T) * np.cos(P)
Y = R * np.sin(T) * np.sin(P)
Z = R * np.cos(T)

ntheta_v, nphi_v = len(theta), len(phi)
vtk_path = os.path.join(Sim_Path, 'pattern_3d.vtk')
with open(vtk_path, 'w') as fv:
    fv.write("# vtk DataFile Version 2.0\n")
    fv.write(f"3D Radiation Pattern {nf2ff_res.freq[0]/1e9:.3f} GHz\n")
    fv.write("ASCII\nDATASET STRUCTURED_GRID\n")
    fv.write(f"DIMENSIONS {ntheta_v} {nphi_v} 1\n")
    fv.write(f"POINTS {ntheta_v * nphi_v} float\n")
    for j in range(nphi_v):
        for i in range(ntheta_v):
            fv.write(f"{X[i,j]:.6f} {Y[i,j]:.6f} {Z[i,j]:.6f}\n")
    fv.write(f"\nPOINT_DATA {ntheta_v * nphi_v}\n")
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

## Cartesian pattern (phi=0 cut)
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

## Polar pattern (phi=0 cut, mirrored)
fig, axis = plt.subplots(num="Pattern_Polar", subplot_kw={'projection': 'polar'}, tight_layout=True)
axis.set_theta_zero_location('N')
axis.set_theta_direction(-1)

theta_rad    = np.deg2rad(theta)
e_total      = np.squeeze(E_norm[:, phi==0])
e_cprh       = np.squeeze(E_CPRH[:, phi==0])
e_cplh       = np.squeeze(E_CPLH[:, phi==0])
angles_full  = np.concatenate([theta_rad, 2*np.pi - theta_rad[::-1]])
e_total_full = np.concatenate([e_total, e_total[::-1]])
e_cprh_full  = np.concatenate([e_cprh,  e_cprh[::-1]])
e_cplh_full  = np.concatenate([e_cplh,  e_cplh[::-1]])

min_level = Dmax_dB - 40
axis.plot(angles_full, np.maximum(e_total_full, min_level), 'k-',  linewidth=2, label='|E total|')
axis.plot(angles_full, np.maximum(e_cprh_full,  min_level), 'r--', linewidth=2, label='|E CPRH|')
axis.plot(angles_full, np.maximum(e_cplh_full,  min_level), 'g-.', linewidth=2, label='|E CPLH|')
axis.set_rlim([min_level, Dmax_dB + 2])
axis.set_title('Radiation Pattern (phi=0° cut) — {:.2f} GHz'.format(nf2ff_res.freq[0]/1e9), pad=15)
axis.legend(loc='lower right')

plt.show()

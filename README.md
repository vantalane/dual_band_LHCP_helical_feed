
### Notice
Designs are still work in progress. Although the OpenEMS simulation scripts work, the strong coupling between the antenna elements on the dual helix antenna causes the radiated energy to linger in the system for a long time, causing very long simulation time and heavily distorted radiation pattern.

# dual_band_LHCP_helical_feed

This repository contains design and simulation files of LHCP helical feed antennas for the reception of weather/scientific satellites using the L and S bands. Currently a L-band helical feed was built and tested, but the dual-coil one is still being worked on.

The L-band requirement mainly focuses on reception of AHRPT streams of weather satellites at 1700MHz.<br>
The S-band requirement is for experimenting with a variety scientific sats such as Aqua (NORAD 27424) and DSCOVR (NORAD 40390).


## Available sources in repo

- FreeCAD design of helical LHCP dish feeds and mounting mechanism for (my) Satenne R3 motorized dish.
- OpenEMS simulation scripts of different helical feed antenna configurations.
- Links to refence blogs, websites and online calculators
- Experiment findings, matlab plots for verifications, calculations and general notes.
- (W.I.P.) Interdigital RF filter and impedance matching circuit designs. (Ant. -> 50R)


## Getting started

In case you want to experiment yourself, install OpenEMS and CSXCAD on your system together with the OpenEMS python interface. If you prefer MATLAB or Octave, you'll need to rewrite the python scripts.

OpenEMS installation can be found at:<br> [www.docs.openems.de](https://docs.openems.de/).


Then you can just run the python scripts in the same manner as the tutorial scripts of OpenEMS.


## Contact
To ask/chat about this:<br>
mete (at) kestech.net 


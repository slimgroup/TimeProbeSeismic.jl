[![DOI](https://zenodo.org/badge/346371878.svg)](https://zenodo.org/badge/latestdoi/346371878)

# TimeProbeSeismic

Wave-equation based inversion with random trace estimation based gradient computation. THis method drastically reduces the memory imprint of adjont-state while managing the loss of accuracy via carefully chosing the probing vector in the range of the wavefield.

## Examples

This repository contains a set of example scripts illustrating time probing for seismic inversion and reproduces the results presented in our SEG abstract.

- [layers_probing.jl](https://github.com/slimgroup/TimeProbeSeismic/blob/master/scripts/layers_probing.jl) is a simple example that computes the gradient for a two layer model for varying number of probing vectors and comapres it against the true gradient.
- [overthrust_probing.jl](https://github.com/slimgroup/TimeProbeSeismic/blob/master/scripts/overthrust_probing.jl) is a similar example computing single source gradients for a more realistic model, the overthrust model.
- [seam_probing.jl](https://github.com/slimgroup/TimeProbeSeismic/blob/master/scripts/seam_probing.jl) is a similar example on the 2D seam model for an OBN setup and higher frequency to highlight or method in imaging settings. Due to the very long recording time, this example requires more probing vectors and a lot of memory to compute the reference true gradient.
- [fwi_2D_overthrust.jl](https://github.com/slimgroup/TimeProbeSeismic/blob/master/scripts/fwi_2D_overthrust.jl) contains an FWI example on the overthrust model comparing FWI with the true gradient to FWI with 32 probing vectors. You can change the number of probing vectors with the variable `ps` in the script.
- [fwi_2D_overthrust_all.jl](https://github.com/slimgroup/TimeProbeSeismic/blob/master/scripts/fwi_2D_overthrust_all.jl) is the main iversion script for the SEG abstract result. This script runs standard FWI, probed fwi with `2,4,8,16,32,64,128,256` probing vector and FWI with on the fly Fourier with `2,4,8,16,32,64,128,256` fourier mode. THis is a total of 17 FWI runs and may take a long time to run.
- [splsrtm_seam.jl](https://github.com/slimgroup/TimeProbeSeismic/blob/master/scripts/splsrtm_seam.jl) this script is untested and in developement. It is intended to run sparsity promoting least square migration on the 2D long-offset sparse OBN seam model with probing for memory effiientcy.


# Author

This software is develloped as Georgia Institute of Technology as part of the ML4Seismic consortium. For questions or issues, please open an issue on github or contact the author:

- Mathias Louboutin: mlouboutin3@gatech.edu

# Cite us

Please use the following citation if you use our software:

```
@inbook{doi:10.1190/segam2021-3584072.1,
author = {Mathias Louboutin and Felix J. Herrmann},
title = {Ultra-low memory seismic inversion with randomized trace estimation},
booktitle = {First International Meeting for Applied Geoscience \&amp; Energy Expanded Abstracts},
chapter = {},
pages = {787-791},
year = {2021},
doi = {10.1190/segam2021-3584072.1},
URL = {https://library.seg.org/doi/abs/10.1190/segam2021-3584072.1},
eprint = {https://library.seg.org/doi/pdf/10.1190/segam2021-3584072.1},
}

@conference {louboutin2022eageewi,
	title = {Enabling wave-based inversion on GPUs with randomized trace estimation},
	booktitle = {EAGE Annual Conference Proceedings},
	year = {2022},
	note = {(EAGE, madrid)},
	month = {03},
	pages = {Seismic Wave Modelling \& Least Square Migration 2 session},
	keywords = {Image Volumes, inversion, RTM, SEAM, stochastic, TTI},
	url = {https://slim.gatech.edu/Publications/Public/Conferences/EAGE/2022/louboutin2022eageewi/louboutinp.html},
	author = {Mathias Louboutin and Felix J. Herrmann}
}
```

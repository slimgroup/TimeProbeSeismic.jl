[![DOI](https://zenodo.org/badge/346371878.svg)](https://zenodo.org/badge/latestdoi/346371878)

# TimeProbeSeismic

## DrWatson

This code base is using the Julia Language and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> TimeProbeSeismic

It is authored by Mathias Louboutin, Felix J. Herrmann.

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box.

## Package

This package can also be installed directly as a standard julia package with the `dev/add` command.


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

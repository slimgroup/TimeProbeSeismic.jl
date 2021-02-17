---
title: Extreme low-memory seismic inversion via trace estimation
author: |
  Mathias Louboutin^1^, Felix J. Herrmann^1^\
  ^1^School of Computational Science and Engineering, Georgia Institute of Technology\
bibliography:
    - bibliography.bib
---

## Abstract


## Introduction


## Theory

In this section, we describe how seismic imaging conditions can be reformulated as a trace estimation problem. In general, wave-equation based seismic imaging condtions can be formaultated as a zero lag crosscorrelation:

```math {#iccc}
\mathcal{I} = \sum_t \rho(\mathbf{u}(t)) \eta(\mathbf{v}(t))
```

where ``\mathbf{u}, \mathbf{v}`` are the forward and adjoint wavefields solutions of the forward and adjoitn wave equations, and ``\rho, \eta`` are differential operators implementing the imaging conditions. We provide a few examples of thes differential operators in Table #ictable for illustration.

### Table: : {#ictable}
|        |  ``\rho(\mathbf{u}(t))`` | ``\eta(\mathbf{v}(t))``|
|:------:|:-------------------------:|:----------------------:|
|Isotropic acoustic | ``\frac{d^2 \mathbf{u}(t)}{dt^2}`` | ``\mathbf{v}(t)`` |
| Inverse scattering | ``\mathbf{m} \frac{d^2 \mathbf{u}(t)}{dt^2} + \Delta \mathbf{u}(t)``| ``\mathbf{v}(t)`` |
| TTI | ``\frac{d^2 (\mathbf{u}_p(t) + \mathbf{u}_s(t))}{dt^2}`` | ``\mathbf{v}_p(t) + \mathbf{v}_s(t)`` |
: Imaging conditions


Following standard linear algebra, this zero-lag correlation over time can be rewritten as the trace of an outer product for each point ``\mathbf{x}`` in space as follows.

```math {#optr}
\mathcal{I}(\mathbf{x}) = \text{tr}(\rho(\mathbf{u}(\mathbf{x})) \eta(\mathbf{v}(\mathbf{x})^\top)).
```

This outerproduct is in practice not possible to compute due to memory (a 1Gb forward wavefield ``\mathbf{u}`` would require around three ExaBytes of memory for the outer product). To tackle this memory issue, we define the estimate of this trace via matrix probing:

```math {#trpr}
    \tilde{\mathcal{I}}(\mathbf{x}) = \frac{1}{N} \sum_{i=1}^{N} \mathbf{z}_i^\top \rho(\mathbf{u}(\mathbf{x})) \eta(\mathbf{v}(\mathbf{x})^\top) \mathbf{z}_i \\
    \mathbf{z}_i \text{ s.t } \mathbb{E}(\mathbf{z}_i^\top \mathbf{z}_i) = 1, \mathbb{E}(\mathbf{z}_i) = 0.
```

Or in its matrix form

```math {#trprM}
    \tilde{\mathcal{I}}(\mathbf{x}) = \frac{1}{N} \text{tr}(\mathbf{Z}^\top \rho(\mathbf{u}(\mathbf{x})) \eta(\mathbf{v}(\mathbf{x})^\top) \mathbf{Z}) \\
```
where ``\mathbf{Z}`` is the matrix with each ``\mathbf{z}_i`` in its column.


This probing, unlike computing the trace, does not require to compute the outer product but only matrix vector products that can be made memory efficient order ing the operations properly. In the above Equation #trpr, that order would be:

- 1. ``v_z = \eta(\mathbf{v}(\mathbf{x})^\top) \mathbf{Z} ``
- 2. ``u_z = \mathbf{Z}^\top \rho(\mathbf{u}(\mathbf{x}))``
- 3. ``\text{tr}_i = \text{tr}(u_z v_z)``.

Through these three steps we can see that we now obtain an unbiased [refs] estimate of the true image ``\mathcal{I}`` via matrix vector products on the outer product of the forward and adjoint wavefield. These three memory optimal steps can then be merged with the time-stepper to implement efficient on-the-fly matrix probing. THe on-the-fly probing is summarized in algoritm #pic\.

### Algorithm: {#pic}
| **for t=1:nt**
| 1. ``\mathbf{u}(t+1) = ts(\mathbf{u}(t), \mathbf{u}(t-1), \mathbf{m})``
| 2. ``u_z(\mathbf{x}) += \mathbf{Z}^\top \rho(\mathbf{u}(t, \mathbf{x}))``
| **for t=nt:1**
| 1. ``\mathbf{v}(t-1) = ts(\mathbf{v}(t), \mathbf{v}(t+1), \mathbf{m})``
| 2. ``v_z(\mathbf{x}) += \eta(\mathbf{v}(t, \mathbf{x})^\top) \mathbf{Z}``
| ``\tilde{\mathcal{I}}(\mathbf{x}) = \text{tr}(u_z v_z)``
: Seismic inversion via probed trace estimation

### Memory estimates

From this formulation, we can estimate the memory imprint of our methid compared to conventional seismic inversion and memory efficient methods for the isotropic acoustic case. This memory overview generalize to other wave equation and imaging condition easily as the memory reuqirements only differ by a constant scalar, related to the number of coupled PDEs, for different physics. We consider a three d imensional domain with ``N_x N_y N_z`` grid points and ``n_t`` time steps. Standard adjoint state requires to save the forward wavefield in its entirety to compute the gradient which leads to a memory requirement of ``4 N n_t`` bytes of memory in single precision. This is the maximum memory necessary. Our method on the other hand, for ``p_s`` probing vector (i.e ``\mathbf{Z} \in \mathbb{R}^{n_t x p_s}``), requires ``4 N p_s`` bytes in the forward pass and ``4 N p_s`` in the backward pass for a total of ``8 N p_s`` byte which leads to a memroy reduction factor of ``\frac{n_t}{2}``. This memory reduction is similar to computing the gradient with ``p_s`` fourier modes, i.e on-the-fly Fourier [refs].

## Illustrative example

Simple two layer model and 2d overthrust single source
- Show gradient
- Show probed gradient
- Show approx A (dobs * dobs') and true A (u v')
- Show  qualitiy ratio ``A-QQ'A``
- Show similarity for increasing probing size ``\frac{<ge, g>}{||ge|| ||g||}``
- Show adjoint test


## 2D FWI

Redo 2d overthrust

## 2D LSRTM

2D SEAM with sparse OBN


## Discussion

## Conclusions

## References
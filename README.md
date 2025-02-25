
# VPF: Variational potential flow

Application that can simulate potential flow
using a variational formulation.

For details on the formulation see:


I. Akkerman, J. H. A. Meijer, and M. F. P. ten Eikelder.
"Isogeometric analysis of linear free-surface potential flow."
 Ocean Engineering 201 (2020): 107114.

O. Colomés,  F. Verdugo, and I. Akkerman.
"A monolithic finite element formulation for the
hydroelastic analysis of very large floating structures."
International Journal for Numerical Methods in Engineering 124.3 (2023):714-751.

The parallel application is build on [mfem](https://github.com/mfem/mfem), a
lightweight, general, scalable C++ library for finite element methods.

## Install -- Local

To install VPF on a Ubuntu-based machine make sure the appropriate packages are installed.

Before installing the packages make sure the machine is updated, by running the folliwing commands in a terminal:

```
sudo apt-get update
sudo apt-get upgrade
```

Install the required packages by running the following command in a terminal:

```
sudo apt-get install cmake git build-essential openmpi-common libopenmpi-dev libmetis5 libmetis-dev
```

The VPF code can be downloaded using the following command:

```
git clone git@github.com:IdoAkkerman/vpf.git

```
or
```
git clone https://github.com/IdoAkkerman/vpf.git

```
This will create a directory `vpf` with the code in it. This code can be compiled using the following commands:

```
cd vpf
mkdir build
cd build
cmake ..
make -j 16
```

Here the number 16 is the number of files that will be compiled concurrently.
NB: This number you choose should depend on the number of cores you have available.

Depending on the machine compilation might take a few minutes.
After this hypre and MFEM will be installed in the `build/external` directory, and
the vpf executable can be found in the `bin` directory.


## Test Cases


2D test cases:
1. ?? `cases/poiseuille`
2. ?? `cases/lid-driven-cavity`


3D test cases: tbd
1. ??
2. ??

## Run
The code can be run by running
```
$BUILD/bin/vpf
```

Where `$BUILD` is the directory where `cmake` was run.

To see all the command line options run:
```
$BUILD/bin/vpf -h
```

### Isogeometric


### GMSH

GMSH is an open-source mesher. On Ubuntu-based machine the package can be installed
using the following command:

```
sudo apt-get install gmsh
```

### Visualization using Visit

The results can be visualized using [Visit] https://visit-dav.github.io/visit-website/index.html.
Which is an open-source tool that can natively visualize NURBS based solutions.

# Licenses

**RBVM** is distributed under the terms of the Apache License (Version 2.0).

All new contributions must be made under the Apache-2.0 licenses.

**MFEM** is distributed under the terms of the BSD-3 license. All new contributions 
must be made under this license. See LICENSE and NOTICE for details.

SPDX-License-Identifier: BSD-3-Clause
LLNL Release Number: LLNL-CODE-806117
DOI: 10.11578/dc.20171025.1248

**HYPRE** is distributed under the terms of both the MIT license and the Apache License (Version 2.0). 
Users may choose either license, at their option.

All new contributions must be made under both the MIT and Apache-2.0 licenses.

See LICENSE-MIT, LICENSE-APACHE, COPYRIGHT, and NOTICE for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-778117


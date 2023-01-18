# Post-processing

This code is designed to post process results from PyFR simulations. Ideally for aerofoil simulation with periodic boundaries in span direction but some functions works for all kinds of simulations.

# Functions and features

## Region
Get a region of interesting from simulation mesh. This region is relied on certain boundary, i.e. 'wall' boundary.

## Span average
Do span average for all snapshots and all variables to get statistics. In this version of code, this is done via doing Fourier transform and save first several modes. NOTE: currently only span average function is supported.

## Sampling data
This function can put probes into the solution field and extract solution for all snapshots. This probes can be either exact location that user specified or closest point it can find from the solution field. Default is closest point.

## Spectrum

## sPOD

## Statistics

## Duplicated points
A function to do average boundary points for adjacent elements and reshape to 1-D array.

## Boundary layer process

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 21:57:21 2026

@author: WangX3

Inspects the data structure of adv.nc files

"""

import h5py

file_path = (
    r"C:\dev\Python\LongWaveAnalysis\ADV"
    r"\adv_dvn_201804_F1.nc"
)

variables = [
    "time",
    "East",
    "North",
    "Up",
    "East_despiked",
    "North_despiked",
    "Up_despiked",
]

with h5py.File(file_path, "r") as f:
    for variable_name in variables:
        print("\n" + "=" * 70)
        print(variable_name)

        try:
            dset = f[variable_name]
            print("shape:", dset.shape)
            print("dtype:", dset.dtype)
            print("chunks:", dset.chunks)
            print("compression:", dset.compression)
        except Exception as exc:
            print("Could not inspect dataset:", type(exc).__name__, exc)
            continue

        print("Attributes:")

        try:
            attr_names = list(dset.attrs.keys())
        except Exception as exc:
            print(
                "Could not list attributes:",
                type(exc).__name__,
                exc,
            )
            attr_names = []

        for attr_name in attr_names:
            try:
                print(" ", attr_name, "=", dset.attrs[attr_name])
            except Exception as exc:
                print(
                    " ",
                    attr_name,
                    "FAILED:",
                    type(exc).__name__,
                    exc,
                )

# inspects the second dimension -> two elevations
names = [
    "instrument",
    "lat",
    "lon",
    "Z",
    "Z_vel",
    "Z_pres",
]

with h5py.File(file_path, "r") as f:
    for name in names:
        print("\n" + "=" * 60)
        print(name)

        dset = f[name]
        print("shape:", dset.shape)
        print("dtype:", dset.dtype)

        try:
            print("values:", dset[...])
        except Exception as exc:
            print("Could not read values:", type(exc).__name__, exc)

        try:
            print("attributes:")
            for attr_name in dset.attrs.keys():
                print(" ", attr_name, "=", dset.attrs[attr_name])
        except Exception as exc:
            print("Could not read attributes:", type(exc).__name__, exc)
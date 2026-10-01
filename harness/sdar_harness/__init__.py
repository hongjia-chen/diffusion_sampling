"""Shared block-diffusion decoding harness (SDAR, with Dream as an optional verifier).

Model loading and pipeline sharding (`model`), the block loop (`blockwise`), the
fixed-k / adaptive scheduler (`scheduler`), the cross-model MI-ratio sampler
(`cross`), the VGB walk (`vgb`), answer scoring (`scoring`), timing and I/O.
Lifted unchanged from exp06_vgb/adaptive_re; see ../README.md.

Heavy model dependencies are intentionally not imported here, so dataset
preparation and result inspection remain usable in CPU-only environments.
"""

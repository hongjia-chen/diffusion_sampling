"""Adaptive-delta sampling with a cross-model verifier and remasking.

SDAR-1.7B proposes, Dream-7B verifies and detects corrections (DESIGN.md).
Heavy model dependencies are intentionally not imported here, so dataset
preparation and result inspection remain usable in CPU-only environments.
"""

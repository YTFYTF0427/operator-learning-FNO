"""Tests for neuropaths.utils.device.

TODO:
    * get_device('auto') prefers cuda > mps > cpu.
    * get_device('cuda') raises when cuda is unavailable (no silent cpu).
    * get_device('cpu') always returns cpu.
    * describe_device(cpu) returns 'cpu'.
"""

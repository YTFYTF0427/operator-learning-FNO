"""Smoke tests for the console_scripts entry points.

TODO:
    * `neuropaths-generate --help`, `neuropaths-train --help`,
      `neuropaths-evaluate --help` all print and exit 0.
    * Passing --config pointing to configs/square_32.yaml loads without
      error (NotImplementedError is allowed here; this test only covers
      config plumbing).
    * Importing neuropaths.cli.generate / train / evaluate does NOT
      execute any main() side effects (regression test against the
      legacy bare-main pattern that triggered training on import).
"""

"""Plotting stack for the cross-representation consistency experiment.

The real results do not exist yet, so the whole stack is developed against
`synth.py` and reads only the columns named in `SCHEMA_COLUMNS`. Nothing here may
touch a column the eval does not promise to emit: the point of building against a
synthetic generator is that the day real data lands, the figures are already
reviewed and the only thing that changes is which CSV is passed in.
"""

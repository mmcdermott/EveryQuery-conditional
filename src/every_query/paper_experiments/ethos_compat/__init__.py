"""Ethos-compat: rewrite Ethos-tokenized MEDS shards into singleton-token MEDS shards.

Paper-experiments utility for the EQ-vs-Ethos training-objective comparison (#174).
Ethos splits each ICD10/CM code into 3 events and each ATC code into 3 events at the
same timestamp.  EQ's task grammar treats ``code`` as an atom, so a query for "did
``I10`` occur" against an Ethos-tokenized corpus would need conjunctive reasoning over
``["ICD//CM//I10", "ICD//CM//3-6//00", "ICD//CM//SFX//"]``.  This module reverses the
splits — collapsing each (head, optional mid, optional sfx) triplet at a shared
timestamp back into a single deterministic code — so EQ_train and EQ_evaluate run
unchanged on the recombined output.

The output code vocabulary is *not* the same as raw MEDS; tokenization is a property
of the preprocessing run, and tasks are scoped to whatever vocab their corpus has.

See ``reprocess.py`` for the recombination logic and CLI entry point.
"""

# Exact response with jumps

Working notes on a kicked one-dimensional system: the flow `dx/dt = x^2 - 1`
on `[-1, 1]` (Exercise 9.7.1.1) plus a periodic "fire" `x -> x/K` at times
`T, 2T, ...`.

The notes cover the evolution of the density, three equivalent ways of
computing expectations of observables (direct, Kawasaki, TTCF), and an exact
TTCF formula that includes the jumps.

- `kicked_ttcf.tex`: LaTeX source (compile with `pdflatex kicked_ttcf.tex`,
  twice; it includes `kicked_ttcf.py` as an appendix)
- `kicked_ttcf.pdf`: compiled notes
- `kicked_ttcf.py`: numerical check (needs only numpy):
  `python3 kicked_ttcf.py`

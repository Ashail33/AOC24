"""Causal lattice Z^{1,d-1} for Hawkes-on-graph simulation (Workstream A).

The construction in the handoff labels vertices by (t, x_vec) with causal order
    (t, x) < (t', x')  iff  t < t'  and  |x' - x| <= t' - t.
For a *continuous-time* Hawkes process the temporal coordinate is carried by the
event times themselves, so the object that actually parameterises the coupling
between vertices is the *spatial* lattice Z^{d-1} together with its
"immediate causal neighbour" graph: from a site x an event can excite x itself
and the sites reachable in one light-cone step, i.e. the 2(d-1) nearest spatial
neighbours.

So `spacetime dimension d`  <=>  `spatial lattice Z^{d-1}` with nearest-neighbour
(+ self) coupling.  d in {2,3,4}  ->  spatial dims {1,2,3}.

This module only knows about geometry: enumerating neighbours, mapping between a
vector site and a flat integer index inside a finite box, and computing graph
(chemical) distance from the origin.  All of the dynamics lives in
`simulator.py` / `propagator.py`.
"""

from __future__ import annotations

import numpy as np


class CausalLattice:
    """Finite box of the spatial lattice Z^{d-1} with self+nearest-neighbour
    causal coupling.

    Parameters
    ----------
    d : int
        Space-time dimension (d in {2,3,4}).  Spatial dimension is d-1.
    half_width : int
        The box is the cube [-half_width, half_width]^{d-1}.  Must be large
        enough that a cascade started at the origin does not reach the boundary
        within the measurement window (the caller is responsible for checking
        this; `boundary_hits` on the simulator output reports leakage).
    """

    def __init__(self, d: int, half_width: int):
        if d < 2:
            raise ValueError("d must be >= 2 (at least one spatial dimension)")
        if half_width < 1:
            raise ValueError("half_width must be >= 1")
        self.d = int(d)
        self.space_dim = int(d - 1)
        self.half_width = int(half_width)
        self.side = 2 * half_width + 1
        self.n_sites = self.side**self.space_dim

        # self + 2*space_dim nearest neighbours
        self.coordination = 2 * self.space_dim + 1
        # branching weight per coupling so that sum over targets == 1
        self.weight = 1.0 / self.coordination

        # Precompute neighbour offsets (including the zero/self offset first).
        offsets = [np.zeros(self.space_dim, dtype=np.int64)]
        for axis in range(self.space_dim):
            for sgn in (+1, -1):
                off = np.zeros(self.space_dim, dtype=np.int64)
                off[axis] = sgn
                offsets.append(off)
        self.offsets = np.array(offsets, dtype=np.int64)  # shape (coordination, space_dim)

        # Cached lookup tables (lazily built, vectorised).
        self._dist_table: np.ndarray | None = None
        self._neighbour_table: tuple[np.ndarray, np.ndarray] | None = None

    # ---- index <-> coordinate helpers -------------------------------------
    def coord_to_index(self, coord: np.ndarray) -> int:
        """Map an integer coordinate vector (relative to box centre) to a flat
        index in [0, n_sites).  Returns -1 if the coordinate is outside the box.
        """
        coord = np.asarray(coord, dtype=np.int64)
        if np.any(np.abs(coord) > self.half_width):
            return -1
        shifted = coord + self.half_width  # now in [0, side)
        idx = 0
        for c in shifted:
            idx = idx * self.side + int(c)
        return idx

    def index_to_coord(self, index: int) -> np.ndarray:
        coord = np.empty(self.space_dim, dtype=np.int64)
        for axis in range(self.space_dim - 1, -1, -1):
            coord[axis] = index % self.side
            index //= self.side
        return coord - self.half_width

    @property
    def origin_index(self) -> int:
        return self.coord_to_index(np.zeros(self.space_dim, dtype=np.int64))

    # ---- neighbour / distance ---------------------------------------------
    def neighbours(self, index: int) -> list[int]:
        """Flat indices of {self} + nearest neighbours that lie inside the box."""
        coord = self.index_to_coord(index)
        out = []
        for off in self.offsets:
            j = self.coord_to_index(coord + off)
            if j >= 0:
                out.append(j)
        return out

    def graph_distance_from_origin(self, index: int) -> int:
        """L1 (chemical) graph distance from the origin -- the minimum number of
        nearest-neighbour hops, which on Z^{d-1} equals the L1 norm."""
        return int(np.sum(np.abs(self.index_to_coord(index))))

    def all_coords(self) -> np.ndarray:
        """(n_sites, space_dim) array of integer coordinates (relative to centre)
        for every flat index, vectorised."""
        idx = np.arange(self.n_sites, dtype=np.int64)
        coords = np.empty((self.n_sites, self.space_dim), dtype=np.int64)
        for axis in range(self.space_dim - 1, -1, -1):
            coords[:, axis] = idx % self.side
            idx //= self.side
        return coords - self.half_width

    def all_distances(self) -> np.ndarray:
        """Cached array of length n_sites giving L1 distance-from-origin of every
        site (vectorised)."""
        if self._dist_table is None:
            self._dist_table = np.sum(np.abs(self.all_coords()), axis=1).astype(np.int64)
        return self._dist_table

    def build_neighbour_table(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (targets, counts): a dense (n_sites, coordination) int array of
        neighbour indices and a (n_sites,) count array, with the in-box neighbours
        packed contiguously at the front of each row.  Cached and vectorised.

        Convention: column 0 is always the self-index; the remaining valid columns
        (up to counts[i]) are the in-box nearest neighbours.  Out-of-box neighbours
        are dropped, so counts[i] < coordination flags a boundary site.
        """
        if self._neighbour_table is not None:
            return self._neighbour_table

        coords = self.all_coords()                    # (n_sites, space_dim)
        n = self.n_sites
        targets = np.full((n, self.coordination), -1, dtype=np.int64)
        counts = np.zeros(n, dtype=np.int64)
        # column index per row where the next valid neighbour goes
        col = np.zeros(n, dtype=np.int64)
        flat_index = np.arange(n, dtype=np.int64)
        for off in self.offsets:
            nbr = coords + off                        # (n_sites, space_dim)
            inside = np.all(np.abs(nbr) <= self.half_width, axis=1)
            shifted = nbr + self.half_width
            j = np.zeros(n, dtype=np.int64)
            for c in range(self.space_dim):
                j = j * self.side + shifted[:, c]
            rows = flat_index[inside]
            cols = col[inside]
            targets[rows, cols] = j[inside]
            col[inside] += 1
            counts[inside] += 1
        self._neighbour_table = (targets, counts)
        return self._neighbour_table

    def __repr__(self) -> str:
        return (f"CausalLattice(d={self.d}, space_dim={self.space_dim}, "
                f"half_width={self.half_width}, n_sites={self.n_sites})")

import numpy as np
from dscribe.descriptors import SOAP
import umap


# ==========================================================
# SOAP + UMAP
# ==========================================================
def compute_soap_umap(
    frames,
    r_cut=5.0,
    n_max=8,
    l_max=6,
    periodic=True,
    n_neighbors=10,
    min_dist=0.1,
    random_state=42,
):
    """
    Compute SOAP descriptors for each frame and project to 2D using UMAP.

    Args:
        frames: list of ASE Atoms objects
        r_cut: SOAP cutoff radius
        n_max: radial basis size
        l_max: angular basis size
        periodic: whether structures are periodic
        n_neighbors: UMAP parameter
        min_dist: UMAP parameter
        random_state: UMAP random seed

    Returns:
        xs, ys: 2D coordinates (arrays of shape [n_frames])
        X: SOAP vectors (array of shape [n_frames, n_features])
    """

    # Collect all species present in the trajectory
    species = sorted(
        set(sum([atoms.get_chemical_symbols() for atoms in frames], []))
    )

    # Build SOAP descriptor
    soap = SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        periodic=periodic,
        sparse=False,
    )

    # Compute one vector per frame by averaging per-atom descriptors
    soap_vectors = []

    for atoms in frames:
        desc = soap.create(atoms)       # shape: (n_atoms, n_features)
        frame_vec = desc.mean(axis=0)  # average over atoms
        soap_vectors.append(frame_vec)

    X = np.array(soap_vectors)

    # UMAP projection
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )

    embedding = reducer.fit_transform(X)

    xs = embedding[:, 0]
    ys = embedding[:, 1]

    return xs, ys, X


# ==========================================================
# Farthest Point Sampling
# ==========================================================
def farthest_point_sampling(points, n_select, start_index=None):
    """
    Farthest Point Sampling (FPS) over 2D/ND points.

    Args:
        points: array of shape [N, D]
        n_select: number of points to select
        start_index: initial index (default: last point)

    Returns:
        selected: list of selected indices in order
    """

    points = np.asarray(points)
    N = len(points)

    if N == 0:
        return []

    n_select = min(n_select, N)

    if start_index is None:
        start_index = N - 1  # default: start from last point

    selected = [int(start_index)]

    # Initialize min distance to the selected set
    min_dist = np.linalg.norm(points - points[selected[0]], axis=1)

    for _ in range(1, n_select):
        # Pick the point farthest from the selected set
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)

        # Update min distance
        dist_to_new = np.linalg.norm(points - points[next_idx], axis=1)
        min_dist = np.minimum(min_dist, dist_to_new)

    return selected

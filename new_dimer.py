import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Dimer lattice helpers (H/V representation)
# ============================================================

def initialize_columnar_configs(L: int, Nw: int):
    """
    Initialize Nw walkers on an LxL periodic lattice with simple
    columnar dimer coverings (half vertical, half horizontal).

    Geometry / convention:
      - Sites are labeled (x, y) with:
            x = 0..L-1  (left to right)
            y = 0..L-1  (bottom to top)
      - h[x, y] = 1 if there is a horizontal dimer on bond
                  (x, y) -> (x+1, y)     (modulo L)
      - v[x, y] = 1 if there is a vertical dimer on bond
                  (x, y) -> (x, y+1)     (modulo L)
      - We require L to be even so that close-packed dimers exist with PBC.
    """
    if L % 2 != 0:
        raise ValueError("L must be even for close-packed dimers with PBC.")

    h = np.zeros((Nw, L, L), dtype=np.uint8)
    v = np.zeros((Nw, L, L), dtype=np.uint8)

    for iw in range(Nw):
        if iw % 2 == 0:
            # Vertical columnar pattern:
            # For each x, place vertical dimers on bonds (x,y)->(x,y+1) for even y.
            # This gives exactly one dimer per site on an even LxL lattice.
            v[iw, :, 0::2] = 1
        else:
            # Horizontal columnar pattern:
            # For each y, place horizontal dimers on bonds (x,y)->(x+1,y) for even x.
            h[iw, 0::2, :] = 1

    return h, v


def plaquette_state_single(h_single: np.ndarray, v_single: np.ndarray, x: int, y: int) -> int:
    """
    Plaquette with bottom-left corner at (x, y) on a periodic LxL lattice.

    Returns:
        +1  if plaquette is horizontally flippable
        -1  if plaquette is vertically flippable
         0  otherwise

    Edges:
      bottom: h[x, y]
      top:    h[x, y+1]
      left:   v[x, y]
      right:  v[x+1, y]
    """
    L = h_single.shape[0]
    xp = (x + 1) % L
    yp = (y + 1) % L

    h_bottom = h_single[x, y]
    h_top    = h_single[x, yp]
    v_left   = v_single[x, y]
    v_right  = v_single[xp, y]

    # horizontally flippable
    if h_bottom and h_top and not v_left and not v_right:
        return 1

    # vertically flippable
    if v_left and v_right and not h_bottom and not h_top:
        return -1

    return 0


def compute_flippable_map(h_single: np.ndarray, v_single: np.ndarray) -> np.ndarray:
    """
    Compute boolean map of flippable plaquettes for a single walker.

    flippable[x, y] is True if plaquette (x, y) is either
    horizontally or vertically flippable.
    """
    L = h_single.shape[0]
    flippable = np.zeros((L, L), dtype=bool)
    for x in range(L):
        for y in range(L):
            flippable[x, y] = (plaquette_state_single(h_single, v_single, x, y) != 0)
    return flippable


def affected_plaquettes(x: int, y: int, L: int):
    """
    Plaquette (x,y) only affects flippability of itself and its four neighbors.
    Return the list of (qx,qy) indices to update.
    """
    return [
        (x, y),
        ((x + 1) % L, y),
        ((x - 1) % L, y),
        (x, (y + 1) % L),
        (x, (y - 1) % L),
    ]


def flip_plaquette_and_update(
    h_single: np.ndarray,
    v_single: np.ndarray,
    flippable_single: np.ndarray,
    x: int,
    y: int,
) -> int:
    """
    Perform a Rokhsar-Kivelson plaquette flip at plaquette (x, y)
    for a single walker, if flippable. Update flippable_single locally.

    Returns:
        dNf : change in the total number of flippable plaquettes Nf
    """
    L = h_single.shape[0]
    xp = (x + 1) % L
    yp = (y + 1) % L

    h_bottom = h_single[x, y]
    h_top    = h_single[x, yp]
    v_left   = v_single[x, y]
    v_right  = v_single[xp, y]

    # Decide which way (if any) to flip: horizontal <-> vertical
    flipped = False
    if h_bottom and h_top and not v_left and not v_right:
        # horizontal -> vertical
        h_single[x, y]  = 0
        h_single[x, yp] = 0
        v_single[x, y]  = 1
        v_single[xp, y] = 1
        flipped = True
    elif v_left and v_right and not h_bottom and not h_top:
        # vertical -> horizontal
        v_single[x, y]  = 0
        v_single[xp, y] = 0
        h_single[x, y]  = 1
        h_single[x, yp] = 1
        flipped = True

    if not flipped:
        return 0

    # Update flippability locally in a 5-plaquette neighborhood
    dNf = 0
    for qx, qy in affected_plaquettes(x, y, L):
        old_flippable = flippable_single[qx, qy]
        new_flippable = (plaquette_state_single(h_single, v_single, qx, qy) != 0)
        flippable_single[qx, qy] = new_flippable

        if old_flippable and not new_flippable:
            dNf -= 1
        elif (not old_flippable) and new_flippable:
            dNf += 1

    return dNf


def measure_profile_line(h_single: np.ndarray, v_single: np.ndarray) -> np.ndarray:
    """
    Plaquette orientation profile along a vertical line:

      For x_fixed = L//2, define for each y:

        G[y] = +1 if horizontally flippable
               -1 if vertically flippable
                0 otherwise

    Returns:
        G : shape (L,)
    """
    L = h_single.shape[0]
    G = np.zeros(L, dtype=float)

    x_fixed = L // 2
    for y in range(L):
        state = plaquette_state_single(h_single, v_single, x_fixed, y)
        if state == 1:
            G[y] = 1.0
        elif state == -1:
            G[y] = -1.0
        else:
            G[y] = 0.0

    return G


def measure_avg_plaquette_orientation(h_single: np.ndarray, v_single: np.ndarray) -> float:
    """
    Average plaquette orientation over the entire lattice:

      B_p = +1 if horizontally flippable
            -1 if vertically flippable
             0 otherwise

      <B> = (1/L^2) sum_p B_p
    """
    L = h_single.shape[0]
    total = 0.0
    for x in range(L):
        for y in range(L):
            total += plaquette_state_single(h_single, v_single, x, y)
    return total / (L * L)


# Wavevectors at which we evaluate structure factors
Q_PEAKS = [
    (np.pi, 0.0),    # (pi, 0)
    (0.0, np.pi),    # (0, pi)
    (np.pi, np.pi),  # (pi, pi)
]


def measure_structure_factor_peaks(
    h_single: np.ndarray,
    v_single: np.ndarray,
    q_list=Q_PEAKS,
):
    """
    Evaluate dimer structure factor peaks for horizontal and vertical dimers
    at a set of momenta q_list.

    For each q = (qx,qy), compute:

       Sx(q) = (1/N) | sum_r n_x(r) e^{i q·r} |^2
       Sy(q) = (1/N) | sum_r n_y(r) e^{i q·r} |^2

    where n_x(r) = h[x,y], n_y(r) = v[x,y].
    Here we treat bond centers as living at integer coordinates (x,y);
    this is fine for commensurate q like (pi,0), (0,pi), (pi,pi).

    Returns:
        Sx_vals, Sy_vals : 1D arrays of shape (len(q_list),)
    """
    L = h_single.shape[0]
    x_coords, y_coords = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")

    n_norm = float(L * L)
    Sx_vals = np.empty(len(q_list), dtype=float)
    Sy_vals = np.empty(len(q_list), dtype=float)

    n_x = h_single.astype(float)
    n_y = v_single.astype(float)

    for idx, (qx, qy) in enumerate(q_list):
        phase = np.exp(1j * (qx * x_coords + qy * y_coords))
        Sx = np.abs((n_x * phase).sum())**2 / n_norm
        Sy = np.abs((n_y * phase).sum())**2 / n_norm
        Sx_vals[idx] = Sx.real
        Sy_vals[idx] = Sy.real

    return Sx_vals, Sy_vals


# ============================================================
# GFMC core
# ============================================================

def lgfmc(
    k: int = 8,
    Nw: int = 30,
    P: float = 0.5,
    Mct: int = 20_000,
    Mck: int = 100,
    Mcx: int = 10_000,
    Mcf: int = 1_000,
):
    """
    Green's Function Monte Carlo for the square-lattice quantum dimer model
    using an H/V dimer representation.

    Parameters:
        k   : controls lattice size L = 2*k
        Nw  : number of walkers
        P   : (J - V)/J parameter in the QDM
        Mct : number of reconfigurations used for measurement
        Mck : number of time steps between reconfigurations
        Mcx : number of thermalization reconfigurations
        Mcf : interval for progress prints

    Returns:
        A dict with fields needed for forward walking:
            h, v : final configurations (Nw, L, L)
            nf   : Nf history (McT, Nw)
            O    : measured operator history (McT, Nw, op_dim)
            jw   : ancestry indices (McT, Nw)
            wt   : log normalization factors per reconfig (McT,)
            lw   : log weights per walker (McT, Nw)
            L    : lattice size
            Nw   : number of walkers
            Mct, Mcx, McT
    """
    # Lattice size and basic counts
    L = 2 * k               # even linear size
    Nplaquettes = L * L
    McT = Mct + Mcx

    # Initialize walkers
    h, v = initialize_columnar_configs(L, Nw)

    # Flippability maps and Nf for each walker
    flippable = np.zeros((Nw, L, L), dtype=bool)
    Nf = np.zeros(Nw, dtype=int)
    for iw in range(Nw):
        flippable[iw] = compute_flippable_map(h[iw], v[iw])
        Nf[iw] = np.count_nonzero(flippable[iw])

    # GFMC bookkeeping
    lw = np.zeros((McT, Nw), dtype=float)  # log weights per walker per reconfig
    wt = np.zeros(McT, dtype=float)        # log normalization per reconfig
    nf = np.zeros((McT, Nw), dtype=int)    # Nf history
    nf[0, :] = Nf

    # Observable layout:
    #   0 .. L-1                 : G(y) profile along vertical cut
    #   L                        : average plaquette orientation <B>
    #   L+1 .. L+3               : Sx(q) at q = (pi,0),(0,pi),(pi,pi)
    #   L+4 .. L+6               : Sy(q) at same qs
    n_q = len(Q_PEAKS)
    op_dim = L + 1 + 2 * n_q
    O = np.zeros((McT, Nw, op_dim), dtype=float)

    jw = np.zeros((McT, Nw), dtype=int)           # ancestry indices

    # Random seed (like rng('shuffle') in MATLAB)
    np.random.seed()

    print("Starting GFMC run...", flush=True)

    # Main GFMC loop over reconfigurations
    for it in range(1, McT):
        # Work with copies for this reconfiguration
        h_curr = h.copy()
        v_curr = v.copy()
        flipp_curr = flippable.copy()
        Nf_curr = nf[it - 1].copy()
        lW = np.zeros(Nw, dtype=float)

        # Propagate each walker independently
        for iw in range(Nw):
            te = 0  # "time" counter within this reconfiguration

            while te < Mck:
                # Diagonal part: effective waiting time before the next off-diagonal event
                denom = 1.0 + P * Nf_curr[iw] / Nplaquettes
                if Nf_curr[iw] > 0:
                    p_jump = (Nf_curr[iw] / Nplaquettes) / denom
                else:
                    p_jump = 0.0

                if p_jump <= 0.0:
                    # No off-diagonal events can occur; just accumulate diagonal
                    td = Mck - te
                else:
                    # Geometric waiting time
                    u = np.random.rand()
                    td = int(np.floor(np.log(u) / np.log(1.0 - p_jump)))

                if td < 0:
                    td = 0
                if te + td > Mck:
                    td = Mck - te

                te += td
                lW[iw] += td * np.log(denom)

                # Stop if we've exhausted this reconfiguration time window
                if te >= Mck:
                    break

                # If no flippable plaquettes, skip off-diagonal move
                if Nf_curr[iw] <= 0:
                    break

                # Try to flip one plaquette (off-diagonal move)
                max_tries = 1000
                found = False
                dNf_total = 0
                for _ in range(max_tries):
                    x = np.random.randint(0, L)
                    y = np.random.randint(0, L)

                    if not flipp_curr[iw, x, y]:
                        continue

                    dNf = flip_plaquette_and_update(
                        h_curr[iw], v_curr[iw], flipp_curr[iw], x, y
                    )
                    dNf_total += dNf
                    found = True
                    break

                # If we never found a flippable plaquette to flip, end this walker
                if not found:
                    break

                # Off-diagonal step taken
                te += 1
                Nf_curr[iw] += dNf_total
                denom2 = 1.0 + P * Nf_curr[iw] / Nplaquettes
                lW[iw] += np.log(denom2)

        # Measure observables after this reconfiguration
        for iw in range(Nw):
            G_line = measure_profile_line(h_curr[iw], v_curr[iw])
            avg_B = measure_avg_plaquette_orientation(h_curr[iw], v_curr[iw])
            Sx_vals, Sy_vals = measure_structure_factor_peaks(h_curr[iw], v_curr[iw], Q_PEAKS)

            # Pack into O[it, iw, :]
            O[it, iw, 0:L] = G_line
            O[it, iw, L] = avg_B
            O[it, iw, L + 1 : L + 1 + n_q] = Sx_vals
            O[it, iw, L + 1 + n_q : L + 1 + 2 * n_q] = Sy_vals

        # Population control (reconfiguration of walkers)
        lw[it, :] = lW
        lWt = np.mean(lW)
        # Log-normalization (same idea as your original code)
        Wt = lWt + np.log(np.sum(np.exp(lW - lWt)) / Nw)
        wt[it] = Wt

        # Relative weights for resampling
        rel_weights = np.exp(lW - Wt)
        rel_weights_sum = np.sum(rel_weights)
        if rel_weights_sum <= 0.0:
            # Extremely unlikely; fallback to uniform
            probs = np.full(Nw, 1.0 / Nw)
        else:
            probs = rel_weights / rel_weights_sum

        # Resample walkers according to probs
        ancestors = np.random.choice(Nw, size=Nw, p=probs)
        jw[it, :] = ancestors

        # New ensemble for next iteration
        h = h_curr[ancestors]
        v = v_curr[ancestors]
        flippable = flipp_curr[ancestors]
        nf[it, :] = Nf_curr[ancestors]

        # Optional progress print
        if (it + 1) % Mcf == 0:
            print(f"Reconfig {it+1}/{McT}, Wt={Wt:.4f}", flush=True)

    return {
        "h": h,
        "v": v,
        "nf": nf,
        "O": O,
        "jw": jw,
        "wt": wt,
        "lw": lw,
        "L": L,
        "Nw": Nw,
        "Mct": Mct,
        "Mcx": Mcx,
        "McT": McT,
    }


# ============================================================
# Forward walking estimator
# ============================================================

def forward_walking_fwalk(data, L_hist: int = 20, N_proj: int = 20):
    """
    Forward-walking estimator for <O>.

    O can be:
      - (T, Nw)          -> scalar observable
      - (T, Nw, op_dim)  -> vector observable of length op_dim

    Parameters:
        data    : dictionary from lgfmc()
        L_hist  : history length in wt used to compute G1
        N_proj  : projection length in ancestry (number of JW steps)

    Returns:
        G : ground-state estimate of <O>, shape (op_dim,)
    """
    jw = data["jw"]
    wt = data["wt"]
    O = data["O"]
    Mct = data["Mct"]
    Mcx = data["Mcx"]
    McT = data["McT"]
    Nw = data["Nw"]

    T = McT

    if O.ndim == 2:
        op_dim = 1
    else:
        op_dim = O.shape[2]

    # Average weight over measurement window
    awt = np.sum(wt[Mcx:T]) / Mct

    G1_list = []
    G2_list = []

    # it runs from Mcx .. McT - N_proj - 1
    for it in range(Mcx, T - N_proj):
        # Start from all walkers at time it
        j = np.arange(Nw, dtype=int)

        # Follow ancestry N_proj steps forward
        for ij in range(N_proj):
            j = jw[it + ij, j]

        # Weight factor
        t0 = max(0, it - L_hist + 1)
        t1 = min(T - 1, it + N_proj)
        G1 = np.exp(np.sum(wt[t0:t1 + 1]) - awt * (L_hist + N_proj))

        # Observable at time it, for those ancestors
        if O.ndim == 2:
            O_slice = O[it, j]          # (Nw,)
            avg_O = np.mean(O_slice)    # scalar
        else:
            O_slice = O[it, j, :]       # (Nw, op_dim)
            avg_O = np.mean(O_slice, axis=0)  # (op_dim,)

        G1_list.append(G1)
        G2_list.append(G1 * avg_O)

    G1_arr = np.array(G1_list)
    G2_arr = np.array(G2_list)

    G = np.sum(G2_arr, axis=0) / np.sum(G1_arr)
    return np.atleast_1d(G)


# ============================================================
# Main (example usage)
# ============================================================

if __name__ == "__main__":
    data = lgfmc()
    print("GFMC run finished.", flush=True)
    print("O shape:", data["O"].shape, flush=True)

    G = forward_walking_fwalk(data, L_hist=20, N_proj=20)
    L = data["L"]
    n_q = len(Q_PEAKS)

    # Unpack observables from G
    G_profile = G[0:L]
    avg_B = G[L]
    Sx_vals = G[L + 1 : L + 1 + n_q]
    Sy_vals = G[L + 1 + n_q : L + 1 + 2 * n_q]

    print("Ground-state <G(y)> profile shape:", G_profile.shape)
    print("Ground-state <B> (avg plaquette orientation):", avg_B)
    print("Ground-state Sx(q) at q = (pi,0),(0,pi),(pi,pi):", Sx_vals)
    print("Ground-state Sy(q) at q = (pi,0),(0,pi),(pi,pi):", Sy_vals)

    # Plot the profile G(y)
    y_sites = np.arange(L)
    plt.figure()
    plt.plot(y_sites, G_profile, marker="o")
    plt.xlabel("plaquette index along vertical cut (y)")
    plt.ylabel("G(y)")
    plt.title("Forward-walking plaquette orientation profile G(y)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("forward_walking_profile.png", dpi=300)
    plt.close()

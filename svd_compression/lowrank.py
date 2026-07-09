"""Low-rank compression of MoE expert weights.

Two methods:
  - plain_svd:            truncated SVD of the weight; the SVD factors are used as-is and NOT modified.
                          Minimises the Frobenius weight error ||W - U S V||   (data-agnostic).
  - activation_aware_svd: initialise the factors from the plain SVD, then REFINE them by an alternating
                          least-squares algorithm that minimises the OUTPUT error ||X W^T - X U V||
                          over calibration activations X, with per-token gate weighting. Same rank and
                          output shape as plain SVD; the difference is that the factors are data-fitted.
                          Mirrors NNCF's lora_correction.calculate_low_rank_matrices.

Both RECONSTRUCT IN PLACE (return a full-shape weight), so they measure the accuracy ceiling of a
rank-r approximation. To actually save memory you would store the factors separately and change the
forward pass; that is out of scope here.

Shape legend (as in NNCF lora_correction): O - output dim, H - input/hidden dim, SS - samples size,
R - rank. Matrices below follow this notation for the SVD/least-squares math.
"""
import torch


def svd(matrix, full_matrices=False):
    """SVD (matrix = U @ diag(S) @ V) with a CPU fallback: cuSOLVER's gesvdj can raise on
    ill-conditioned bf16 matrices, so retry on CPU if it does."""
    try:
        return torch.linalg.svd(matrix, full_matrices=full_matrices)
    except torch._C._LinAlgError:
        U, S, V = torch.linalg.svd(matrix.cpu(), full_matrices=full_matrices)
        return U.to(matrix.device), S.to(matrix.device), V.to(matrix.device)


@torch.no_grad()
def plain_svd(weight, rank):
    """Truncated-SVD reconstruction of `weight` ([O, H]) to `rank`. Frobenius-optimal, data-agnostic.
    Returns a full-shape weight of the original dtype."""
    weight_fp32 = weight.float()
    U, S, V = svd(weight_fp32)
    rank = min(rank, S.shape[0])
    reconstruction = (U[:, :rank] * S[:rank]) @ V[:rank, :]
    return reconstruction.to(weight.dtype)


@torch.no_grad()
def activation_aware_svd(weight, activations, rank, num_iterations=3, gate_weights=None,
                         mean_baseline=False):
    """Activation-aware rank-`rank` approximation of `weight` ([O, H]).

    Fits low-rank factors U ([H, R]) and V ([R, O]) to minimise ||diag(sqrt(gate)) (X W^T - X U V)||
    over calibration activations `activations` (X, [SS, H]), initialised from the plain SVD and refined
    by alternating least squares (fix U -> solve V, fix V -> solve U). `gate_weights` ([SS]) is the
    per-token router-softmax probability for this expert: rows of X are scaled by sqrt(gate) so the fit
    focuses on the tokens the expert actually attends to WITHOUT hard-filtering (which would
    under-determine rarely-routed experts). Falls back to plain SVD if the refinement diverges.

    If `mean_baseline=True`, a fixed rank-1 baseline (1/H * ones([O, H])) is stripped from W before
    fitting. The baseline maps each token to the mean of its input channels; all `rank` directions of
    U,V then model the deviation from that mean rather than re-learning it. Reconstruction adds the
    baseline back: W_approx = baseline + (U @ V).T.

    The SVD-init + alternating-least-squares refinement is adapted from NNCF's lora_correction:
    https://github.com/openvinotoolkit/nncf/blob/develop/nncf/quantization/algorithms/weight_compression/lora_correction.py
    (function `calculate_low_rank_matrices`; simplified to no fake-quant residual / no regularization,
    with sqrt(gate) per-token weighting added).
    """
    weight_fp32 = weight.float()
    num_input_channels = weight_fp32.shape[1]        # H (Hidden dim)
    X = activations.float()                          # [SS, H]
    if gate_weights is not None:
        # sqrt so that ||sqrt(gate) * .||^2 == gate * ||.||^2  (gate-weighted least squares)
        X = X * gate_weights.float().clamp_min(0).sqrt().unsqueeze(1)   # [SS, H]

    # Optional fixed baseline: (1/H)*ones([O,H]).  Strips the mean-of-inputs component from W so
    # U,V fit only the structured residual, then it is added back at reconstruction.
    if mean_baseline:
        baseline = torch.full_like(weight_fp32, 1.0 / num_input_channels)   # [O, H]
        residual = (weight_fp32 - baseline).t().contiguous()                 # [H, O]
    else:
        baseline = None
        residual = weight_fp32.t().contiguous()                              # [H, O]

    # Low-rank approximation (SVD init on the residual, as in NNCF lora_correction).
    U_full, S_full, V_full = svd(residual)
    rank = min(rank, S_full.shape[0])
    U = U_full[:, :rank].contiguous()                # [H, R]
    V = (torch.diag(S_full[:rank]) @ V_full[:rank, :]).contiguous()   # [R, O]

    target_output = X @ residual                     # [SS, O]  (target: X W^T or X (W-baseline)^T)

    def lstsq(a, b):
        # solves a @ x = b in the least-squares sense; gels is the CUDA-supported driver
        return torch.linalg.lstsq(a, b, driver="gels").solution

    def pinv(matrix):
        # robust pseudo-inverse via SVD (cuSOLVER's pinv can fail on ill-conditioned factors)
        U_p, S_p, V_p = svd(matrix)
        tolerance = S_p.max() * max(matrix.shape) * torch.finfo(S_p.dtype).eps
        S_p_inv = torch.where(S_p > tolerance, 1.0 / S_p, torch.zeros_like(S_p))
        return (V_p.t() * S_p_inv) @ U_p.t()

    # Iterative correction of the low-rank factors.
    converged = True
    for _ in range(num_iterations):
        # Part 1: U fixed, find V.   X @ U @ V = target_output
        XU = X @ U                                   # [SS, R]
        V = lstsq(XU, target_output)                 # [R, O]
        # Part 2: V fixed, find U.   X @ U = target_output @ V^-1
        try:
            VI = pinv(V)                             # [O, R]
        except torch._C._LinAlgError:
            converged = False
            break
        U = lstsq(X, target_output @ VI)             # [H, R]

    if not converged or not torch.isfinite(U).all() or not torch.isfinite(V).all():
        # refinement diverged (too few / degenerate routed tokens) -> fall back to Frobenius SVD
        U_full, S_full, V_full = svd(weight_fp32)
        rank = min(rank, S_full.shape[0])
        return ((U_full[:, :rank] * S_full[:rank]) @ V_full[:rank, :]).to(weight.dtype)

    low_rank_weight = (U @ V).t()                    # [O, H], rank-R, activation-optimal
    if baseline is not None:
        low_rank_weight = low_rank_weight + baseline
    return low_rank_weight.to(weight.dtype)

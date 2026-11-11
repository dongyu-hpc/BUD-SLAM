import numpy as np
import torch
import torch.nn.functional as F


def compute_latent_uncertainty(tokens, patch_start_idx, image_height, image_width,
                                patch_size, window=3, eps=1e-8):
    """
    Quantum-inspired uncertainty from latent tokens, using the
    density-matrix / von Neumann entropy formalism from quantum mechanics:

      1. Treat each patch token as a unit-normalized state vector |psi_k>
         (analogous to a pure quantum state in feature space).
      2. For each spatial window x window neighborhood of patches, form the
         ensemble density matrix rho = (1/K) * sum_k |psi_k><psi_k|, which
         is Hermitian, PSD, trace 1 -- a genuine density matrix.
      3. Diagonalize rho (the "basis function decomposition"):
         rho = sum_i lambda_i |e_i><e_i|. The eigenvalues lambda_i are
         Born-rule-style occupation probabilities over the eigenbasis.
      4. Uncertainty = von Neumann entropy S(rho) = -sum_i lambda_i log(lambda_i),
         normalized to [0, 1]. A neighborhood of near-identical tokens
         collapses to a near-pure state (low entropy). A neighborhood of
         divergent tokens spreads across eigenmodes (high entropy).

    Implementation note: rho is C x C (C = token feature dim, e.g. 768+),
    too expensive to diagonalize per patch directly. Since rho is built
    from only window*window outer products, its rank is <= window*window.
    We instead diagonalize the K x K Gram matrix G_ab = <psi_a, psi_b> / K
    (K = window*window), whose nonzero eigenvalues are identical to rho's --
    the same covariance/Gram duality used in PCA when sample count < feature
    dimension. This is exact, not an approximation.

    Args:
        tokens: (S, N, C) latent tokens for one submap at a chosen
            aggregator layer (e.g. aggregated_tokens_list[layer_idx]
            squeezed of its batch dim). N must be patch_start_idx prefix
            tokens followed by H_p*W_p patch tokens in row-major order.
        patch_start_idx: number of non-patch prefix tokens (e.g. register
            tokens), same value used elsewhere as
            model.aggregator.patch_start_idx.
        image_height, image_width: pixel resolution these tokens
            correspond to.
        patch_size: model.aggregator.patch_size.
        window: side length (in patches) of the square neighborhood used
            to build each local density matrix. Must be >= 1 and odd for
            symmetric padding.
        eps: numerical floor to avoid log(0) and divide-by-zero.

    Returns:
        uncertainty: (S, image_height, image_width) float32 torch tensor
            in [0, 1], on the same device as `tokens`.
    """
    if window < 1:
        raise ValueError("window must be >= 1")

    S, N, C = tokens.shape
    h_p = image_height // patch_size
    w_p = image_width // patch_size
    expected_patch_tokens = h_p * w_p
    n_patch_tokens = N - patch_start_idx
    if n_patch_tokens != expected_patch_tokens:
        raise ValueError(
            f"Token count mismatch: got {n_patch_tokens} patch tokens "
            f"(N={N}, patch_start_idx={patch_start_idx}) but expected "
            f"{expected_patch_tokens} = {h_p} x {w_p} from image size "
            f"{image_height}x{image_width} at patch_size={patch_size}. "
            f"This likely means the assumed layout of aggregated_tokens_list "
            f"does not match this model -- check the actual tensor shape."
        )

    patch_tokens = tokens[:, patch_start_idx:, :].reshape(S, h_p, w_p, C)

    # Unit-normalize each token so it behaves like a pure quantum state.
    patch_tokens = F.normalize(patch_tokens, dim=-1, eps=eps)

    # (S, C, h_p, w_p) for unfold-based neighborhood extraction.
    grid = patch_tokens.permute(0, 3, 1, 2)
    pad = window // 2
    grid = F.pad(grid, (pad, pad, pad, pad), mode="replicate")

    # Extract window x window neighborhoods around every patch position.
    neighborhoods = F.unfold(grid, kernel_size=window)  # (S, C*K*K, h_p*w_p)
    K = window * window
    neighborhoods = neighborhoods.view(S, C, K, h_p * w_p)
    neighborhoods = neighborhoods.permute(0, 3, 2, 1)  # (S, P, K, C)

    # Gram matrix per neighborhood: G_ab = <psi_a, psi_b>, shape (S, P, K, K).
    gram = torch.matmul(neighborhoods, neighborhoods.transpose(-1, -2))
    rho_reduced = gram / K  # trace-1, same nonzero eigenvalues as the true
                            # C x C density matrix.

    eigvals = torch.linalg.eigvalsh(rho_reduced)  # (S, P, K), ascending
    eigvals = eigvals.clamp(min=0.0)  # guard tiny negative numerical noise

    log_eigvals = torch.log(eigvals + eps)
    entropy = -(eigvals * log_eigvals).sum(dim=-1)  # (S, P)

    max_entropy = float(np.log(K))
    uncertainty_patches = (entropy / max_entropy).clamp(0.0, 1.0)  # (S, P)

    uncertainty_grid = uncertainty_patches.view(S, 1, h_p, w_p)
    uncertainty_full = F.interpolate(
        uncertainty_grid, size=(image_height, image_width),
        mode="bilinear", align_corners=False,
    )
    return uncertainty_full.squeeze(1)  # (S, image_height, image_width)

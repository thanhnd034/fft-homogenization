import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path


def generate_microstructure_from_files(X, Y, sample_id=8281, data_dir=None):
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent
    else:
        data_dir = Path(data_dir)

    coeff_vec_samples = np.load(data_dir / "coeff_vec.npy")
    phi0_samples = np.load(data_dir / "phi0.npy")
    l_samples = np.load(data_dir / "l_data.npy")
    m_samples = np.load(data_dir / "m_data.npy")

    coeff_vec = jnp.array(coeff_vec_samples[sample_id])
    phi_0 = jnp.array(phi0_samples).reshape(-1)[sample_id]
    l = jnp.array(l_samples).reshape(-1)[sample_id]
    m = jnp.array(m_samples).reshape(-1)[sample_id]

    x_freqs = jnp.array([0, 1, 2, 3])
    y_freqs = jnp.array([0, 1, 2, 3])

    phi = jnp.zeros_like(X)
    phi = phi + phi_0 / 2.0

    coeff_counter = 0
    for freq_x in x_freqs:
        for freq_y in y_freqs:
            phi = phi + coeff_vec[coeff_counter] * jnp.cos(freq_x * 2.0 * jnp.pi * X) * jnp.cos(freq_y * 2.0 * jnp.pi * Y)
            coeff_counter += 1

    phase = (jax.nn.sigmoid(m * (phi - l)) + 0.05) / 1.05

    return phase
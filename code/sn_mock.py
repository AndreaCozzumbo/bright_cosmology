"""
Mock Pantheon+ catalog for a chosen cosmology.

Keeps redshifts and full covariance of the real Pantheon+ sample, and replaces the
magnitudes m_b_corr of the SNe used in the likelihood (zHD > 0.01) with

    m_b = 5 log10( (1+zHEL) (1+zHD) d_A(zHD) ) + 25 + M + noise,

where the noise is drawn from the STAT+SYS covariance. This is the prediction of cobaya's
sn.pantheonplus. M is fitted to the real data, so that the mock has a realistic absolute
magnitude.

The real data are those installed by `cobaya-install sn.pantheonplus -p packages`.
"""
import os

import numpy as np
import pandas as pd

ZMIN = 0.01  # cut of cobaya's sn.pantheonplus


def make_sn_mock(cosmo, name, sn_dir='packages/data/sn_data', seed=43):
    """
    cosmo  : object with angular_distance(z) in Mpc (classy or cosmology.FlatBackground)
    name   : label of the cosmology, used in the file names
    sn_dir : cobaya's sn_data folder, where the mock is written as well

    Returns the path of the .dataset file relative to `sn_dir`, and the fitted M.
    """
    folder = os.path.join(sn_dir, 'PantheonPlus')
    covfile = 'Pantheon+SH0ES_STAT+SYS.cov'
    df = pd.read_csv(os.path.join(folder, 'Pantheon+SH0ES.dat'), sep=r'\s+')
    flat = np.loadtxt(os.path.join(folder, covfile))
    n = int(flat[0])
    assert n == len(df)
    cut = (df['zHD'] > ZMIN).to_numpy()
    cov = flat[1:].reshape(n, n)[np.ix_(cut, cut)]

    z, zhel = df['zHD'].to_numpy()[cut], df['zHEL'].to_numpy()[cut]
    dA = np.array([cosmo.angular_distance(zi) for zi in z])
    mu = 5 * np.log10((1 + zhel) * (1 + z) * dA) + 25

    # generalised least squares for the constant offset M
    weights = np.linalg.solve(cov, np.ones(len(z)))
    M = weights @ (df['m_b_corr'].to_numpy()[cut] - mu) / weights.sum()

    rng = np.random.default_rng(seed)
    df.loc[cut, 'm_b_corr'] = rng.multivariate_normal(mu + M, cov, method='cholesky')

    datafile = f'Pantheon+SH0ES_{name}.dat'
    df.to_csv(os.path.join(folder, datafile), index=False, sep=' ')
    dataset = f'Pantheon_plus_{name}.dataset'
    with open(os.path.join(folder, dataset), 'w') as f:
        f.write(f'name = PANTHEONPLUS\n'
                f'data_file = PantheonPlus/{datafile}\n'
                f'mag_covmat_file = PantheonPlus/{covfile}\n')
    return f'PantheonPlus/{dataset}', M


if __name__ == '__main__':
    import cosmology
    import read_input

    os.chdir(read_input.REPO_ROOT)
    # fiducial cosmology of the mock GW data
    read_input.CONFIG_FILE = 'code/pede_ET.yaml'
    cfg = read_input.read()
    name = cfg['CosmologicalModel']
    cosmo = cosmology.SetCosmology(name, cfg['Parameters'], cfg['Mode'], settings=cfg['Settings'])
    dataset, M = make_sn_mock(cosmo, name)
    print(f'{dataset} written, M = {M:.4f}')

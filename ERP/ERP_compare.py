import os
import json
import numpy as np
import matplotlib as mpl
mpl.use('Agg')

from utils import sampling
from display import plot_perm_test
from cluster_test import permutation_cluster_test

import sys
sys.path.append('../config/')

def significant_cluster(cluster, cluster_p, alpha=0.05):
    if not len(cluster):
        return 0
    sig_cluster = []
    
    for j, p in enumerate(cluster_p):
        if p <= alpha:
            c = cluster[j][0]
            sig_cluster.append([c.start, c.stop])
    return sig_cluster

def ERP_compare(data, ref_data, n_jobs=4):
    """ERP comparison, find significantly different clusters within each lobe.

    Parameters
    ----------
    data: dict
        {lobe: ch * n_trial * time_points}
    ref_data: dict
        {lobe: ch * n_trial * time_points}
    cortex: list

    Returns
    -------
    stats:
    clusters:
    cluster_ps:
    cortex_plot:

    Notes
    -----
    """
    # Input sanity check
    assert set(data.keys()) == set(ref_data.keys()), "Electrodes mismatched"
    cortex = list(data.keys())
    
    # Grouping by lobe
    z_s = []
    clusters = []
    cluster_ps = []
    cortex_plot = []

    for i, lobe in enumerate(cortex):
        window = np.mean(data[lobe], axis=0)
        ref_window = np.mean(ref_data[lobe], axis=0)
    
        # Resample for significance test
        window = sampling(window)
        ref_window = sampling(ref_window)

        z, cluster, cluster_p, _ = permutation_cluster_test([window, ref_window], n_permutations=1000, n_jobs=n_jobs)

        if not len(cluster):
            continue

        z_s.append(z)
        clusters.append(cluster)
        cluster_ps.append(cluster_p)
        cortex_plot.append(lobe)

    return z_s, clusters, cluster_ps, cortex_plot

def save_result(prefix, output_dir, res, save_fig=True):
    z_s, clusters, cluster_ps, cortex_plot = res

    # Saving result to figures
    fig_fname = os.path.join(output_dir, f'{prefix}_perm')
    if save_fig and len(cortex_plot):
        plot_perm_test(clusters, cluster_ps, cortex_plot, z_s, fig_fname)
        
    # Saving result to .json file
    sig_cluster_lobe = {}
    for i, lobe in enumerate(cortex_plot):
        tmp_cluster = significant_cluster(clusters[i], cluster_ps[i])
        sig_cluster_lobe[lobe] = tmp_cluster
        
    output_fname = os.path.join(output_dir, f'{prefix}_significant_cluster.json')
    with open(output_fname, 'w') as f:
        json.dump(sig_cluster_lobe, f)
    print(f"Saved to {output_fname}")
    return

def main(sub_sess, mode, noise=False, silence=False, res_dir='./res/plot_data'):
    output_dir = os.path.join(res_dir, mode)
    assert mode in ["onset", "struct", "struct_vs_onset"], "invalid mode"
    assert not (noise or silence), "silence and noise can't be true at the same time"

    prefix = f'sub{sub_sess}'

    if mode == 'struct_vs_onset':
        fname = os.path.join(res_dir, 'struct', f'{prefix}.npz')
        ref_fname = os.path.join(res_dir, 'onset', f'{prefix}.npz')
    else:
        fname = os.path.join(res_dir, mode, f'{prefix}.npz')
        ref_fname = os.path.join(res_dir, 'reference', f'{prefix}.npz')

    if noise:
        ref_fname = os.path.join(res_dir, 'noise', f'sub{sub_sess}.npz')
    if silence:
        ref_fname = os.path.join(res_dir, 'silence', f'sub{sub_sess}.npz')

    data = np.load(fname)
    ref_data = np.load(ref_fname)
    res = ERP_compare(data, ref_data)

    if not len(res[-1]):
        print("No cluster found.")
        return

    save_result(prefix, output_dir, res)
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('sub_sess', type=str, help='Subject ID and Session ID')
    parser.add_argument('-m', '--mode', dest='mode', type=str, default='struct_vs_onset', help='onset, struct, struct_vs_onset')
    parser.add_argument('-s', '--silence', dest='silence', default=False, action='store_true', help='bool, whether use washout period as reference')
    parser.add_argument('-n', '--noise', dest='noise', default=False, action='store_true', help='bool, whether use noise as reference')

    args = parser.parse_args()
    main(args.sub_sess.upper(), args.mode, args.noise, args.silence)


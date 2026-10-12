# Data

The files shipped in `data/` are **mock data**. They were generated from a flat ΛCDM model
($H_0 = 68$, $\Omega_m = 0.31$, $r_{\rm drag} = 147.1$ Mpc) plus Gaussian noise, in exactly
the formats of the real datasets. They let the code run out of the box, but they are not
suitable for science.

To run the real analysis, download the public datasets with the commands below, from the
repository root. Each command overwrites the corresponding mock file, so no change to
`config.yaml` is needed.

## BAO: DESI DR2 (Cobaya `bao_data`)

```bash
BAO=https://raw.githubusercontent.com/CobayaSampler/bao_data/master/desi_bao_dr2
curl -L -o data/bao/desi_gaussian_bao_ALL_GCcomb_mean.txt $BAO/desi_gaussian_bao_ALL_GCcomb_mean.txt
curl -L -o data/bao/desi_gaussian_bao_ALL_GCcomb_cov.txt  $BAO/desi_gaussian_bao_ALL_GCcomb_cov.txt
```

## Type Ia supernovae (Cobaya `sn_data`)

```bash
SN=https://raw.githubusercontent.com/CobayaSampler/sn_data/master

# Pantheon+ (used by pantheon_plus and pantheon_plus_sh0es), about 33 MB
curl -L -o data/sn/PantheonPlus/Pantheon+SH0ES.dat              "$SN/PantheonPlus/Pantheon%2BSH0ES.dat"
curl -L -o data/sn/PantheonPlus/Pantheon+SH0ES_STAT+SYS.cov     "$SN/PantheonPlus/Pantheon%2BSH0ES_STAT%2BSYS.cov"

# Union3
curl -L -o data/sn/Union3/lcparam_full.txt  $SN/Union3/lcparam_full.txt
curl -L -o data/sn/Union3/mag_covmat.txt    $SN/Union3/mag_covmat.txt

# DES-Dovekie
curl -L -o data/sn/DES-Dovekie/DES-Dovekie_HD.csv     $SN/DES-Dovekie/DES-Dovekie_HD.csv
curl -L -o data/sn/DES-Dovekie/covtot_inv_000.npz     $SN/DES-Dovekie/covtot_inv_000.npz
```

Alternatively, clone both repositories and copy the files:

```bash
git clone --depth 1 https://github.com/CobayaSampler/bao_data.git
git clone --depth 1 https://github.com/CobayaSampler/sn_data.git
cp bao_data/desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_*.txt data/bao/
cp sn_data/PantheonPlus/Pantheon+SH0ES*  data/sn/PantheonPlus/
cp sn_data/Union3/lcparam_full.txt sn_data/Union3/mag_covmat.txt  data/sn/Union3/
cp sn_data/DES-Dovekie/DES-Dovekie_HD.csv sn_data/DES-Dovekie/covtot_inv_000.npz  data/sn/DES-Dovekie/
```

## Cosmic chronometers (Moresco, `CCcovariance`)

The analysis uses the 15 Moresco et al. (2012, 2015, 2016) cosmic-chronometer measurements
and the systematic-error table of Moresco et al. (2020), both from the `CCcovariance`
repository:

```bash
CC=https://gitlab.com/mmoresco/CCcovariance/-/raw/master/data
curl -L -o data/cc/HzTable_MM_BC03.dat $CC/HzTable_MM_BC03.dat
curl -L -o data/cc/data_MM20.dat       $CC/data_MM20.dat
```

The covariance is built as in the `CCcovariance` notebooks: the diagonal errors (`errHz`)
plus the fully correlated IMF and SPS-model (`mod_ooo`) systematics.

## Checking the files

After downloading, run a short test. `config.yaml` uses `supernovae.dataset` to choose the
SN sample; the corresponding files must be present.

```bash
python main.py --no-plot
```

The `[CC]`, `[SN]` and `Data` lines at the start of the log show what was loaded. With the
real Pantheon+ files you should see `Pantheon+: 1624/1701 SNe`.

## References

Please cite the original data papers when you use the data:
* Cosmic chronometers and their covariance: Moresco et al. 2012, 2015, 2016, 2020.
* BAO: DESI Collaboration, DR2 BAO results.
* SNe Ia: Pantheon+ (Scolnic et al. 2022, Brout et al. 2022), Union3 (Rubin et al. 2023),
  DES-SN5YR / Dovekie recalibration.

**Do not commit the downloaded files.** They overwrite the mock files at the same paths,
so `git status` will show them as modified.

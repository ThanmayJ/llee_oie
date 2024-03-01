# llee_oie

## Usage

### Weighted Average (WA)
```bash
python train.py --use_wa True --use_pos True --wt_src 0.6 --wt_pos 0.4
python train.py --use_wa True --use_syndp True --wt_src 0.6 --wt_syndp 0.4
python train.py --use_wa True --use_pos True --use_syndp True --wt_src 0.6 --wt_pos 0.2 --wt_syndp 0.2
```

### Linearized Concatenation (LC)
```bash
python train.py --use_lc True --use_pos True --dim_pos 20
python train.py --use_lc True --use_syndp True --dim_syndp 20
python train.py --use_lc True --use_pos True --use_syndp True --dim_pos 20 --dim_syndp 20
```

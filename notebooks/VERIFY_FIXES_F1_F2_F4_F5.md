# Colab verification run — F1, F2, F4, F5 + the ResNet-18 swap

Paste these cells in order into a fresh Colab notebook (GPU runtime).
Cells 1–4 are the verification; cell 5 is the F5 memory proof and is the
only one that needs a lot of GPU time.

What each fix should show up as:

| Fix | What to look for |
|---|---|
| ResNet-18 swap | `model_dim=` around 11.2M, not 23,628,798 |
| F1 (unmasking) | `aggregate=EXACT` on **every** round line, rounds 0 and 1 |
| F2 (AES mask PRG) | `phase1=` far below the previous 1102.9s |
| F4 (batched sketch) | bootstrap prints "N in one pass"; `phase2=` far below 257.7s |
| F5 (streaming) | cell 5: a `vectors:` line, and RAM that stops climbing |

Baseline to compare against — your 2026-11-13 ResNet-50 run, 8 clients:

```
bootstrap: 33.4s per sample x 10 = ~330s
round 0 timing: phase1=1102.9s, phase2=257.7s, verify=256.9s, total=1617.5s
```

---

## Cell 1 — fresh clone and dependencies

Clones fresh every time on purpose: a stale or half-synced clone caused
two `ModuleNotFoundError` dead ends before, and re-cloning this repo is
cheap.

```python
import os, shutil, importlib

print("kernel cwd:", os.getcwd())   # if this itself raises/errors, the
                                     # OS-level cwd is already broken --
                                     # restart the runtime before anything else

# Move OFF /content/FL-ZKP before deleting it. If the kernel's cwd is
# already inside the directory being rmtree'd (true on a second run of
# this cell in the same kernel, which is exactly what happened the
# first time this was pasted), getcwd() breaks at the OS level for
# every subprocess spawned afterward -- git and pip both failed with
# "shell-init: error retrieving current directory" for this reason,
# and pip silently did NOT upgrade pynacl, which is the real cause of
# the "ristretto255 available: False" that followed.
%cd /content
shutil.rmtree('/content/FL-ZKP', ignore_errors=True)
!git clone -q https://github.com/YashasweeNarayan/FL-ZKP.git /content/FL-ZKP
%cd /content/FL-ZKP
!git log --oneline -1

# pynacl: the EC layer. --upgrade matters here: Colab's base image
# ships an old pynacl, and `pip install pynacl` with no version pin is
# a no-op if ANY version is already present. nacl.bindings.
# crypto_core_ristretto255_from_hash (and the rest of ristretto255) only
# exists from pynacl>=1.5.0 -- an older one imports fine and fails much
# later with a bare AttributeError, which is what happened on the first
# run of this cell. cryptography: F2's AES-CTR mask PRG (no slow
# fallback exists on purpose -- two parties on different expanders
# would derive different masks). flwr-datasets: FEMNIST.
!pip install -q --upgrade --force-reinstall 'pynacl>=1.5.0' flwr-datasets cryptography

import sys
sys.path.insert(0, '/content/FL-ZKP')
for mod in ('torch', 'nacl', 'cryptography', 'flwr_datasets', 'numpy'):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:16s} OK  {getattr(m, '__version__', '')}")
    except ImportError as e:
        print(f"{mod:16s} MISSING -> {e}")

import nacl.bindings as _nb
print("ristretto255 available:", hasattr(_nb, 'crypto_core_ristretto255_from_hash'))

import torch
print("CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
```

Expect the commit line to read `10656e4` or later. If `cryptography`
is missing the run will fail immediately and loudly — that is deliberate.

**If this cell upgraded pynacl (the previous attempt had an older one
already loaded), restart the runtime now — Runtime → Restart session —
and re-run every cell from here.** A live kernel keeps the old
`nacl.bindings` module even after `pip install --upgrade` finishes; only
a restart actually loads the new one. If you skip this and hit
`crypto_core_ristretto255_from_hash` again, this is why — the fix just
made the error message say so explicitly instead of a bare
`AttributeError` three frames deep.

---

## Cell 2 — config, Drive, FEMNIST

```python
import yaml
from google.colab import drive
drive.mount('/content/drive')

with open('configs/mvp_3day.yaml') as f:
    CFG = yaml.safe_load(f)
print("models in config:", CFG['mvp_scope']['models'])   # resnet18 first now

from fl.data_noniid import prepare_case2
femnist_cfg = CFG['datasets']['femnist']
case2_data = prepare_case2(seed=CFG['seed'], cache_dir=femnist_cfg['cache_dir'])
print("clients:", len(case2_data['client_train']))
print("unseen_test writers:", len(case2_data['split'].unseen_test_writers))
```

---

## Cell 3 — models and keys

```python
from fl.models import build_model, flatten_state_dict

# Only resnet18 for this verification run, to spend as little GPU
# quota as possible. Add 'mobilenet_v2' back once the fixes check out.
MODELS_TO_RUN = ['resnet18']

models = {name: build_model(name, num_classes=62, in_channels=1) for name in MODELS_TO_RUN}
flat_weights = {}
for name, m in models.items():
    flat, _ = flatten_state_dict(m)
    flat_weights[name] = flat
    print(f"{name} model_dim = {flat.numel():,}")   # <-- resnet18 real number

from crypto.ristretto import Scalar, scalar_mul_base
from crypto.dkg import generate_contribution, finalize
from fl.round_loop import build_x25519_keys, precompute_pairwise_secrets

n_clients = len(case2_data['client_train'])
t = CFG['protocol']['threshold']
n_c = CFG['protocol']['n_key_holders']

client_sks = [Scalar.random() for _ in range(n_clients)]
ring = [scalar_mul_base(sk) for sk in client_sks]
x25519_sks, x25519_pks = build_x25519_keys(n_clients)
pairwise_secrets = precompute_pairwise_secrets(x25519_sks, x25519_pks)

key_holder_ids = list(range(1, n_c + 1))
PK_trace, key_holder_shares = finalize(
    [generate_contribution(h, t, key_holder_ids) for h in key_holder_ids], key_holder_ids)
print(f"ring of {len(ring)} published, PK_trace among {n_c} holders (t={t})")
```

`model_dim` here is the number to report back — my 11.2M was an estimate
from standard ResNet-18 parameter counts, not a measurement.

---

## Cell 4 — the verification run (8 clients, 2 rounds)

```python
import time, numpy as np
from fl.round_loop import run_training_loop, bootstrap_norms_from_model_dim, release_gpu_memory

SMOKE_ROUNDS = 2
SMOKE_CLIENTS = 8        # must be >= K_MIN (8) in fl/server.py

results = {}
for model_name in MODELS_TO_RUN:
    print(f"\n==== {model_name} (verification: {SMOKE_ROUNDS} rounds, {SMOKE_CLIENTS} clients) ====")
    model = models[model_name]
    dim = flat_weights[model_name].numel()

    t0 = time.time()
    root_norms = bootstrap_norms_from_model_dim(
        model_dim=dim,
        sketch_dim=CFG['sketch']['m'],
        sketch_seed=f"bootstrap-{model_name}".encode(),
        quant_bits=16,
        rng=np.random.default_rng(CFG['seed']),
    )
    print(f"bootstrap total: {time.time()-t0:.1f}s  (was ~330s at resnet50, one sample at a time)")

    final_w_t, history = run_training_loop(
        model=model,
        model_dim=dim,
        client_datasets=case2_data['client_train'][:SMOKE_CLIENTS],
        ring=ring[:SMOKE_CLIENTS],
        client_sks=client_sks[:SMOKE_CLIENTS],
        pairwise_secrets={i: {j: pairwise_secrets[i][j]
                              for j in range(SMOKE_CLIENTS) if j != i}
                          for i in range(SMOKE_CLIENTS)},
        pk_trace=PK_trace,
        server_root_norms=root_norms,
        CFG=CFG,
        rounds=SMOKE_ROUNDS,
    )
    results[model_name] = {'final_w_t': final_w_t, 'history': history}
    release_gpu_memory()

print("\n--- per-round summary ---")
for model_name, r in results.items():
    for h in r['history']:
        print(model_name, h)
```

**The one thing that must hold:** every round line says
`aggregate=EXACT`. If a round instead prints `AGGREGATE MISMATCH`, F1 is
wrong and nothing downstream is trustworthy — send me the output and stop
there.

At 8 clients and ResNet-18 the vectors fit inside the 2 GiB budget, so
there will be **no** `vectors:` line in cell 4. That absence is expected;
cell 5 is what exercises the spill path.

Also worth noting: with `aggregate=EXACT` holding, the round-1
overflow/NaN warning you used to see should be gone, since the weights
are no longer being destroyed in round 0.

---

## Cell 5 — F5 memory proof (30 clients — the count that crashed before)

Only run this if you have GPU time to spare. 30 clients at ResNet-18 is
5.0 GiB of client vectors, which used to be most of the way to the
12.7 GB ceiling; it should now sit near 2.5 GiB with the rest on disk.

```python
import psutil, os, time, numpy as np
from fl.round_loop import run_training_loop, bootstrap_norms_from_model_dim

MEM_CLIENTS = 30
proc = psutil.Process(os.getpid())
print(f"RSS before: {proc.memory_info().rss / 2**30:.2f} GiB")

model_name = MODELS_TO_RUN[0]
dim = flat_weights[model_name].numel()
print(f"client vectors the old code would have pinned: "
      f"{2 * MEM_CLIENTS * dim * 8 / 2**30:.1f} GiB")

root_norms = bootstrap_norms_from_model_dim(
    model_dim=dim, sketch_dim=CFG['sketch']['m'],
    sketch_seed=b"bootstrap-mem", quant_bits=16,
    rng=np.random.default_rng(CFG['seed']), verbose=False)

t0 = time.time()
_, mem_history = run_training_loop(
    model=models[model_name],
    model_dim=dim,
    client_datasets=case2_data['client_train'][:MEM_CLIENTS],
    ring=ring[:MEM_CLIENTS],
    client_sks=client_sks[:MEM_CLIENTS],
    pairwise_secrets={i: {j: pairwise_secrets[i][j]
                          for j in range(MEM_CLIENTS) if j != i}
                      for i in range(MEM_CLIENTS)},
    pk_trace=PK_trace,
    server_root_norms=root_norms,
    CFG=CFG,
    rounds=1,
)
print(f"\nRSS after: {proc.memory_info().rss / 2**30:.2f} GiB   "
      f"(wall {time.time()-t0:.0f}s)")
print(mem_history)
```

What to look for:

- a `round 0 vectors: ... spilled to disk ...` line — the store doing its job;
- `aggregate=EXACT`;
- RSS staying in the low single-digit GiB instead of climbing past 10.

One caveat on this cell: with 30 clients the share threshold is
`(2*30)//3 + 1 = 21`, so if more than 9 clients get rejected the round
aborts with "only N distinct self-mask shares, need 21". That is correct
behaviour, not a bug — but it means a V3 rejection storm would show up
here as an unmasking abort.

---

## If something fails

Send me the full output. The things I most want to see either way:

1. the real `model_dim` for ResNet-18;
2. the bootstrap total and the `round N timing:` line;
3. whether `aggregate=EXACT` held on both rounds;
4. from cell 5, the `vectors:` line and the before/after RSS.

Known untested-until-now paths, in case one of them is what breaks:
`TorchSketchProjector.project_batch` (F4 on GPU) and the whole
`round_loop` store wiring (F5) — neither has ever run anywhere torch is
actually installed.

"""Direct Preference Optimization on top of the SFT LoRA adapter, in plain MLX.

mlx-lm (0.32) ships SFT/LoRA trainers but no DPO trainer, so this module implements the standard DPO
objective (Rafailov et al., 2023) directly:

    L = -log sigmoid( beta * [ (log pi(y_w|x) - log ref(y_w|x)) - (log pi(y_l|x) - log ref(y_l|x)) ] )

* policy    = base model + the SFT adapter, whose LoRA weights keep training (nothing else is trainable)
* reference = the same model *before* any DPO update, i.e. the SFT model. Its sequence log-probabilities are
              computed once up front, so no second copy of the model is held in memory.
* sequence log-probs are sums over the completion tokens only (prompt tokens are masked), including the
  closing `<|im_end|>` so ending the reply correctly is part of what is preferred.
* dropout is off for both policy and reference, so at step 0 the policy equals the reference and the loss is
  exactly ln 2 (a built-in sanity check that is logged).

    python -m finsight_llm.dpo --pairs data/preferences --adapter artifacts/adapters --out artifacts/adapters_dpo
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

END_OF_TURN = "<|im_end|>"


def load_pairs(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def encode_pair(tok, pair: dict, max_len: int) -> dict:
    """Token ids for prompt, chosen and rejected completions (prompt rendered with the chat template)."""
    prompt = tok.apply_chat_template(pair["prompt"], add_generation_prompt=True, tokenize=False)
    p = tok.encode(prompt)
    c = tok.encode(pair["chosen"] + END_OF_TURN)
    r = tok.encode(pair["rejected"] + END_OF_TURN)
    return {"prompt": p, "chosen": c[: max(1, max_len - len(p))], "rejected": r[: max(1, max_len - len(p))]}


def dpo_loss_value(policy_margin: float, ref_margin: float, beta: float) -> float:
    """Scalar DPO loss from the two chosen-minus-rejected log-prob margins (for tests and logging)."""
    z = beta * (policy_margin - ref_margin)
    return math.log1p(math.exp(-z)) if z > -30 else -z


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="artifacts/adapters", help="SFT adapter to start from (and to use as reference)")
    ap.add_argument("--pairs", default="data/preferences", type=Path)
    ap.add_argument("--out", default="artifacts/adapters_dpo", type=Path)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--iters", type=int, default=700)
    ap.add_argument("--batch", type=int, default=4, help="preference pairs per step (2x sequences)")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--sft-alpha", type=float, default=0.0, help="optional weight of a per-token NLL term on the chosen reply")
    ap.add_argument("--grad-checkpoint", action="store_true", help="recompute activations in backward (less memory)")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="use only the first N training pairs (smoke tests)")
    args = ap.parse_args(argv)

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load

    random.seed(args.seed)
    mx.random.seed(args.seed)
    model, tok = load(args.model, adapter_path=args.adapter)
    model.freeze()
    # only the LoRA matrices of the SFT adapter train
    for _, m in model.named_modules():
        if hasattr(m, "lora_a") and hasattr(m, "lora_b"):
            m.unfreeze(keys=["lora_a", "lora_b"], strict=False)
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    if args.grad_checkpoint:
        from mlx_lm.tuner.trainer import grad_checkpoint
        grad_checkpoint(model.layers[0])
    model.eval()  # dropout off for policy and reference

    train = load_pairs(args.pairs / "train.jsonl")
    # one pass over a seeded subset of exactly the pairs the run will consume (reference log-probs are the slow part)
    train = random.Random(args.seed).sample(train, min(len(train), args.iters * args.batch))[: args.limit]
    valid = load_pairs(args.pairs / "valid.jsonl")[: args.limit]
    enc_train = [encode_pair(tok, p, args.max_len) for p in train]
    enc_valid = [encode_pair(tok, p, args.max_len) for p in valid]
    print(f"trainable params {n_train / 1e6:.3f}M | train pairs {len(train)} | valid pairs {len(valid)} | "
          f"beta {args.beta} lr {args.lr} iters {args.iters} batch {args.batch}", flush=True)

    def make_batch(encs):
        seqs, starts, ends = [], [], []
        for e in encs:
            for key in ("chosen", "rejected"):
                seqs.append(e["prompt"] + e[key])
                starts.append(len(e["prompt"]))
                ends.append(len(e["prompt"]) + len(e[key]))
        width = max(len(s) for s in seqs)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        ids = mx.array([s + [pad] * (width - len(s)) for s in seqs])
        pos = mx.arange(width - 1)[None, :]
        # target position t predicts token t+1; keep targets inside the completion span
        mask = ((pos + 1) >= mx.array(starts)[:, None]) & ((pos + 1) < mx.array(ends)[:, None])
        return ids, mask.astype(mx.float32)

    def seq_logps(m, ids, mask):
        logits = m(ids[:, :-1]).astype(mx.float32)
        tgt = ids[:, 1:]
        picked = mx.take_along_axis(logits, tgt[..., None], axis=-1).squeeze(-1)
        lp = picked - mx.logsumexp(logits, axis=-1)
        return (lp * mask).sum(-1), mask.sum(-1)

    # ---- reference log-probs, once, from the untouched SFT model
    t_ref = time.time()
    ref: dict[int, tuple[float, float]] = {}

    def fill_ref(encs, offset):
        for i in range(0, len(encs), 4):
            chunk = encs[i: i + 4]
            ids, mask = make_batch(chunk)
            lp, _ = seq_logps(model, ids, mask)
            mx.clear_cache()
            mx.eval(lp)
            vals = lp.tolist()
            for j in range(len(chunk)):
                ref[offset + i + j] = (vals[2 * j], vals[2 * j + 1])

    cache = args.out / "ref_logps.json"
    key = [len(train), len(valid), args.max_len, args.seed, args.iters, args.batch]
    args.out.mkdir(parents=True, exist_ok=True)
    if cache.exists() and json.loads(cache.read_text())["key"] == key:
        ref = {int(k): tuple(v) for k, v in json.loads(cache.read_text())["ref"].items()}
    else:
        fill_ref(enc_train, 0)
        fill_ref(enc_valid, 1_000_000)
        cache.write_text(json.dumps({"key": key, "ref": {str(k): v for k, v in ref.items()}}))
    print(f"reference log-probs for {len(ref)} pairs in {time.time() - t_ref:.0f}s", flush=True)

    def loss_fn(m, ids, mask, ref_c, ref_r):
        lp, ntok = seq_logps(m, ids, mask)
        pc, pr = lp[0::2], lp[1::2]
        z = args.beta * ((pc - ref_c) - (pr - ref_r))
        loss = nn.losses.binary_cross_entropy(z, mx.ones_like(z), with_logits=True, reduction="mean")
        if args.sft_alpha:
            loss = loss + args.sft_alpha * (-(pc / ntok[0::2])).mean()
        return loss, z

    def evaluate(encs, offset):
        losses, accs, margins = [], [], []
        for i in range(0, len(encs), 4):
            chunk = encs[i: i + 4]
            ids, mask = make_batch(chunk)
            rc = mx.array([ref[offset + i + j][0] for j in range(len(chunk))])
            rr = mx.array([ref[offset + i + j][1] for j in range(len(chunk))])
            loss, z = loss_fn(model, ids, mask, rc, rr)
            mx.eval(loss, z)
            losses += [loss.item()] * len(chunk)
            accs += (z > 0).astype(mx.float32).tolist()
            margins += (z / args.beta).tolist()
        n = len(losses)
        return {"loss": sum(losses) / n, "reward_acc": sum(accs) / n, "margin": sum(margins) / n}

    warm = optim.linear_schedule(1e-7, args.lr, args.warmup)
    cosine = optim.cosine_decay(args.lr, max(1, args.iters - args.warmup), args.lr * 0.1)
    opt = optim.AdamW(learning_rate=optim.join_schedules([warm, cosine], [args.warmup]), weight_decay=0.0)
    vg = nn.value_and_grad(model, loss_fn)

    args.out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((Path(args.adapter) / "adapter_config.json").read_text())
    cfg["adapter_path"] = str(args.out)
    (args.out / "adapter_config.json").write_text(json.dumps(cfg, indent=4) + "\n")

    def save(tag: str | None = None):
        w = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(args.out / "adapters.safetensors"), w)
        if tag:
            shutil.copy(args.out / "adapters.safetensors", args.out / f"{tag}_adapters.safetensors")

    first = evaluate(enc_valid, 1_000_000)
    print(f"iter 0 | valid loss {first['loss']:.4f} (ln2 = {math.log(2):.4f}) acc {first['reward_acc']:.3f} "
          f"margin {first['margin']:.3f}", flush=True)
    history = [{"iter": 0, **{f"valid_{k}": v for k, v in first.items()}}]

    order: list[int] = []
    t0 = time.time()
    run_loss = run_acc = 0.0
    seen = 0
    diverged = False
    for it in range(1, args.iters + 1):
        if len(order) < args.batch:
            order += random.sample(range(len(enc_train)), len(enc_train))
        idx = [order.pop() for _ in range(args.batch)]
        ids, mask = make_batch([enc_train[i] for i in idx])
        rc = mx.array([ref[i][0] for i in idx])
        rr = mx.array([ref[i][1] for i in idx])
        (loss, z), grads = vg(model, ids, mask, rc, rr)
        if args.grad_clip:
            grads, _ = optim.clip_grad_norm(grads, args.grad_clip)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss, z)
        lv = loss.item()
        run_loss += lv
        run_acc += (z > 0).astype(mx.float32).mean().item()
        seen += 1
        if not math.isfinite(lv) or lv > 5.0:
            print(f"iter {it} | loss {lv} -> diverged, stopping (lower --lr)", flush=True)
            diverged = True
            break
        if it % 20 == 0:
            print(f"iter {it} | train loss {run_loss / seen:.4f} acc {run_acc / seen:.3f} | "
                  f"{(time.time() - t0) / it:.2f}s/it | peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
            run_loss = run_acc = 0.0
            seen = 0
        if it % args.eval_every == 0 or it == args.iters:
            v = evaluate(enc_valid, 1_000_000)
            history.append({"iter": it, **{f"valid_{k}": x for k, x in v.items()}})
            print(f"iter {it} | valid loss {v['loss']:.4f} acc {v['reward_acc']:.3f} margin {v['margin']:.3f}", flush=True)
        if it % args.save_every == 0:
            save(f"{it:07d}")
    if not diverged:
        save()
    stats = {"beta": args.beta, "lr": args.lr, "iters": it, "batch_pairs": args.batch, "diverged": diverged,
             "train_pairs": len(train), "valid_pairs": len(valid), "trainable_params": n_train,
             "seconds": time.time() - t0, "peak_memory_gb": mx.get_peak_memory() / 1e9, "valid_history": history,
             "sft_alpha": args.sft_alpha, "grad_checkpoint": args.grad_checkpoint, "grad_clip": args.grad_clip}
    (args.out / "dpo_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "valid_history"}), flush=True)


if __name__ == "__main__":
    main()

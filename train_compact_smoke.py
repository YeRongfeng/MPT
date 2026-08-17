"""Short GPU training check for the compact Path MeanFlow generator.

This deliberately reuses the production Stage-1 dataset and loss helpers, but
does not write a deployable checkpoint or alter the formal training entrypoint.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dit.Models import CompactPathMeanFlowTransformer
from posterior_pipeline import make_partial_dataset, prior_transport_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataFolder", default="data/dataset0")
    parser.add_argument("--output-dir", default="diagnostics/compact_training_smoke")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--train-contexts", type=int, default=4)
    parser.add_argument("--val-contexts", type=int, default=2)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--inner", type=int, default=768)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU training smoke test")
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(args.seed)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_set, train_envs = make_partial_dataset(
        args.dataFolder,
        "train",
        mask_seed=args.seed,
        p_mask=0.5,
        mask_mode="stage1_demo_valid",
        dynamic_mask_noise=False,
    )
    val_set, val_envs = make_partial_dataset(
        args.dataFolder,
        "val",
        mask_seed=args.seed,
        p_mask=0.5,
        mask_mode="stage1_demo_valid",
        dynamic_mask_noise=False,
    )
    train_set = Subset(train_set, range(min(len(train_set), args.train_contexts)))
    val_set = Subset(val_set, range(min(len(val_set), args.val_contexts)))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = CompactPathMeanFlowTransformer(
        n_layers=args.layers,
        n_heads=args.heads,
        d_model=args.width,
        d_inner=args.inner,
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    model.train()
    train_losses = []
    train_iter = iter(train_loader)
    for update in range(args.updates):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        optimizer.zero_grad(set_to_none=True)
        loss, terms = prior_transport_loss(model, batch, device)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at update {update}: {loss}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(float(loss.detach().cpu()))
        print(
            f"update={update + 1}/{args.updates} "
            f"loss={train_losses[-1]:.7f} grad={float(grad_norm):.5f} "
            f"flow={float(terms['flow']):.7f} endpoint={float(terms['endpoint']):.7f}",
            flush=True,
        )

    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            loss, _ = prior_transport_loss(model, batch, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite validation loss: {loss}")
            val_losses.append(float(loss.detach().cpu()))

    summary = {
        "status": "smoke_pass",
        "architecture": "compact",
        "device": torch.cuda.get_device_name(0),
        "seed": args.seed,
        "train_environments": train_envs,
        "validation_environments": val_envs,
        "train_contexts": len(train_set),
        "validation_contexts": len(val_set),
        "updates": args.updates,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "layers": args.layers,
        "heads": args.heads,
        "width": args.width,
        "inner": args.inner,
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "train_loss_first": train_losses[0],
        "train_loss_last": train_losses[-1],
        "validation_loss": float(np.mean(val_losses)) if val_losses else None,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

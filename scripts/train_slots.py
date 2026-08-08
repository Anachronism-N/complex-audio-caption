#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from sceneledger.models.losses import slot_set_loss
from sceneledger.models.slots import TrackEventSlotConfig, TrackEventSlotDecoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--config", default="configs/model/track_event_slots.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(config["training"].get("seed", 20260808))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    paths = sorted(Path(args.features).glob("*.npz"))
    if not paths:
        raise ValueError(f"No NPZ features in {args.features}")
    first = np.load(paths[0])
    model_config = TrackEventSlotConfig(
        input_dim=int(first["features"].shape[1]), **config["model"]
    )
    model = TrackEventSlotDecoder(model_config).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    epochs = int(config["training"].get("epochs", 10))
    gradient_clip = float(config["training"].get("gradient_clip", 1.0))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(epochs):
        random.shuffle(paths)
        totals = []
        for path in paths:
            item = np.load(path)
            features = torch.from_numpy(item["features"]).unsqueeze(0).to(args.device)
            target = {
                key: torch.from_numpy(item[key]).to(args.device)
                for key in (
                    "track_type",
                    "track_activity",
                    "event_type",
                    "event_activity",
                    "event_track",
                )
            }
            losses = slot_set_loss(model(features), [target])
            loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            totals.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(totals))
        history.append({"epoch": epoch + 1, "loss": epoch_loss})
        print(json.dumps(history[-1]))
        torch.save(
            {"model": model.state_dict(), "config": config, "epoch": epoch + 1},
            output / "last.pt",
        )
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

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
    parser.add_argument("--validation-features")
    parser.add_argument("--resume")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--overfit-samples", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(config["training"].get("seed", 20260808))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_paths = sorted(Path(args.features).glob("*.npz"))
    if not train_paths:
        raise ValueError(f"No NPZ features in {args.features}")
    validation_paths = (
        sorted(Path(args.validation_features).glob("*.npz"))
        if args.validation_features
        else []
    )
    if args.validation_features and not validation_paths:
        raise ValueError(f"No validation NPZ features in {args.validation_features}")
    overfit_samples = args.overfit_samples or config["training"].get("overfit_samples")
    if overfit_samples:
        train_paths = train_paths[: int(overfit_samples)]
        validation_paths = []
    with np.load(train_paths[0]) as first:
        input_dim = int(first["features"].shape[1])
    model_config = TrackEventSlotConfig(
        input_dim=input_dim, **config["model"]
    )
    device = torch.device(args.device)
    model = TrackEventSlotDecoder(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    epochs = int(config["training"].get("epochs", 10))
    gradient_clip = float(config["training"].get("gradient_clip", 1.0))
    accumulation_steps = int(config["training"].get("gradient_accumulation_steps", 1))
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.json"
    history = (
        json.loads(history_path.read_text(encoding="utf-8"))
        if args.resume and history_path.exists()
        else []
    )
    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        best_loss = float(checkpoint.get("best_loss", best_loss))

    for epoch in range(start_epoch, epochs):
        train_loss = _run_epoch(
            model,
            train_paths,
            device,
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip=gradient_clip,
            accumulation_steps=accumulation_steps,
            use_amp=use_amp,
            shuffle_seed=seed + epoch,
        )
        validation_loss = (
            _run_epoch(model, validation_paths, device, use_amp=use_amp)
            if validation_paths
            else None
        )
        monitored_loss = validation_loss if validation_loss is not None else train_loss
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(json.dumps(history[-1]))
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "epoch": epoch + 1,
            "best_loss": min(best_loss, monitored_loss),
            "input_dim": input_dim,
            "train_samples": len(train_paths),
            "validation_samples": len(validation_paths),
        }
        torch.save(state, output / "last.pt")
        if monitored_loss < best_loss:
            best_loss = monitored_loss
            state["best_loss"] = best_loss
            torch.save(state, output / "best.pt")
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def _run_epoch(
    model: TrackEventSlotDecoder,
    paths: list[Path],
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    gradient_clip: float = 1.0,
    accumulation_steps: int = 1,
    use_amp: bool = False,
    shuffle_seed: int | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    ordered = list(paths)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(ordered)
    totals: list[float] = []
    if training:
        optimizer.zero_grad(set_to_none=True)
    for index, path in enumerate(ordered):
        with np.load(path) as item:
            features = torch.from_numpy(item["features"]).unsqueeze(0).to(device)
            target = {
                key: torch.from_numpy(item[key]).to(device)
                for key in (
                    "track_type",
                    "track_activity",
                    "event_type",
                    "event_activity",
                    "event_track",
                )
            }
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, enabled=use_amp
        ):
            losses = slot_set_loss(model(features), [target])
            loss = sum(losses.values())
        totals.append(float(loss.detach().cpu()))
        if not training:
            continue
        scaled_loss = loss / accumulation_steps
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        should_step = (index + 1) % accumulation_steps == 0 or index + 1 == len(ordered)
        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    if not totals:
        raise ValueError("Epoch contains no feature files")
    return float(np.mean(totals))


if __name__ == "__main__":
    main()

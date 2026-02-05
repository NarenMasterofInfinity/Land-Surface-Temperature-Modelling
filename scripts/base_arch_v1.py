from __future__ import annotations

import argparse
import time
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from arch_v1_data import DummyLSTDataset, build_zarr_datasets, collate_batch
from arch_v1_model import LSTFusionModel, build_default_channel_indices
from arch_v1_train import StageConfig, TrainConfig, save_checkpoint, train_one_epoch, validate
from arch_v1_utils import build_output_dirs, compute_input_stats, normalize_batch_global, plot_metrics, save_metrics, set_seed, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, default=".")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-doy", action="store_true")
    parser.add_argument("--use-dummy-data", action="store_true")
    parser.add_argument("--dummy-samples", type=int, default=64)
    parser.add_argument("--dummy-h", type=int, default=128)
    parser.add_argument("--dummy-w", type=int, default=128)
    parser.add_argument("--root-30m", type=str, default="madurai_30m.zarr")
    parser.add_argument("--root-daily", type=str, default="madurai.zarr")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--samples-per-epoch", type=int, default=1000)
    parser.add_argument("--samples-val", type=int, default=500)
    parser.add_argument("--quality-csv", type=str, default="metrics/arch_v1/date_quality.csv")
    parser.add_argument("--stage-a", type=int, default=20)
    parser.add_argument("--stage-b", type=int, default=90)
    parser.add_argument("--stage-c", type=int, default=0)
    parser.add_argument("--stage-a2", type=int, default=40)
    parser.add_argument("--lambda-start", type=float, default=0.02)
    parser.add_argument("--lambda-end", type=float, default=0.10)
    parser.add_argument("--save-fig-every", type=int, default=5)
    parser.add_argument("--freeze-gate", action="store_true", default=True)
    parser.add_argument("--gate-lr-mult", type=float, default=0.1)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dirs = build_output_dirs(args.root_dir)
    logger = setup_logging(f"{out_dirs['logs']}/train.log")

    ch = build_default_channel_indices()
    model = LSTFusionModel(
        era5_idx=ch["era5"],
        s2_idx=ch["s2"],
        s1_idx=ch["s1"],
        dem_idx=ch["dem"],
        world_idx=ch["world"],
        dyn_idx=ch["dyn"],
        use_doy=args.use_doy,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    try:
        from torchinfo import summary  # type: ignore

        h = args.dummy_h if args.use_dummy_data else args.patch_size
        w = args.dummy_w if args.use_dummy_data else args.patch_size
        dummy_x = torch.zeros(1, 36, h, w, device=device)
        dummy_doy = torch.zeros(1, device=device)
        logger.info("model summary:\n%s", summary(model, input_data=(dummy_x, dummy_doy), verbose=0))
    except Exception as exc:
        logger.warning("torchinfo summary skipped: %s", exc)

    train_cfg = TrainConfig(batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay, grad_clip=args.grad_clip)
    stage_cfg = StageConfig(
        stage_a_epochs=args.stage_a,
        stage_b_epochs=args.stage_b,
        stage_a2_epochs=args.stage_a2,
        ramp_epochs=30,
        lambda_start=args.lambda_start,
        lambda_end=args.lambda_end,
    )

    if args.use_dummy_data:
        in_ch = 36
        dataset = DummyLSTDataset(args.dummy_samples, in_ch, args.dummy_h, args.dummy_w)
        n_train = int(0.8 * len(dataset))
        train_set, val_set = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])
        mask_idx = []
        mu_x = torch.zeros(in_ch)
        sigma_x = torch.ones(in_ch)
    else:
        train_set, val_set, info = build_zarr_datasets(
            root_30m_path=args.root_30m,
            root_daily_path=args.root_daily,
            patch_size=args.patch_size,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.seed,
            samples_per_epoch=args.samples_per_epoch,
            samples_val=args.samples_val,
            quality_csv=args.quality_csv,
        )
        sizes = train_set.channel_sizes()
        offsets = {}
        idx = 0
        for name in ["era5", "s1", "s2", "dem", "world", "dyn"]:
            offsets[name] = idx
            idx += sizes[name]
        mask_idx = list(range(offsets["world"], offsets["world"] + sizes["world"])) + list(
            range(offsets["dyn"], offsets["dyn"] + sizes["dyn"])
        )
        mu_x, sigma_x = compute_input_stats(train_set, n_samples=min(200, len(train_set)), mask_idx=mask_idx)
        logger.info(
            "dataset: available=%d train_dates=%d val_dates=%d landsat=%d modis=%d viirs=%d",
            info["available_dates"],
            info["train_dates"],
            info["val_dates"],
            info["landsat_present"],
            info["modis_present"],
            info["viirs_present"],
        )
        logger.info("quality report saved: %s", info["quality_csv"])
        logger.info("input_stats computed: channels=%d mask_channels=%d", int(mu_x.numel()), len(mask_idx))

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
        pin_memory=True,
    )

    if args.freeze_gate:
        for p in model.gate.parameters():
            p.requires_grad = False
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
    else:
        gate_params = list(model.gate.parameters())
        gate_ids = {id(p) for p in gate_params}
        main_params = [p for p in model.parameters() if id(p) not in gate_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": main_params, "lr": train_cfg.lr},
                {"params": gate_params, "lr": train_cfg.lr * args.gate_lr_mult},
            ],
            weight_decay=train_cfg.weight_decay,
        )
    base_lr = train_cfg.lr
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_rmse = float("inf")
    history: List[Dict[str, float]] = []

    for epoch in range(args.epochs):
        stage, lam, cons_weight = stage_cfg.stage_at(epoch)
        lr_mult = 0.1 if stage == "A2" else 1.0
        for group in optimizer.param_groups:
            group["lr"] = base_lr * lr_mult
        t0 = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            stage,
            lam,
            cons_weight,
            train_cfg,
            stage_cfg.force_alpha_weak,
            logger=logger,
            mu_x=mu_x,
            sigma_x=sigma_x,
            mask_idx=mask_idx,
        )
        save_fig_dir = out_dirs["figures"] if (epoch % args.save_fig_every == 0) else None
        val_metrics = validate(
            model,
            val_loader,
            device,
            save_fig_dir,
            epoch,
            stage,
            lam,
            cons_weight,
            train_cfg,
            stage_cfg.force_alpha_weak,
            mu_x=mu_x,
            sigma_x=sigma_x,
            mask_idx=mask_idx,
        )
        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "stage": stage,
            "lambda": lam,
            "train_loss": train_metrics["train_loss"],
            "val_loss": val_metrics["val_loss"],
            "train_rmse_all": train_metrics["train_rmse_all"],
            "val_rmse_all": val_metrics["val_rmse_all"],
            "train_rmse_ls": train_metrics["train_rmse_ls"],
            "val_rmse_ls": val_metrics["val_rmse_ls"],
            "train_rmse_wk": train_metrics["train_rmse_wk"],
            "val_rmse_wk": val_metrics["val_rmse_wk"],
            "train_alpha_mean": train_metrics["train_alpha_mean"],
            "train_alpha_std": train_metrics["train_alpha_std"],
            "train_alpha_p01": train_metrics["train_alpha_p01"],
            "train_alpha_p09": train_metrics["train_alpha_p09"],
            "val_alpha_mean": val_metrics["val_alpha_mean"],
            "val_alpha_std": val_metrics["val_alpha_std"],
            "val_alpha_p01": val_metrics["val_alpha_p01"],
            "val_alpha_p09": val_metrics["val_alpha_p09"],
            "skipped_batches": train_metrics["skipped_batches"],
            "lr": train_cfg.lr,
            "time_sec": elapsed,
        }
        history.append(row)
        save_metrics(f"{out_dirs['metrics']}/metrics.csv", history)
        plot_metrics(history, out_dirs["figures"])

        logger.info(
            "epoch=%03d stage=%s lambda=%.3f lr=%.6f train_loss=%.4f train_rmse_all=%.4f train_rmse_ls=%.4f train_rmse_wk=%.4f "
            "val_loss=%.4f val_rmse_all=%.4f val_rmse_ls=%.4f val_rmse_wk=%.4f "
            "alpha_train(mean=%.3f std=%.3f p01=%.3f p09=%.3f) alpha_val(mean=%.3f std=%.3f p01=%.3f p09=%.3f) time=%.1fs",
            epoch,
            stage,
            lam,
            optimizer.param_groups[0]["lr"],
            train_metrics["train_loss"],
            train_metrics["train_rmse_all"],
            train_metrics["train_rmse_ls"],
            train_metrics["train_rmse_wk"],
            val_metrics["val_loss"],
            val_metrics["val_rmse_all"],
            val_metrics["val_rmse_ls"],
            val_metrics["val_rmse_wk"],
            train_metrics["train_alpha_mean"],
            train_metrics["train_alpha_std"],
            train_metrics["train_alpha_p01"],
            train_metrics["train_alpha_p09"],
            val_metrics["val_alpha_mean"],
            val_metrics["val_alpha_std"],
            val_metrics["val_alpha_p01"],
            val_metrics["val_alpha_p09"],
            elapsed,
        )

        if val_metrics["val_rmse_ls"] < best_rmse:
            best_rmse = val_metrics["val_rmse_ls"]
            save_checkpoint(f"{out_dirs['models']}/best.pt", model, optimizer, epoch, best_rmse)

    save_checkpoint(f"{out_dirs['models']}/last.pt", model, optimizer, args.epochs - 1, best_rmse)
    logger.info("done. best_rmse=%.4f", best_rmse)


if __name__ == "__main__":
    main()

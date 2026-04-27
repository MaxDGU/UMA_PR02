from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from BBT.src import distill_train
from BBT.src import human_finetune


def test_distill_train_command_contains_scratch_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    holder = {"cmd": []}

    def _fake_run(cmd, dry_run):
        holder["cmd"] = cmd

    monkeypatch.setattr(distill_train, "run_command", _fake_run)

    input_csv = tmp_path / "translated.csv.gz"
    input_csv.write_text("instruction_nl,response_nl\n", encoding="utf-8")

    cfg = {
        "experiment": {"seed": 42},
        "paths": {"sp2013_csv": "results/UMA_replication/sp2013.csv"},
        "external_inputs": {"translated_csv": str(input_csv)},
        "subpipelines": {
            "distill_train": {
                "output_dir": "{run_dir}/03_distill_train",
                "script": "results/transformer_replication/train_transformer_hf.py",
                "model_name": "HuggingFaceTB/SmolLM2-135M",
                "init_from_scratch": True,
                "eval_sp2013": False,
                "eval_id_final_answer": False,
                "use_lora": False,
                "local_files_only": True,
                "sp2013_use_param_grid": True,
            }
        },
    }

    distill_train.run_subpipeline(
        config=cfg,
        run_dir=tmp_path,
        python_exec="python",
        dry_run=True,
    )

    cmd = holder["cmd"]
    assert "--model_name" in cmd
    assert "--init_from_scratch" in cmd
    assert "--no-eval_sp2013" in cmd


def test_human_finetune_defaults_use_migrated_entrypoint_and_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    holder = {"cmd": []}

    def _fake_run(cmd, dry_run):
        holder["cmd"] = cmd

    monkeypatch.setattr(human_finetune, "run_command", _fake_run)

    model_dir = tmp_path / "distilled_model"
    model_dir.mkdir()

    cfg = {
        "experiment": {"seed": 42},
        "paths": {"sp2013_csv": "results/UMA_replication/sp2013.csv"},
        "subpipelines": {
            "human_finetune": {
                "output_dir": "{run_dir}/05_human_finetune",
                "checkpoint_subdir": "best",
                "eval_sp2013": False,
                "eval_loaded_model": False,
            }
        },
    }

    human_finetune.run_subpipeline(
        config=cfg,
        run_dir=tmp_path,
        python_exec="python",
        dry_run=True,
        upstream_primary_output=model_dir,
    )

    cmd = holder["cmd"]
    assert cmd[1].endswith("human_ft/finetune_humandata.py")
    assert "data/human_ft/data_train_nlp.csv" in cmd[cmd.index("--train_csv") + 1]
    assert "data/human_ft/data_val_nlp.csv" in cmd[cmd.index("--val_csv") + 1]
    assert "--no-eval_sp2013" in cmd
    assert "--no-eval_loaded_model" in cmd

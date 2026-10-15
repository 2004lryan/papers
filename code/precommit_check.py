"""
precommit_check.py: 一键代码质量自检（ruff + mypy --strict + pytest）。

运行方式:
    cd <project root>
    python 02code/precommit_check.py

说明:
    仅对已完成规范迁移、纳入门控的文件集执行；随脚本迁移推进，
    把新迁移完成的脚本加入 COMPLIANT_FILES。未迁移的遗留脚本暂不纳入门控。
"""

from __future__ import annotations

import subprocess
import sys

# 已完成合规迁移、纳入三件套门控的文件（随迁移扩充）
COMPLIANT_FILES: list[str] = [
    "02code/export_utils.py",
    "02code/protonet.py",
    "02code/01preprocess.py",
    "02code/03_eda.py",
    "02code/04_baselines.py",
    "02code/05_meta_train.py",
    "02code/06_loeo_kshot.py",
    "02code/07_full_combo_eval.py",
    "02code/08_loeo_element.py",
    "02code/09_sd_confusion_analysis.py",
    "02code/10_loeo_year_foldwise.py",
    "02code/11_kshot_baselines.py",
    "02code/12_loeo_ablate_alpha.py",
    "02code/13_heterogeneity_ablation.py",
    "02code/14_significance_test.py",
    "02code/15_face_ablation_50reps.py",
    "02code/16_protonet_k_large.py",
    "02code/17_alpha_sensitivity.py",
    "02code/18_clean_loeo_year.py",
    "02code/19_perclass_f1_foldwise.py",
    "02code/20_unified_singleface_kshot.py",
    "02code/21_strict_source_zscore.py",
    "02code/22_cnn_resnet_baseline.py",
    "02code/23_reproducibility_multiseed.py",
    "02code/24negctrl.py",
    "02code/25external.py",
    "02code/26calibration.py",
    "02code/27interpret.py",
    "02code/28aggregate_multiseed.py",
    "02code/99project_summary.py",
    "02code/crossdataset_lib.py",
    "02code/encoders.py",
    "02code/64_leakage_controls.py",
    "02code/67saa_figures.py",
    "02code/68si_tables.py",
    "02code/72_kscan_arms.py",
    "02code/74_multi_encoder_leakage.py",
    "02code/75_leakage_generality_summary.py",
    "02code/76_target_tuning_leakage.py",
    "02code/77_generality_tables.py",
    "02code/precommit_check.py",
    "02code/tests",
]


def _run(name: str, cmd: list[str]) -> bool:
    print(f"\n──────── {name} ────────")
    result = subprocess.run(cmd, check=False)
    ok = result.returncode == 0
    print(f"{'✅' if ok else '❌'} {name}")
    return ok


def main() -> int:
    checks: list[tuple[str, list[str]]] = [
        ("ruff check", [sys.executable, "-m", "ruff", "check", *COMPLIANT_FILES]),
        ("ruff format --check", [sys.executable, "-m", "ruff", "format", "--check", *COMPLIANT_FILES]),
        ("mypy --strict", [sys.executable, "-m", "mypy", *COMPLIANT_FILES]),
        ("pytest", [sys.executable, "-m", "pytest", "02code/tests", "-q"]),
    ]
    results = [_run(name, cmd) for name, cmd in checks]
    passed = all(results)
    print("\n" + "=" * 52)
    print(f"总计：{sum(results)}/{len(results)} 项通过 —— {'全部通过 ✅' if passed else '存在失败 ❌'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

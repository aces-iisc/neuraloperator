"""Which bench produces which table, and in what order to run them.

One entry per paper table.  A bench that emits several tables from a single sweep appears once
with all of them listed: splitting it into a file per table would re-run the same measurement
several times, which is the thing this package exists to avoid.

`needs` records what an entry consumes, so run_all can order them and skip what cannot run:
  data     one of the dataset files must be present
  weights  trained weights from train_all
  traces   xprof captures written by an earlier entry (parsed, not re-measured)
"""

# Each docs_* bench writes wherever its own env var says.  CSV_ENV routes them into REPRO_OUT
# so one run leaves one directory of results rather than scattering them across /tmp.
CSV_ENV = {
    "docs_bench2": "DOCS_B2_CSV",
    "docs_kappa": "DOCS_KAPPA_CSV",
    "docs_karatsuba": "DOCS_KAR_CSV",
    "docs_gemm": "DOCS_GEMM_CSV",
    "docs_pack": "DOCS_PACK_CSV",
    "docs_cbk": "DOCS_CBK_CSV",
    "docs_modes": "DOCS_MODES_OUT",
    "docs_bstar": "DOCS_BSTAR_CSV",
    "docs_tuner": "DOCS_TUNER_CSV",
    "docs_final": "DOCS_FINAL_CSV",
    "docs_fact": "DOCS_FACT_CSV",
    "docs_modesweep2d": "SW_CSV",
}


# (module, [tables it fills], needs, one-line note)
ENTRIES = [
    (".train_all", [], ("data",),
     "trains one model per dataset on H1 and caches the weights"),

    (".tab_ladder", ["tab:ladder"], ("data", "weights"),
     "every Section 5 optimisation, each measured against the previous one"),

    ("..experimental.bench.docs_bench2",
     ["tab:engines-1d", "tab:engines-bf16", "tab:partial-1d",
      "tab:coupled-partial", "tab:fno-inference"], ("traces",),
     "one 1-D engine sweep; the four tables above read the same run"),

    ("..experimental.bench.parse_ops",
     ["tab:enginebreakdown", "tab:enginetrade", "tab:ops256"], ("traces", "tensorflow"),
     "per-operation device time from docs_bench2's traces; needs TensorFlow to read xplane, "
     "so it runs off-box after the traces come back"),

    ("..experimental.bench.docs_kappa", ["tab:leafsmall"], (),
     "leaf size against splitting all the way"),

    ("..experimental.bench.docs_gemm", ["tab:blockstacked"], (),
     "stacking and blocking the channel GEMM"),

    ("..experimental.bench.docs_pack", ["tab:nopack"], (),
     "coupled and reduced row packings against the shipped one"),

    ("..experimental.bench.docs_karatsuba", ["tab:karatsuba"], (),
     "three real matmuls against four"),

    ("..experimental.bench.docs_cbk", ["tab:cbk"], (),
     "folding the stage transpose into the matmul"),

    ("..experimental.bench.docs_modes", ["tab:modesweep", "tab:modeenergy"], ("data",),
     "n_modes against accuracy and cost"),

    ("..experimental.bench.docs_bstar", ["tab:bstar"], ("data", "weights"),
     "inference at the training batch and at B*"),

    ("..experimental.bench.docs_tuner", ["tab:tuner"], (),
     "the autotuner's search, on random data"),

    ("..experimental.bench.docs_final", ["tab:results"], ("data", "weights"),
     "baseline against the tuned configuration, per dataset"),

    ("..experimental.bench.docs_fact", ["tab:factorized"], ("data",),
     "F-FNO and TFNO against the dense filter"),

    # 772 runs, roughly two hours.  Fills a figure rather than a table, so it is not part of
    # a default reproduction; ask for it by name.
    ("..experimental.bench.docs_modesweep2d", ["fig:modesweep2d"], ("slow",),
     "row methods over the n_modes x N grid"),
]

# tables with no measurement behind them: settings, algebra, hardware, or another machine
STATIC = {
    "tab:config-paper": "settings quoted from the FNO paper",
    "tab:config-repo": "settings from the shipped configuration files",
    "tab:notation": "symbol definitions",
    "tab:shapes": "algebraic shapes, not measured",
    "tab:modesetup": "training hyperparameters held fixed",
    "tab:tpu-v5e": "vendor specification",
    "tab:torch": "measured on GPU",
    "tab:turbofno": "measured on GPU",
}


def covered():
    return {t for _, tables, _, _ in ENTRIES for t in tables}

"""One command for a full reproduction, on or off the TPU.

  python -m tpu.repro.launch                 everything
  python -m tpu.repro.launch ladder results  only those entries
  python -m tpu.repro.launch --keep          leave the VM up afterwards

`Launcher` runs the benches in place when a TPU is already visible.  Otherwise it drives a
`TpuVm`, which is a context manager so the node is deleted on the way out however the block
ends: a spot VM left running bills until someone notices.
"""
import os
import shlex
import subprocess
import sys
import time


class Shell:
    """Thin wrapper over subprocess that echoes what it runs."""

    def __init__(self, verbose=True):
        self.verbose = verbose

    def run(self, cmd, timeout=None, check=False, quiet=False):
        if self.verbose and not quiet:
            shown = cmd if isinstance(cmd, str) else " ".join(cmd)
            print(f"  $ {shown[:140]}", flush=True)
        result = subprocess.run(cmd if isinstance(cmd, list) else shlex.split(cmd),
                                capture_output=True, text=True, timeout=timeout)
        if check and result.returncode:
            raise RuntimeError(f"failed ({result.returncode}): {result.stderr[-400:]}")
        return result


class TpuVm:
    """A spot Cloud TPU, held for the duration of a `with` block.

    Creation walks a zone pool because spot capacity is refused rather than queued, so the
    first zone with room wins.  Exiting deletes the node unless `keep` was asked for.
    """

    PROJECT = os.environ.get("TPU_PROJECT", "fno-tpu")
    NAME = os.environ.get("TPU_NAME", "tpu-aces-ani")
    ACCEL = os.environ.get("TPU_ACCEL", "v5litepod-1")
    VERSION = os.environ.get("TPU_VERSION", "v2-alpha-tpuv5-lite")
    ZONES = os.environ.get("TPU_ZONE_POOL",
                           "us-west1-c us-west4-a us-central1-a us-east1-c us-east5-b "
                           "asia-east1-c europe-west4-b").split()

    def __init__(self, keep=False, shell=None):
        self.keep = keep
        self.sh = shell or Shell()
        self.zone = None

    # -- lifecycle ------------------------------------------------------------------
    def __enter__(self):
        self.zone = self._create()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.zone is None:
            return False
        if self.keep:
            print(f"\n--keep: {self.NAME} left running in {self.zone}. Delete it with:",
                  flush=True)
            print(f"  gcloud compute tpus tpu-vm delete {self.NAME} "
                  f"--zone={self.zone} --project={self.PROJECT} --quiet", flush=True)
        else:
            print(f"\ndeleting {self.NAME}", flush=True)
            self._gcloud("delete", self.NAME, "--quiet", timeout=900)
            print(f"  deleted (state now {self.state() or 'gone'})", flush=True)
        return False                        # never swallow the original exception

    def _gcloud(self, *args, **kw):
        return self.sh.run(["gcloud", "compute", "tpus", "tpu-vm", *args,
                            f"--zone={self.zone}", f"--project={self.PROJECT}"], **kw)

    def state(self):
        out = self._gcloud("describe", self.NAME, "--format=value(state)",
                           quiet=True, timeout=120).stdout
        return out.strip()

    def _create(self):
        for zone in self.ZONES:
            self.zone = zone
            print(f"  trying {zone}", flush=True)
            self._gcloud("create", self.NAME, f"--accelerator-type={self.ACCEL}",
                         f"--version={self.VERSION}", "--spot", timeout=900)
            if self.state() == "READY":
                print(f"  READY in {zone}", flush=True)
                return zone
        self.zone = None
        raise RuntimeError(f"no spot capacity for {self.ACCEL} in: {' '.join(self.ZONES)}")

    # -- remote operations ----------------------------------------------------------
    def ssh(self, command, timeout=None):
        return self._gcloud("ssh", self.NAME, "--worker=0", f"--command={command}",
                            timeout=timeout)

    def put(self, local, remote, timeout=1800):
        return self._gcloud("scp", local, f"{self.NAME}:{remote}", "--worker=0",
                            timeout=timeout)

    def get(self, remote, local, timeout=1800):
        return self._gcloud("scp", f"{self.NAME}:{remote}", local, "--worker=0",
                            timeout=timeout)


class Launcher:
    """Runs the reproduction, bringing a TPU up only if this machine has none."""

    HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RESULTS = os.path.join(HERE, "tpu", "results", "repro")
    LOCAL_DATA = os.path.expanduser(os.environ.get("DOCS_DATA", "~/data"))
    DATA_FILES = ["darcy_train_32.pt", "darcy_test_32.pt",
                  "darcy_train_64.pt", "darcy_test_64.pt",
                  "nsforcing_train_128.pt", "nsforcing_test_128.pt",
                  "burgers_train_16.pt", "burgers_test_16.pt"]
    POLL_SECONDS = 90
    POLL_LIMIT = 400

    def __init__(self, entries=(), keep=False):
        self.entries = list(entries)
        self.keep = keep
        self.sh = Shell()

    # -- where are we ---------------------------------------------------------------
    @staticmethod
    def tpu_visible():
        try:
            import jax
            return any(d.platform == "tpu" for d in jax.devices())
        except Exception:
            return False

    def run(self):
        if self.tpu_visible():
            print("TPU visible here; running in place\n", flush=True)
            from .run_all import main as run_main
            return run_main(["run_all"] + self.entries)
        print(f"no local TPU; bringing one up ({TpuVm.ACCEL}, spot)\n", flush=True)
        with TpuVm(keep=self.keep, shell=self.sh) as vm:
            self.upload(vm)
            self.bootstrap(vm)
            finished = self.run_remote(vm)
            self.collect(vm)
            print("\nREPRO_DONE" if finished else "\nREPRO_TIMED_OUT", flush=True)
            print(f"results in {self.RESULTS}", flush=True)

    # -- stages ---------------------------------------------------------------------
    WEIGHTS_LOCAL = os.path.join(HERE, "tpu", "results", "repro", "weights")

    def upload(self, vm):
        archive = "/tmp/repro_tree.tar.gz"
        self.sh.run(["tar", "czf", archive, "-C", self.HERE, "--exclude=__pycache__",
                     "--exclude=.venv", "--exclude=tpu/results", "--exclude=tpu/traces",
                     "tpu"], check=True, quiet=True)
        vm.put(archive, "~/")
        vm.ssh("rm -rf ~/nop && mkdir -p ~/nop ~/data && "
               "tar xzf ~/repro_tree.tar.gz -C ~/nop", timeout=600)
        # Weights already trained are shipped back up so a rerun infers rather than retrains.
        # train_all skips any case/engine whose file is already present.
        if os.path.isdir(self.WEIGHTS_LOCAL):
            kept = [f for f in sorted(os.listdir(self.WEIGHTS_LOCAL)) if f.endswith(".npz")]
            if kept:
                vm.ssh("mkdir -p ~/repro_out/weights", timeout=300)
                for name in kept:
                    print(f"  weights {name}", flush=True)
                    vm.put(os.path.join(self.WEIGHTS_LOCAL, name), "~/repro_out/weights/",
                           timeout=1800)
        for name in self.DATA_FILES:
            path = os.path.join(self.LOCAL_DATA, name)
            if os.path.exists(path):
                print(f"  data {name}", flush=True)
                vm.put(path, "~/data/", timeout=3600)
            else:
                print(f"  data {name}: absent locally, tables needing it will skip", flush=True)

    def bootstrap(self, vm):
        vm.ssh("curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; "
               "export PATH=$HOME/.local/bin:$PATH; "
               "uv venv ~/nopenv --python 3.12 >/dev/null 2>&1 && "
               "uv pip install --python ~/nopenv --quiet --upgrade 'jax[tpu]' numpy torch "
               "--index-strategy unsafe-best-match", timeout=2400)
        out = vm.ssh("~/nopenv/bin/python -c "
                     "'import jax; print(jax.__version__, jax.devices())'", timeout=600).stdout
        print(f"  {out.strip().splitlines()[-1] if out.strip() else '(no output)'}", flush=True)

    def run_remote(self, vm):
        """Launch detached, then poll, pulling CSVs as they appear so a crash loses nothing."""
        os.makedirs(self.RESULTS, exist_ok=True)
        vm.ssh("export PATH=$HOME/.local/bin:$PATH; cd ~/nop && rm -f /tmp/repro.log && "
               "setsid nohup env DOCS_DATA=$HOME/data REPRO_OUT=$HOME/repro_out "
               "PYTHONPATH=$HOME/nop ~/nopenv/bin/python -m tpu.repro.run_all "
               f"{' '.join(self.entries)} > /tmp/repro.log 2>&1 < /dev/null & disown",
               timeout=300)
        print("  launched; polling", flush=True)
        for i in range(self.POLL_LIMIT):
            tail = vm.ssh("tail -3 /tmp/repro.log 2>/dev/null", timeout=200).stdout.splitlines()
            if tail:
                print(f"  [{i}] {tail[-1][:120]}", flush=True)
            vm.get("~/repro_out/*.csv", self.RESULTS + "/", timeout=400)
            if any("REPRO_ALL_DONE" in line for line in tail):
                return True
            time.sleep(self.POLL_SECONDS)
        return False

    def collect(self, vm):
        for remote, label in (("repro_out/weights", "weights"), ("repro_out/traces", "traces")):
            vm.ssh(f"cd ~ && tar czf /tmp/{label}.tar.gz {remote} 2>/dev/null", timeout=600)
            vm.get(f"/tmp/{label}.tar.gz", f"{self.RESULTS}/")
            print(f"  pulled {label}", flush=True)


def main(argv):
    Launcher(entries=[a for a in argv[1:] if not a.startswith("--")],
             keep="--keep" in argv).run()


if __name__ == "__main__":
    main(sys.argv)

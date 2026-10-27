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

    # `gcloud ... ssh` re-propagates a key on every call and fails as a unit when that step
    # breaks, even though the node is up and already holds the key.  These are the pieces
    # needed to talk to it directly instead.
    SSH_USER = os.environ.get("TPU_SSH_USER") or os.environ.get("USER") or "root"
    SSH_KEY = os.path.expanduser(os.environ.get("TPU_SSH_KEY", "~/.ssh/google_compute_engine"))
    # -n matters: none of these commands read stdin, and leaving it attached is one more
    # channel ssh would wait on before returning.
    SSH_OPTS = ["-n", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ConnectTimeout=20", "-o", "BatchMode=yes"]

    SCP_OPTS = [o for o in SSH_OPTS if o != "-n"]

    def __init__(self, keep=False, shell=None):
        self.keep = keep
        self.sh = shell or Shell()
        self.zone = None
        self.direct_ip = None          # set once gcloud ssh is found to be broken

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
            # Spot is the default because these runs are short and restartable, but a
            # generation can be unavailable across every zone for hours; TPU_SPOT=0 pays for
            # an on-demand node rather than leaving a result unmeasured.
            flags = ["--spot"] if os.environ.get("TPU_SPOT", "1") != "0" else []
            self._gcloud("create", self.NAME, f"--accelerator-type={self.ACCEL}",
                         f"--version={self.VERSION}", *flags, timeout=900)
            if self.state() == "READY":
                print(f"  READY in {zone}", flush=True)
                return zone
        self.zone = None
        raise RuntimeError(f"no spot capacity for {self.ACCEL} in: {' '.join(self.ZONES)}")

    # -- remote operations ----------------------------------------------------------
    def external_ip(self):
        out = self._gcloud("describe", self.NAME, "--format=value("
                           "networkEndpoints[0].accessConfig.externalIp)",
                           quiet=True, timeout=120).stdout.strip()
        return out.splitlines()[-1].strip() if out else ""

    def ssh(self, command, timeout=None):
        if self.direct_ip:
            return self.sh.run(["ssh", "-i", self.SSH_KEY, *self.SSH_OPTS,
                                f"{self.SSH_USER}@{self.direct_ip}", command], timeout=timeout)
        return self._gcloud("ssh", self.NAME, "--worker=0", f"--command={command}",
                            timeout=timeout)

    def await_ssh(self, attempts=20, pause=20):
        """Block until the node answers.  A READY node is not necessarily a reachable one.

        Key propagation can fail for a while after creation, and `ssh` reports that the same
        way it reports a command with no output, so a caller that does not check here cannot
        tell a broken node from a quiet one.
        """
        for attempt in range(attempts):
            if "tpu_ssh_ok" in self.ssh("echo tpu_ssh_ok", timeout=200).stdout:
                where = f" (direct to {self.direct_ip})" if self.direct_ip else ""
                print(f"  ssh answering{where}", flush=True)
                return True
            # Two attempts is enough to tell a node still booting from a key-propagation
            # failure, which no amount of further waiting fixes.  The node itself is usually
            # reachable at that point, so try it directly before giving up on the run.
            if attempt == 1 and not self.direct_ip:
                ip = self.external_ip()
                if ip:
                    print(f"  gcloud ssh is failing; trying {ip} directly", flush=True)
                    self.direct_ip = ip
                    continue
            print(f"  ssh not answering yet ({attempt + 1}/{attempts})", flush=True)
            time.sleep(pause)
        raise RuntimeError(f"{self.NAME} is READY but never answered ssh; nothing was run")

    def put(self, local, remote, timeout=1800):
        if self.direct_ip:
            return self.sh.run(["scp", "-i", self.SSH_KEY, *self.SCP_OPTS, local,
                                f"{self.SSH_USER}@{self.direct_ip}:{remote}"], timeout=timeout)
        return self._gcloud("scp", local, f"{self.NAME}:{remote}", "--worker=0",
                            timeout=timeout)

    def get(self, remote, local, timeout=1800):
        if self.direct_ip:
            # `remote` may be a glob.  Quoting it stops the local shell expanding it, while
            # scp still hands it to a shell on the far side, which is where it should expand.
            target = shlex.quote(f"{self.SSH_USER}@{self.direct_ip}:{remote}")
            return self.sh.run(["bash", "-c", " ".join(
                ["scp", "-i", shlex.quote(self.SSH_KEY), *self.SCP_OPTS,
                 target, shlex.quote(local)])], timeout=timeout)
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

    def needs(self, what):
        """Whether any chosen entry consumes `what`, so staging it is not dead weight.

        A bench on synthetic inputs wants neither the datasets nor the trained weights, and
        shipping them anyway is gigabytes of upload against a billing clock. With no entries
        named the run is everything, so everything is staged.
        """
        from .registry import ENTRIES
        if not self.entries:
            return True
        chosen = [e for e in ENTRIES
                  if any(a in e[0] or any(a in t for t in e[1]) for a in self.entries)]
        return any(what in e[2] for e in chosen) or not chosen

    def upload(self, vm):
        archive = "/tmp/repro_tree.tar.gz"
        self.sh.run(["tar", "czf", archive, "-C", self.HERE, "--exclude=__pycache__",
                     "--exclude=.venv", "--exclude=tpu/results", "--exclude=tpu/traces",
                     "tpu"], check=True, quiet=True)
        vm.await_ssh()
        vm.put(archive, "~/")
        vm.ssh("rm -rf ~/nop && mkdir -p ~/nop ~/data && "
               "tar xzf ~/repro_tree.tar.gz -C ~/nop", timeout=600)
        # Weights already trained are shipped back up so a rerun infers rather than retrains.
        # train_all skips any case/engine whose file is already present.
        if os.path.isdir(self.WEIGHTS_LOCAL) and self.needs("weights"):
            kept = [f for f in sorted(os.listdir(self.WEIGHTS_LOCAL)) if f.endswith(".npz")]
            if kept:
                vm.ssh("mkdir -p ~/repro_out/weights", timeout=300)
                for name in kept:
                    print(f"  weights {name}", flush=True)
                    vm.put(os.path.join(self.WEIGHTS_LOCAL, name), "~/repro_out/weights/",
                           timeout=1800)
        if not self.needs("data"):
            print("  no chosen entry reads a dataset; skipping the data upload", flush=True)
            return
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
        # A bench configured by the environment is configured on the node, not here, so the
        # settings have to travel with the job.  Without this a sweep asked for locally runs
        # remotely with its defaults and reports a different measurement than the one requested.
        forwarded = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(os.environ.items())
                             if k.startswith(("REPRO_", "TUNER_", "DOCS_", "FACT_"))
                             and k not in ("DOCS_DATA", "REPRO_OUT"))
        if forwarded:
            print(f"  forwarding {forwarded}", flush=True)
        started = vm.ssh(
            "export PATH=$HOME/.local/bin:$PATH; cd ~/nop && rm -f /tmp/repro.log && "
            "{ setsid nohup env DOCS_DATA=$HOME/data REPRO_OUT=$HOME/repro_out "
            f"{forwarded} "
            "PYTHONPATH=$HOME/nop ~/nopenv/bin/python -m tpu.repro.run_all "
            f"{' '.join(self.entries)} > /tmp/repro.log 2>&1 < /dev/null & "
            "} >/dev/null 2>&1 && echo LAUNCH_OK", timeout=300)
        if "LAUNCH_OK" not in started.stdout:
            raise RuntimeError(f"the remote launch did not confirm: "
                               f"{(started.stderr or started.stdout)[-300:]}")
        print("  launched; polling", flush=True)
        silent = 0
        for i in range(self.POLL_LIMIT):
            # A spot node can be reclaimed mid-run.  Its ssh then fails like any other quiet
            # poll, so without this the loop waits out its whole limit against a dead host.
            state = vm.state()
            if state and state not in ("READY", "CREATING", "STARTING"):
                raise RuntimeError(f"{vm.NAME} is {state} -- the node went away mid-run "
                                   f"(spot preemption); rerun with TPU_SPOT=0 for on-demand")
            tail = vm.ssh("tail -3 /tmp/repro.log 2>/dev/null", timeout=200).stdout.splitlines()
            if tail:
                silent = 0
                print(f"  [{i}] {tail[-1][:120]}", flush=True)
            else:
                # an empty remote log early on is the job still importing; an empty one for
                # several minutes means it never started, which is worth saying rather than
                # polling quietly for the rest of the limit
                silent += 1
                print(f"  [{i}] remote log still empty", flush=True)
                if silent * self.POLL_SECONDS >= 300:
                    raise RuntimeError("remote log empty for five minutes: the job never "
                                       "started (check ssh and the remote environment)")
            vm.get("~/repro_out/*.csv", self.RESULTS + "/", timeout=400)
            if any("REPRO_ALL_DONE" in line for line in tail):
                return True
            time.sleep(self.POLL_SECONDS)
        return False

    def collect(self, vm):
        wanted = []
        # Weights go UP from here before the run, so pulling them back only means anything when
        # something trained new ones -- and the tarball is half a gigabyte.
        if not self.entries or any("train" in e for e in self.entries):
            wanted.append(("repro_out/weights", "weights"))
        # A bench that reads its own traces on the node has already put the numbers in its CSV;
        # the archive is then only worth its minutes if the traces will be re-read off-box.
        if os.environ.get("REPRO_SKIP_TRACES") != "1":
            wanted.append(("repro_out/traces", "traces"))
        for remote, label in wanted:
            vm.ssh(f"cd ~ && tar czf /tmp/{label}.tar.gz {remote} 2>/dev/null", timeout=600)
            vm.get(f"/tmp/{label}.tar.gz", f"{self.RESULTS}/")
            print(f"  pulled {label}", flush=True)


def main(argv):
    Launcher(entries=[a for a in argv[1:] if not a.startswith("--")],
             keep="--keep" in argv).run()


if __name__ == "__main__":
    main(sys.argv)

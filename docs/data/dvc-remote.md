# The DVC remote

The calibration artifacts — 70,176 hourly load rows, four years of island-hour meteorology, the
reconciled ledger, the scenario library — are DVC-tracked and git-ignored. Without a remote they
exist on exactly one laptop, `dvc pull` returns nothing, and everyone downstream is blocked on a
file transfer.

The whole cache is **9.1 MB across 11 objects**. This is a reachability problem, not a storage one.

## Why it is a mount and not a `dvc` cloud backend

DVC ships backends for S3, GCS, Azure Blob, SSH, WebDAV, GDrive and HTTP. **There is no OneDrive
backend.** The nearest fit is WebDAV against the SharePoint endpoint, which needs legacy basic
auth — disabled tenant-wide by Microsoft in 2022 and not re-enabled on `sliit.lk`.

rclone does have a first-class OneDrive backend with modern OAuth. So the remote is a plain local
directory that rclone mounts onto the shared folder, and DVC never learns the difference.

```
dvc  ->  .dvc/onedrive-mount  ->  rclone  ->  OneDrive:MetaCore/dvc-store
                                              (metacore@sliit.lk, private group)
```

The remote path is **repo-relative** (`url = onedrive-mount`, resolved against `.dvc/`), so it is
identical on all four machines and nobody needs a per-machine override. It is git-ignored via
`.dvc/.gitignore`.

## One-time setup, per person

rclone's OneDrive setup opens a browser to sign in to `metacore@sliit.lk`. Do this yourself — it
is an authentication step.

```bash
# 1. install rclone (once per machine)
sudo apt install rclone            # or: curl https://rclone.org/install.sh | sudo bash

# 2. authorise OneDrive. name it `onedrive`, type `onedrive`, accept the defaults,
#    and pick the shared MetaCore folder when prompted for the drive.
rclone config

# 3. mount it where DVC expects to find it
mkdir -p .dvc/onedrive-mount
rclone mount onedrive:MetaCore/dvc-store .dvc/onedrive-mount --daemon
```

Add the mount to your login items if you want it to survive a reboot; nothing in the repo depends
on it being up except `artifacts:pull` and `artifacts:push`, which fail with a clear message
rather than a DVC stack trace when it is not.

## Daily use

```bash
task artifacts:pull     # fetch artifacts someone else rebuilt
task artifacts:push     # publish artifacts you rebuilt
task data               # rebuild them locally from source instead
```

`task artifacts:pull` is the fast path and the one to use unless you are changing a stage. It
avoids re-running the pipeline and, more importantly, avoids needing the CEB workbook.

## What is and is not published

`dvc push` ships tracked **outputs** only:

| Published | Not published |
|---|---|
| `processed/ceb_generation_tidy.csv` | `external/ceb_jaffna/Generation_2024_2025.xlsx` |
| `processed/island_load_hourly.csv`, `load_parameters.json` | anything else in `external/` |
| `processed/events.csv`, `scenario_library.json` | |
| `raw/nasa_power/` | |

The CEB workbook is a stage *dependency*, not an output, so it stays put. That is deliberate:
`data/README.md` promises nothing in `external/` becomes a build requirement, and the workbook is
state-entity data provided for calibration. The reconciled tidy table derived from it is what the
team shares.

A clean clone with no workbook and no remote access can still run the full pipeline — see the
synthetic fallback in `docs/adr/0004-two-ingestion-paths.md`.

## If the mount is flaky

rclone caches aggressively and OneDrive throttles. If `dvc push` stalls, the fallback is a plain
local remote plus an explicit sync:

```bash
dvc remote add --local staging ~/.metacore-dvc-store
dvc push -r staging
rclone sync ~/.metacore-dvc-store onedrive:MetaCore/dvc-store
```

`--local` writes to `.dvc/config.local`, which is git-ignored, so it does not change the committed
remote for anyone else.

"""Utilities for preparing Manchester nuclear Gazebo assets for Isaac Sim."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import urllib.request
import zipfile

from sim.blender_environment import resolve_blender_executable

ROOT = Path(__file__).resolve().parents[2]
FIGSHARE_ARTICLE_URL = (
    "https://figshare.manchester.ac.uk/articles/software/"
    "3D_Simulation_Assets_for_Nuclear_Environments_Gazebo_Format_/25224974"
)
FIGSHARE_API_URL = "https://api.figshare.com/v2/articles/25224974"
DEFAULT_ASSET_ROOT = ROOT / "data" / "manchester_nuclear_assets"
DEFAULT_CONVERTER_SCRIPT = ROOT / "scripts" / "convert_manchester_sdf_to_usd.py"


@dataclass(frozen=True)
class ManchesterDatasetFile:
    """Describe one downloadable Manchester Figshare file."""

    key: str
    filename: str
    file_id: int
    size_bytes: int
    md5: str

    @property
    def download_url(self) -> str:
        """Return the public Figshare file download URL."""
        return f"https://ndownloader.figshare.com/files/{self.file_id}"

    @property
    def slug(self) -> str:
        """Return a filesystem-friendly lowercase asset slug."""
        return asset_slug(self.key)


MANCHESTER_DATASET_FILES: tuple[ManchesterDatasetFile, ...] = (
    ManchesterDatasetFile(
        key="500L_Drum",
        filename="500L_Drum.zip",
        file_id=44556491,
        size_bytes=15142843,
        md5="4efe7f3cc6d61030dc0ecb8acbb5a9a9",
    ),
    ManchesterDatasetFile(
        key="500L_Drum_Store",
        filename="500L_Drum_Store.zip",
        file_id=44556494,
        size_bytes=429980069,
        md5="4f70bc1c1b8a88486d0120fdd4d19cc6",
    ),
    ManchesterDatasetFile(
        key="Barrel",
        filename="Barrel.zip",
        file_id=44556497,
        size_bytes=7828956,
        md5="0e69dfa862f679a28a773722179dcb93",
    ),
    ManchesterDatasetFile(
        key="Barrier",
        filename="Barrier.zip",
        file_id=44556500,
        size_bytes=23441752,
        md5="8232fb9359864352c9253eb833586a8e",
    ),
    ManchesterDatasetFile(
        key="Crate",
        filename="Crate.zip",
        file_id=44556503,
        size_bytes=11556660,
        md5="db437741a04cd65e7e1d41e829de49e1",
    ),
    ManchesterDatasetFile(
        key="Drum_Store",
        filename="Drum_Store.zip",
        file_id=44556530,
        size_bytes=555414778,
        md5="33cf1a5857339b204377ca56d86e8087",
    ),
    ManchesterDatasetFile(
        key="Fire_Extinguisher",
        filename="Fire_Extinguisher.zip",
        file_id=44556533,
        size_bytes=21526598,
        md5="e484eb2d3786ab2baa36e25465ccde5c",
    ),
    ManchesterDatasetFile(
        key="Intermediate_Bulk_Container",
        filename="Intermediate_Bulk_Container.zip",
        file_id=44556539,
        size_bytes=18323866,
        md5="fd0df03bc1a0d2cba1a5b48d88638742",
    ),
    ManchesterDatasetFile(
        key="Pallet",
        filename="Pallet.zip",
        file_id=44556542,
        size_bytes=20905397,
        md5="6ba8b3067d989d75d4b3860d34a24f1b",
    ),
    ManchesterDatasetFile(
        key="Pallet_Trolley",
        filename="Pallet_Trolley.zip",
        file_id=44556545,
        size_bytes=32180755,
        md5="ecbe25c8d279e97dff2d759eb9ed9bfa",
    ),
    ManchesterDatasetFile(
        key="Industrial_Environment",
        filename="Industrial_Environment.zip",
        file_id=44556551,
        size_bytes=1672484569,
        md5="d573c43c8d34cb1e1805460f76bceb73",
    ),
)


def _normalize_asset_token(value: str) -> str:
    """Normalize an asset name for forgiving user input matching."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def asset_slug(value: str) -> str:
    """Return a stable lowercase slug for a Manchester asset name."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    return slug or "asset"


def available_manchester_assets() -> tuple[ManchesterDatasetFile, ...]:
    """Return the known public Manchester dataset files."""
    return MANCHESTER_DATASET_FILES


def resolve_manchester_asset(name: str) -> ManchesterDatasetFile:
    """Resolve an asset by key, display name, or zip filename."""
    requested = _normalize_asset_token(name)
    for asset in MANCHESTER_DATASET_FILES:
        candidates = {
            asset.key,
            asset.filename,
            asset.filename.removesuffix(".zip"),
            asset.key.replace("_", " "),
        }
        if requested in {_normalize_asset_token(candidate) for candidate in candidates}:
            return asset
    valid = ", ".join(asset.key for asset in MANCHESTER_DATASET_FILES)
    raise ValueError(f"Unknown Manchester asset: {name}. Valid assets: {valid}")


def md5sum(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the hexadecimal MD5 digest for a file."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_md5(path: Path, expected_md5: str) -> bool:
    """Return True when a file exists and matches the expected MD5 digest."""
    return path.exists() and md5sum(path) == expected_md5.lower()


def download_manchester_asset(
    asset: ManchesterDatasetFile | str,
    download_dir: Path = DEFAULT_ASSET_ROOT / "downloads",
    *,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download a Manchester dataset ZIP, reusing a verified local copy."""
    resolved = resolve_manchester_asset(asset) if isinstance(asset, str) else asset
    download_dir = download_dir.expanduser().resolve()
    download_dir.mkdir(parents=True, exist_ok=True)
    output_path = download_dir / resolved.filename
    if verify_md5(output_path, resolved.md5):
        return output_path
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    with urllib.request.urlopen(resolved.download_url) as response:
        with tmp_path.open("wb") as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
    actual_md5 = md5sum(tmp_path)
    if actual_md5 != resolved.md5.lower():
        tmp_path.unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded {resolved.filename} failed MD5 verification: "
            f"expected {resolved.md5}, got {actual_md5}"
        )
    tmp_path.replace(output_path)
    return output_path


def _safe_zip_member_path(target_root: Path, member_name: str) -> Path:
    """Return a safe extraction path for a zip member."""
    destination = (target_root / member_name).resolve()
    root = target_root.resolve()
    if destination != root and root not in destination.parents:
        raise ValueError(f"Unsafe zip member path: {member_name}")
    return destination


def extract_manchester_asset(
    zip_path: Path,
    extract_root: Path = DEFAULT_ASSET_ROOT / "raw",
    *,
    overwrite: bool = False,
) -> Path:
    """Extract a Manchester dataset ZIP and return its top-level directory."""
    zip_path = zip_path.expanduser().resolve()
    extract_root = extract_root.expanduser().resolve()
    extract_root.mkdir(parents=True, exist_ok=True)
    top_level_names: list[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            _safe_zip_member_path(extract_root, member.filename)
            parts = Path(member.filename).parts
            if parts:
                top_level_names.append(parts[0])
        if not top_level_names:
            raise ValueError(f"Zip archive is empty: {zip_path}")
        top_level = top_level_names[0]
        target_dir = extract_root / top_level
        if target_dir.exists() and overwrite:
            shutil.rmtree(target_dir)
        if not target_dir.exists():
            for member in archive.infolist():
                destination = _safe_zip_member_path(extract_root, member.filename)
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    return target_dir.resolve()


def find_manchester_sdf_path(asset_root: Path) -> Path:
    """Return the most likely top-level SDF file under an extracted asset."""
    asset_root = asset_root.expanduser().resolve()
    model_sdf = asset_root / "model.sdf"
    if model_sdf.exists():
        return model_sdf
    sdf_paths = sorted(asset_root.rglob("*.sdf"))
    if not sdf_paths:
        raise FileNotFoundError(f"No SDF file found under {asset_root}")
    preferred = [path for path in sdf_paths if path.name in {"model.sdf", "world.sdf"}]
    return (preferred or sdf_paths)[0]


def convert_manchester_sdf_to_usd(
    sdf_path: Path,
    output_path: Path,
    *,
    asset_root: Path | None = None,
    blender_executable: str | None = None,
    converter_script: Path = DEFAULT_CONVERTER_SCRIPT,
    timeout_s: float = 300.0,
) -> Path:
    """Convert a Manchester SDF scene or model into a USD file via Blender."""
    sdf_path = sdf_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_root = asset_root.expanduser().resolve() if asset_root is not None else sdf_path.parent
    executable = resolve_blender_executable(blender_executable)
    command = [
        executable,
        "--background",
        "--python",
        converter_script.expanduser().resolve().as_posix(),
        "--",
        "--input",
        sdf_path.as_posix(),
        "--output",
        output_path.as_posix(),
        "--model-root",
        model_root.as_posix(),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=float(timeout_s),
    )
    if result.returncode != 0:
        details = "\n".join(
            part for part in ((result.stderr or "").strip(), (result.stdout or "").strip()) if part
        )
        raise RuntimeError(f"Manchester SDF to USD conversion failed:\n{details}")
    if not output_path.exists():
        raise RuntimeError(f"Blender did not create the expected USD file: {output_path}")
    return output_path


def write_isaacsim_config(
    path: Path,
    *,
    usd_path: Path,
    host: str = "127.0.0.1",
    port: int = 5555,
    timeout_s: float = 10.0,
    mode: str = "real",
    headless: bool = True,
) -> Path:
    """Write an Isaac Sim sidecar config for a converted Manchester USD scene."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_usd = usd_path.expanduser().resolve()
    usd_value = os.path.relpath(resolved_usd, start=path.parent).replace(os.sep, "/")
    payload = {
        "backend": "isaacsim",
        "host": str(host),
        "port": int(port),
        "timeout_s": float(timeout_s),
        "mode": str(mode),
        "headless": bool(headless),
        "renderer": "RayTracedLighting",
        "usd_path": usd_value,
        "detector_height_m": 0.5,
        "obstacle_height_m": 2.0,
        "stage_material_rules": [
            {
                "path_prefix": "/World/Environment",
                "material": "concrete",
            }
        ],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path

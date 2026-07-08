#!/usr/bin/env python3
"""Validate the native competition all.usd scene contract."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.expanduser("~/dishwasher_ws/src"))

from dishwasher.scene.native_loader import NativeSceneLoader


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate native all.usd")
    parser.add_argument(
        "--assets",
        default=os.path.expanduser("~/dishwasher_ws/assets/isaac_dishwisher"),
        help="Path to isaac_dishwisher assets directory",
    )
    args = parser.parse_args()

    loader = NativeSceneLoader(args.assets)
    try:
        report = loader.validate()
    except ModuleNotFoundError as exc:
        if exc.name == "pxr":
            print(
                "[FAIL] USD Python bindings are unavailable. "
                "Run inside env_isaaclab.",
                flush=True,
            )
            return 2
        raise

    print("=" * 72)
    print("Native all.usd validation")
    print("=" * 72)
    print(f"path:             {report.path}")
    print(f"exists:           {report.exists}")
    print(f"metersPerUnit:    {report.meters_per_unit}")
    print(f"upAxis:           {report.up_axis}")
    print(f"defaultPrim:      {report.default_prim}")
    print(f"total prims:      {report.total_prims}")
    print(f"cameras:          {len(report.cameras)}")
    print(f"physics scenes:   {len(report.physics_scenes)}")
    print(f"articulationRoot: {len(report.articulation_roots)}")
    print(f"rigid bodies:     {len(report.rigid_bodies)}")
    print(f"collisions:       {len(report.collisions)}")

    if report.missing_prims:
        print("\nMissing required prims:")
        for path in report.missing_prims:
            print(f"  - {path}")

    if report.warnings:
        print("\nWarnings:")
        for warning in report.warnings:
            print(f"  - {warning}")

    print("\nKey cameras:")
    for path in report.cameras:
        print(f"  - {path}")

    print("\nPhysics scenes:")
    for path in report.physics_scenes:
        print(f"  - {path}")

    print("\nArticulation roots:")
    for path in report.articulation_roots:
        print(f"  - {path}")

    print("\nTop type counts:")
    for type_name, count in sorted(
        report.type_counts.items(), key=lambda item: (-item[1], item[0])
    )[:12]:
        print(f"  {type_name}: {count}")

    print("\nresult:", "PASS" if report.ok else "FAIL")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

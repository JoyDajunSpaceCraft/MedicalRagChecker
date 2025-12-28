#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import re
from typing import List, Tuple
"""
用法

先预览（推荐）：

python cleanup_kgfused.py --root /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data --mode dup --dry_run


确认输出没问题后真正删除：

python cleanup_kgfused.py --root /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data --mode dup --yes


如果你想把所有 __kgfused*.json 都删掉（包括正常的那份），用：

python cleanup_kgfused.py --root ... --mode all --yes

"""
def walk_and_collect(root: str, mode: str) -> List[Tuple[str, str]]:
    """
    mode:
      - "all": delete any file that contains "__kgfused" in filename
      - "dup": delete only files where "__kgfused" appears 2+ times (e.g., __kgfused__kgfused.json)
    """
    targets = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".json"):
                continue
            if "__kgfused" not in fn:
                continue

            if mode == "dup":
                if fn.count("__kgfused") < 2:
                    continue

            full = os.path.join(dirpath, fn)
            targets.append((full, fn))
    return targets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory to scan recursively")
    ap.add_argument("--mode", choices=["all", "dup"], default="dup",
                    help="all: delete any *__kgfused*.json; dup: only delete repeated-tag files (default: dup)")
    ap.add_argument("--dry_run", action="store_true", help="Only print files to be deleted")
    ap.add_argument("--yes", action="store_true", help="Actually delete files (required to delete)")
    args = ap.parse_args()

    targets = walk_and_collect(args.root, args.mode)
    targets.sort(key=lambda x: x[0])

    if not targets:
        print("[OK] No matching files found.")
        return

    print(f"[INFO] Found {len(targets)} files to delete (mode={args.mode}).")
    for full, _ in targets:
        print(full)

    if args.dry_run or not args.yes:
        print("\n[DRY] Not deleting anything. Add --yes to delete.")
        return

    deleted = 0
    failed = 0
    for full, _ in targets:
        try:
            os.remove(full)
            deleted += 1
        except Exception as e:
            failed += 1
            print(f"[ERR] Failed to delete: {full} | {e}")

    print(f"\n[DONE] deleted={deleted}, failed={failed}")

if __name__ == "__main__":
    main()

#!/bin/bash
# 换容器：列出并留档"容器本地"需搬运项（这些不在 git 仓库内）。
# 用法: bash scripts/pack_container_local.sh [输出目录]
# 产出: manifest.txt（路径+大小+SHA256）、tilelang_cann9_shim.patch（shim 补丁）、
#       container_local_items.tar.gz（可选打包，默认跳过，大件请自行拷贝）。
set -u
OUT="${1:-$(cd "$(dirname "$0")/.." && pwd)/docs/container_local}"
mkdir -p "$OUT"
MAN="$OUT/manifest.txt"
PATCH="$OUT/tilelang_cann9_shim.patch"

TL=/home/developer/tilelang-mlir-ascend
ITEMS=(
  "$TL"
  "/mnt/workspace/cannlab-migration-kit_20261014.tgz"
  "/mnt/workspace/TileOPs-main"
  "/mnt/workspace/TileOPs-main/.migration_source_map.json"
  "/mnt/workspace/AdaptiveLayerNorm.md"
  "/home/developer/.local/toolchain"
)

{
  echo "# 容器本地项（不在 git repo，需搬到新容器）— 生成于 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# path\tsize\tsha256(若为文件)"
  for p in "${ITEMS[@]}"; do
    if [ -e "$p" ]; then
      sz=$(du -sh "$p" 2>/dev/null | cut -f1)
      if [ -f "$p" ]; then
        sum=$(sha256sum "$p" 2>/dev/null | cut -d' ' -f1)
      else
        sum="(dir)"
      fi
      printf "%s\t%s\t%s\n" "$p" "$sz" "$sum"
    else
      printf "%s\tMISSING\t-\n" "$p"
    fi
  done
} > "$MAN"

# tilelang dev-root 的本地改动（含 CANN9 shim）留成补丁
if [ -d "$TL/.git" ]; then
  git -C "$TL" diff -- tilelang/jit/jit_npu.py > "$PATCH" 2>/dev/null || true
  echo "# tilelang dev-root 全部本地改动（评审用）：" > "$OUT/tilelang_devroot_all_local_changes.patch"
  git -C "$TL" diff >> "$OUT/tilelang_devroot_all_local_changes.patch" 2>/dev/null || true
fi

# 可选：打包小件（大件如 kit/toolchain 请手动拷贝，避免误打包）
echo "manifest  -> $MAN"
echo "shim 补丁 -> $PATCH"
echo "（大件 cannlab-migration-kit_*.tgz / .local/toolchain / tilelang dev-root 建议直接 cp -a 到新容器）"
cat "$MAN"

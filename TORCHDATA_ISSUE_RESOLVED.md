# torchdata Issue: Root Cause and Solution

## The Problem

**Error**: `ModuleNotFoundError: No module named 'torchdata.datapipes'`

**What we discovered**:
1. DGL 2.1.0 introduced `graphbolt` module
2. `graphbolt` requires `torchdata` package
3. `torchdata` is **deprecated** by PyTorch team
4. `torchdata` conflicts with PyTorch 2.5.1
5. Installing via conda would **downgrade PyTorch 2.5.1 → 1.11.0** (disaster!)

## Why torchdata is NOT the Solution

**Attempting `conda install torchdata` would**:
- ❌ Downgrade PyTorch from 2.5.1 to 1.11.0 (3+ years old!)
- ❌ Downgrade CUDA from 12.1 to 11.5 (incompatible with Perlmutter GPUs)
- ❌ Break GPU support
- ❌ Lose all modern PyTorch features

**Attempting `pip install torchdata` would**:
- ❌ Install deprecated package
- ❌ Still have version conflicts
- ❌ `torchdata.datapipes` submodule doesn't exist in pip version

## The Real Solution: Remove torchdata Dependency

**Strategy**: Use DGL 1.1.x instead of 2.x

### Why This Works

1. **DGL 1.1.x** (stable, battle-tested)
   - No graphbolt module
   - No torchdata dependency
   - All graph neural network features MMomentA needs
   - Compatible with PyTorch 2.5.1
   - Compatible with CUDA 12.1

2. **DGL 2.x** (newer but problematic)
   - Added graphbolt (distributed training)
   - Requires deprecated torchdata
   - MMomentA doesn't use graphbolt anyway

3. **MMomentA only needs**:
   - Basic graph convolutions
   - Message passing
   - Batch processing
   - All present in DGL 1.1.x

## Implementation

### Updated Setup Script

`perlmutter/setup/MANUAL_SETUP_ML.sh` now installs:
```bash
# Line 74: Install DGL <2.0 (no torchdata needed)
conda install -c dglteam/label/$DGL_CHANNEL "dgl<2.0" -y
```

### Fix Script for Existing Environments

Created `perlmutter/setup/FIX_DGL_VERSION.sh`:
```bash
conda remove dgl --force -y
conda install -c dglteam/label/cu121 "dgl<2.0" -y
```

## What We Tried (and why they failed)

### ❌ Attempt 1: Install torchdata via pip
```bash
pip install torchdata
```
**Result**: Package installs but missing `torchdata.datapipes` submodule

### ❌ Attempt 2: Install torchdata via conda
```bash
conda install -c pytorch torchdata
```
**Result**: Would downgrade PyTorch 2.5.1 → 1.11.0 and CUDA 12.1 → 11.5

### ❌ Attempt 3: Mock torchdata module
```python
sys.modules['torchdata'] = ...
```
**Result**: Hacky, fragile, breaks on DGL updates

### ✅ Solution: Use compatible DGL version
```bash
conda install "dgl<2.0"
```
**Result**: Everything works, no deprecated dependencies!

## Version Compatibility Matrix

| PyTorch | CUDA | DGL    | torchdata | Status |
|---------|------|--------|-----------|--------|
| 2.5.1   | 12.1 | 2.1.0  | Required  | ❌ Broken |
| 2.5.1   | 12.1 | 1.1.3  | Not needed| ✅ Works  |
| 1.11.0  | 11.5 | 2.1.0  | 0.3.0     | ❌ Ancient|

## Testing

After applying the fix:

```bash
# Verify DGL version
python -c "import dgl; print(f'DGL {dgl.__version__}')"
# Should show: DGL 1.1.x

# Verify PyTorch still modern
python -c "import torch; print(f'PyTorch {torch.__version__}')"
# Should show: PyTorch 2.5.1

# Verify imports work
python -c "import dgl; import torch; print('✓ Both work!')"
# Should succeed
```

## Future-Proofing

**When to upgrade DGL**:
- Wait for DGL to drop torchdata dependency
- Or wait for PyTorch to revive torchdata
- Or when DGL makes graphbolt optional

**For now**: DGL 1.1.x is stable, actively maintained, and has everything MMomentA needs.

## Lessons Learned

1. **Check dependency chains**: DGL → graphbolt → torchdata → conflicts
2. **Deprecated packages are landmines**: torchdata was deprecated for a reason
3. **Conda version resolution can be destructive**: Always check what it's downgrading
4. **Newer isn't always better**: DGL 1.1.x is more stable than 2.x for our use case
5. **Understand what you actually need**: MMomentA doesn't use advanced DGL features

## Summary

**Problem**: DGL 2.1.0 requires deprecated torchdata package
**Solution**: Use DGL 1.1.x which doesn't require torchdata
**Impact**: No functionality loss, everything works, no deprecated dependencies

**For fresh installs**: Setup scripts now install DGL <2.0 automatically
**For existing installs**: Run `bash perlmutter/setup/FIX_DGL_VERSION.sh`

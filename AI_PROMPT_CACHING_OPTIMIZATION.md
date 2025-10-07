# AI Prompt: Optimize Computational Bottleneck with Caching

Use this prompt template when you need an AI to implement caching and optimization for expensive computational steps.

---

## Prompt Template

```
I have a computational bottleneck in my pipeline that I need to optimize. The bottleneck involves expensive calculations that are being recomputed unnecessarily on every run.

### Current Situation

**Script**: `scripts/prepare_dataset.py`
**Bottleneck**: Step 2 - MPFIT charge calculations
**Current Performance**: 20-30 minutes for 159 molecules
**Issues**:
1. No caching - recomputes everything from scratch every run
2. Previously used sequential processing (--n-jobs 1) due to pickle serialization errors
3. No resume capability if process crashes

**How data is currently stored**:
- Output format: HDF5 file (`data/output.h5`)
- Structure: One group per molecule with charges, features, metadata
- No deduplication or incremental updates

**Storage code location**: `MMomentA/data/storage.py`
- `save_dataset_hdf5()` - saves complete dataset
- `load_dataset_hdf5()` - loads complete dataset
- Uses h5py for HDF5 operations

**Batch processing code**: `MMomentA/data/batch.py`
- `BatchProcessor.process()` - runs calculations in parallel using joblib
- Has error handling that converts exceptions to strings (fixes pickle errors)
- Uses temporary directories for each molecule

### Previous Fix Applied

The pickle serialization error was fixed in `batch.py` by wrapping calculator calls in try-except and converting errors to strings:

```python
try:
    result = calculator.compute(molecule)
    if result.success:
        results[method_name] = self._serialize_result(result)
    else:
        error_msg = str(result.error_message)  # Serializable
except Exception as e:
    error_msg = f"{type(e).__name__}: {str(e)}"  # Serializable
```

This means parallel processing should work now.

### Requirements

Create an optimization solution that:

1. **Enables parallel processing** (immediate 4x speedup)
   - Change from --n-jobs 1 to --n-jobs 16
   - Verify the pickle fix is in place

2. **Implements smart caching** (avoid recomputation)
   - Check if output file exists
   - Load existing results
   - Identify which molecules are already computed (use SMILES for deduplication)
   - Only compute new/missing molecules
   - Merge cached + new results
   - Save atomically (temp file + rename)

3. **Maintains modularity**
   - Create new `prepare_dataset_cached.py` wrapper script
   - Don't modify original `prepare_dataset.py`
   - Reuse existing storage utilities
   - Keep same CLI arguments as original

4. **Preserves simplicity**
   - SMILES-based deduplication (simple, effective)
   - No complex database or distributed systems
   - Clear error messages
   - Easy to understand and debug

5. **Provides clear documentation**
   - How caching works
   - Performance comparison
   - Usage examples
   - Troubleshooting guide

### Constraints

- Use existing `MMomentA.data.storage` utilities (don't rewrite HDF5 logic)
- Use existing `MMomentA.data.batch.BatchProcessor` for calculations
- Maintain backwards compatibility with existing pipeline
- No external dependencies (use only existing packages)
- Must handle edge cases:
  - Corrupted cache files
  - Partial results from crashed runs
  - Changed QM parameters (method, basis)
  - Dataset splits changing when adding molecules

### Deliverables

1. **`scripts/prepare_dataset_cached.py`**
   - Wrapper around existing functionality
   - Implements caching logic
   - Same arguments as `prepare_dataset.py` + caching options
   - ~200-300 lines of code

2. **Update pipeline script**
   - Change `prepare_dataset.py` to `prepare_dataset_cached.py`
   - Update from `--n-jobs 1` to `--n-jobs 16`

3. **Documentation (`CACHING_OPTIMIZATION.md`)**
   - How it works (algorithm, data structures)
   - Performance comparison (before/after tables)
   - Usage examples
   - Troubleshooting section
   - Technical details (memory, disk, thread safety)

4. **Integration**
   - Verify works with existing pipeline
   - No breaking changes to existing code
   - Clear migration path

### Expected Outcomes

**Performance improvements**:
- First run: 20-30 min → 5-8 min (4x speedup from parallel)
- Subsequent runs (same molecules): 20-30 min → ~5 seconds (240x+ speedup from caching)
- Partial new molecules: Only compute the new ones

**Development workflow**:
- Iterate on training code without waiting for MPFIT recalculation
- Add molecules incrementally
- Quick debugging cycles

### Technical Preferences

- **Efficiency**: Minimize redundant computation
- **Modularity**: Separate concerns, reusable components
- **Simplicity**: Easy to understand and maintain
- **Robustness**: Handle errors gracefully, atomic operations
- **Documentation**: Clear, comprehensive, with examples

### Code Style

- Type hints for function signatures
- Docstrings with Parameters/Returns sections
- Logging for key operations
- Clear variable names
- Comments for non-obvious logic

Please implement this optimization with the above requirements.
```

---

## Example Usage

When facing a similar optimization task:

1. **Copy the template above**
2. **Customize** the sections:
   - Script names and paths
   - Current performance metrics
   - Storage format and structure
   - Existing code organization
3. **Add context** about:
   - What was already tried
   - Known issues or constraints
   - Specific requirements for your use case
4. **Submit to AI** and review the plan before execution

## Key Principles Demonstrated

### 1. Efficiency First
- Parallel processing: Maximum throughput
- Caching: Avoid redundant work
- Atomic operations: No corruption

### 2. Modularity
- New wrapper script (don't modify original)
- Reuse existing utilities
- Clear separation of concerns

### 3. Simplicity
- SMILES-based deduplication (simple key)
- HDF5 for storage (existing infrastructure)
- No complex distributed systems

### 4. Robustness
- Atomic saves (temp file + rename)
- Error handling
- Graceful degradation (fallback to sequential)

### 5. Documentation
- How it works
- Performance metrics
- Usage examples
- Troubleshooting

## Checklist for AI Implementation

When reviewing AI output, verify:

- [ ] Parallel processing enabled and tested
- [ ] Caching logic handles all edge cases
- [ ] SMILES deduplication is correct
- [ ] Atomic saves prevent corruption
- [ ] No modifications to original code
- [ ] Same CLI arguments + new caching options
- [ ] Performance metrics documented
- [ ] Usage examples provided
- [ ] Troubleshooting guide included
- [ ] Code has type hints and docstrings
- [ ] Integration with existing pipeline verified

## Common Pitfalls to Avoid

1. **Over-engineering**: Don't build a distributed database when SMILES deduplication suffices
2. **Breaking changes**: Don't modify original scripts, create wrappers
3. **Poor error handling**: Atomic saves are critical for data integrity
4. **No documentation**: Future you will thank present you
5. **Ignoring existing code**: Reuse storage and batch processing utilities
6. **No performance metrics**: Document speedups with real numbers
7. **Complex dependencies**: Use existing packages only

## Extension Ideas

Once basic caching works, consider:

1. **Resume capability**: Save incremental checkpoints during batch processing
2. **Content-based deduplication**: Use molecular fingerprints instead of SMILES
3. **Distributed caching**: Share cache across cluster nodes
4. **Cache validation**: Detect when QM parameters change and invalidate cache
5. **Statistics**: Track cache hit rate, computation time per molecule
6. **Garbage collection**: Remove old/unused cache entries

## Related Patterns

This optimization pattern applies to:
- **QM calculations**: DFT, MPFIT, RESP, ESP
- **Molecular dynamics**: Trajectory analysis
- **Machine learning**: Feature extraction, embeddings
- **Data processing**: ETL pipelines
- **Scientific computing**: Any expensive, cacheable computation

The key insight: **Deduplication by canonical identifier** (SMILES, InChI, hash, etc.)

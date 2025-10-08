#!/bin/bash
# Update batch.py with pickle error fix
# Run this on Perlmutter to apply the fix

echo "Updating MMomentA/data/batch.py with pickle error fix..."

cd /global/homes/p/parmar/MoML/MMomentA

# Backup original
cp MMomentA/data/batch.py MMomentA/data/batch.py.backup

# Create fixed version
cat > MMomentA/data/batch_fixed.patch << 'PATCH'
--- batch.py.orig	2025-10-06 20:00:00.000000000 -0700
+++ batch.py	2025-10-06 21:00:00.000000000 -0700
@@ -186,18 +186,31 @@
 
             for method_name, calculator in self.calculators.items():
                 logger.debug(f"[Process {pid}] Running {method_name} on molecule {mol_idx}")
-                result = calculator.compute(molecule)
 
-                if result.success:
-                    results[method_name] = self._serialize_result(result)
-                    logger.debug(
-                        f"[Process {pid}] {method_name} succeeded in {result.time_seconds:.2f}s"
-                    )
-                else:
-                    failed_methods.append(method_name)
-                    logger.warning(
-                        f"[Process {pid}] {method_name} failed: {result.error_message}"
-                    )
-                    results[method_name] = {
-                        "error": result.error_message,
-                        "time": result.time_seconds
-                    }
+                try:
+                    result = calculator.compute(molecule)
+
+                    if result.success:
+                        results[method_name] = self._serialize_result(result)
+                        logger.debug(
+                            f"[Process {pid}] {method_name} succeeded in {result.time_seconds:.2f}s"
+                        )
+                    else:
+                        failed_methods.append(method_name)
+                        # Convert error_message to string to avoid pickling issues
+                        error_msg = str(result.error_message) if result.error_message else "Unknown error"
+                        logger.warning(
+                            f"[Process {pid}] {method_name} failed: {error_msg}"
+                        )
+                        results[method_name] = {
+                            "error": error_msg,
+                            "time": result.time_seconds
+                        }
+                except Exception as e:
+                    # Catch any exceptions and convert to string to avoid pickling issues
+                    failed_methods.append(method_name)
+                    error_msg = f"{type(e).__name__}: {str(e)}"
+                    logger.warning(
+                        f"[Process {pid}] {method_name} raised exception: {error_msg}"
+                    )
+                    results[method_name] = {
+                        "error": error_msg,
+                        "time": 0.0
+                    }
PATCH

echo "Backup saved to: MMomentA/data/batch.py.backup"
echo ""
echo "Please manually apply the fix to MMomentA/data/batch.py:"
echo "1. Open MMomentA/data/batch.py in an editor"
echo "2. Find the section around line 188-201 (for loop with calculators)"
echo "3. Wrap the calculator.compute() call and result handling in try-except"
echo "4. Convert error_message to str() before storing"
echo ""
echo "Or copy the updated batch.py from your local repository"

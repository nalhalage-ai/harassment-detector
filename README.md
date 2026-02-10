 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/README.md b/README.md
index 02084c324f0b0eac522131cc3370eec52c1a478e..c6bdf7ecdfaf3b149237f435f5bc3f9d7da019cb 100644
--- a/README.md
+++ b/README.md
@@ -1 +1,24 @@
-# harassment-detector
\ No newline at end of file
+# harassment-detector
+
+## Synthetic dataset
+
+Generated a gender-inclusive synthetic harassment dataset at:
+
+- `data/synthetic_harassment_dataset.csv`
+
+### Dataset details
+
+- Total examples: **500**
+- Labels covered:
+  - `verbal`
+  - `sexual`
+  - `cyber`
+  - `stalking`
+  - `threat`
+  - `workplace`
+  - `physical`
+  - `non-harassment`
+- CSV columns:
+  - `id`
+  - `text`
+  - `label`
 
EOF
)

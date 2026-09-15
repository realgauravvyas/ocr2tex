import re
from pathlib import Path

orig = Path(r"D:\ocr2tex\v5 training\train_glm_ocr_v5.py").read_text(encoding="utf-8")
dest_file = Path(r"D:\ocr2tex\lightning_ai_migration\train_lightning.py")
dest = dest_file.read_text(encoding="utf-8")

# Extract exact GlmOcrCollator from orig
collator_match = re.search(r"(class GlmOcrCollator:.*?\n        return out\n)", orig, re.DOTALL)
if not collator_match:
    print("Failed to find collator in train_glm_ocr_v5.py")
    exit(1)

collator_code = collator_match.group(1)

# Replace in dest
dest_replaced, count = re.subn(r"class GlmOcrCollator:.*?\n        return batch_out\n", collator_code, dest, flags=re.DOTALL)
if count > 0:
    dest_file.write_text(dest_replaced, encoding="utf-8")
    print(f"SUCCESS: Replaced GlmOcrCollator ({count} occurrence)")
else:
    print("Failed to replace in train_lightning.py")

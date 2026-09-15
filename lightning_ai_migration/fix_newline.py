with open(r"D:\ocr2tex\lightning_ai_migration\train_lightning.py", "rb") as f:
    raw = f.read()

# Replace the broken string
raw = raw.replace(b'self.newline_id = tok.convert_tokens_to_ids("\r\n")', b'self.newline_id = tok.convert_tokens_to_ids("\\n")')
raw = raw.replace(b'self.newline_id = tok.convert_tokens_to_ids("\n")', b'self.newline_id = tok.convert_tokens_to_ids("\\n")')

# Also ensure all line endings are LF (\n)
raw = raw.replace(b'\r\n', b'\n')

with open(r"D:\ocr2tex\lightning_ai_migration\train_lightning.py", "wb") as f:
    f.write(raw)

import py_compile
py_compile.compile(r"D:\ocr2tex\lightning_ai_migration\train_lightning.py", doraise=True)
print("COMPILATION VERIFIED: SUCCESSFUL!")

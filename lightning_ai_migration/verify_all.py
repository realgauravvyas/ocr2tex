import py_compile
for f in ["benchmark_lightning.py", "generate_summary_report.py"]:
    path = rf"D:\ocr2tex\lightning_ai_migration\{f}"
    with open(path, "rb") as fp:
        raw = fp.read().replace(b'\r\n', b'\n')
    with open(path, "wb") as fp:
        fp.write(raw)
    py_compile.compile(path, doraise=True)
    print(f"Verified {f}: OK")

sh_path = r"D:\ocr2tex\lightning_ai_migration\run_train.sh"
with open(sh_path, "rb") as fp:
    raw = fp.read().replace(b'\r\n', b'\n')
with open(sh_path, "wb") as fp:
    fp.write(raw)
print("Verified run_train.sh: OK")

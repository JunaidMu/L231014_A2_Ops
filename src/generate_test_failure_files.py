"""
Helper script to generate the 3 test files needed to demonstrate Task 4.2's
three validation failure cases in the browser:
1. test_uploads/1_wrong_filetype.txt       (Wrong file type: .txt)
2. test_uploads/2_over_size_limit_3mb.jpg  (Over 2.0 MB size limit: 3.1 MB)
3. test_uploads/3_corrupt_fake_image.jpg   (Looks like an image by extension .jpg, but cannot be decoded)
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "test_uploads"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Wrong file type (.txt)
    f1 = OUT_DIR / "1_wrong_filetype.txt"
    f1.write_text("This is a plain text document, not an image.", encoding="utf-8")

    # 2. Over 2.0 MB size limit (3.1 MB dummy payload with .jpg extension)
    f2 = OUT_DIR / "2_over_size_limit_3mb.jpg"
    f2.write_bytes(b"\xff\xd8\xff\xe0" + (b"A" * (3 * 1024 * 1024 + 100000)))

    # 3. Looks like an image (.jpg) and under size limit (2 KB), but corrupt/undecodable bytes
    f3 = OUT_DIR / "3_corrupt_fake_image.jpg"
    f3.write_bytes(b"NOT_A_VALID_JPEG_HEADER_CORRUPT_BINARY_BYTES_" * 40)

    print(f"[INFO] Generated 3 failure test files in: {OUT_DIR}")
    for p in (f1, f2, f3):
        print(f"  - {p.name} ({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

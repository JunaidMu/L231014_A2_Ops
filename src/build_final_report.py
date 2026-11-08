"""
Generates the complete, formatted, publication-grade Word document (`FinalReport.docx`)
for all 8 Tasks (100 Marks + Bonus) using pure Python + OOXML (`zipfile` + `PIL`),
without modifying `Evidence Document.docx`.

Includes:
- Section 11.1 Front Matter (Title, Name: Junaid Muhammad, Roll No: 23L-1014 / L231014_A2_Ops,
  GitHub link, DagsHub link, Live Vercel API URL: https://l231014-a2-ops.vercel.app,
  Commit hash, AI Assistant Disclosure).
- Chapters 1 through 8 with all 62 extracted screenshots from Evidence Document.docx,
  numbered captions, in-text references, formatted tables, and comprehensive explanations
  of what was done, why, how it works, and what went wrong.
"""

import zipfile
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = PROJECT_ROOT / "reports" / "extracted_evidence"
OUTPUT_DOCX = PROJECT_ROOT / "FinalReport.docx"


def xml_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


class DocxBuilder:
    def __init__(self):
        self.body_elements: list[str] = []
        self.media_files: list[tuple[str, Path]] = []
        self._rel_counter = 10
        self._doc_pr_counter = 1

    def add_heading1(self, text: str):
        t = xml_escape(text)
        self.body_elements.append(
            f"""
            <w:p>
              <w:pPr>
                <w:spacing w:before="320" w:after="140"/>
                <w:pBdr>
                  <w:bottom w:val="single" w:sz="12" w:space="4" w:color="1E3A8A"/>
                </w:pBdr>
              </w:pPr>
              <w:r>
                <w:rPr>
                  <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
                  <w:b/>
                  <w:color w:val="1E3A8A"/>
                  <w:sz w:val="32"/>
                </w:rPr>
                <w:t>{t}</w:t>
              </w:r>
            </w:p>
            """
        )

    def add_heading2(self, text: str):
        t = xml_escape(text)
        self.body_elements.append(
            f"""
            <w:p>
              <w:pPr>
                <w:spacing w:before="240" w:after="100"/>
              </w:pPr>
              <w:r>
                <w:rPr>
                  <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
                  <w:b/>
                  <w:color w:val="0F172A"/>
                  <w:sz w:val="26"/>
                </w:rPr>
                <w:t>{t}</w:t>
              </w:r>
            </w:p>
            """
        )

    def add_paragraph(self, text: str, bold_prefix: str = "", italic: bool = False):
        p_runs = ""
        if bold_prefix:
            bp = xml_escape(bold_prefix)
            p_runs += f"""
            <w:r>
              <w:rPr>
                <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
                <w:b/>
                <w:sz w:val="22"/>
              </w:rPr>
              <w:t xml:space="preserve">{bp} </w:t>
            </w:r>
            """
        t = xml_escape(text)
        italic_tag = "<w:i/>" if italic else ""
        p_runs += f"""
        <w:r>
          <w:rPr>
            <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
            {italic_tag}
            <w:sz w:val="22"/>
          </w:rPr>
          <w:t xml:space="preserve">{t}</w:t>
        </w:r>
        """
        self.body_elements.append(
            f"""
            <w:p>
              <w:pPr>
                <w:spacing w:before="60" w:after="100" w:line="276" w:lineRule="auto"/>
                <w:jc w:val="both"/>
              </w:pPr>
              {p_runs}
            </w:p>
            """
        )

    def add_bullet(self, bold_prefix: str, text: str):
        bp = xml_escape(bold_prefix)
        t = xml_escape(text)
        self.body_elements.append(
            f"""
            <w:p>
              <w:pPr>
                <w:ind w:left="480" w:hanging="240"/>
                <w:spacing w:before="40" w:after="60" w:line="264" w:lineRule="auto"/>
              </w:pPr>
              <w:r>
                <w:rPr>
                  <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
                  <w:b/>
                  <w:sz w:val="22"/>
                </w:rPr>
                <w:t xml:space="preserve">• {bp} </w:t>
              </w:r>
              <w:r>
                <w:rPr>
                  <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
                  <w:sz w:val="22"/>
                </w:rPr>
                <w:t xml:space="preserve">{t}</w:t>
              </w:r>
            </w:p>
            """
        )

    def add_table(self, headers: list[str], rows: list[list[str]]):
        header_cells = ""
        for h in headers:
            header_cells += f"""
            <w:tc>
              <w:tcPr>
                <w:shd w:val="clear" w:color="auto" w:fill="1E3A8A"/>
                <w:tcMar>
                  <w:top w:w="80" w:type="dxa"/>
                  <w:bottom w:w="80" w:type="dxa"/>
                  <w:left w:w="100" w:type="dxa"/>
                  <w:right w:w="100" w:type="dxa"/>
                </w:tcMar>
              </w:tcPr>
              <w:p>
                <w:r>
                  <w:rPr>
                    <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
                    <w:b/>
                    <w:color w:val="FFFFFF"/>
                    <w:sz w:val="19"/>
                  </w:rPr>
                  <w:t>{xml_escape(h)}</w:t>
                </w:r>
              </w:p>
            </w:tc>
            """

        body_rows = ""
        for r_idx, row in enumerate(rows):
            fill = "F1F5F9" if r_idx % 2 == 1 else "FFFFFF"
            cells_xml = ""
            for cell in row:
                cells_xml += f"""
                <w:tc>
                  <w:tcPr>
                    <w:shd w:val="clear" w:color="auto" w:fill="{fill}"/>
                    <w:tcMar>
                      <w:top w:w="60" w:type="dxa"/>
                      <w:bottom w:w="60" w:type="dxa"/>
                      <w:left w:w="100" w:type="dxa"/>
                      <w:right w:w="100" w:type="dxa"/>
                    </w:tcMar>
                  </w:tcPr>
                  <w:p>
                    <w:r>
                      <w:rPr>
                        <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
                        <w:sz w:val="19"/>
                      </w:rPr>
                      <w:t>{xml_escape(cell)}</w:t>
                    </w:r>
                  </w:p>
                </w:tc>
                """
            body_rows += f"<w:tr>{cells_xml}</w:tr>"

        self.body_elements.append(
            f"""
            <w:tbl>
              <w:tblPr>
                <w:tblW w:w="5000" w:type="pct"/>
                <w:tblBorders>
                  <w:top w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>
                  <w:left w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>
                  <w:bottom w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>
                  <w:right w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>
                  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>
                  <w:insideV w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>
                </w:tblBorders>
              </w:tblPr>
              <w:tr>{header_cells}</w:tr>
              {body_rows}
            </w:tbl>
            <w:p><w:pPr><w:spacing w:after="80"/></w:pPr></w:p>
            """
        )

    def add_image(self, img_path: Path, caption: str, max_width_inches: float = 5.8):
        if not img_path.exists():
            self.add_paragraph(f"[Missing Image: {img_path.name}] — {caption}", italic=True)
            return

        with Image.open(img_path) as im:
            w_px, h_px = im.size

        aspect = h_px / float(w_px) if w_px else 0.6
        width_in = min(max_width_inches, 6.0)
        height_in = width_in * aspect
        if height_in > 4.1:
            height_in = 4.1
            width_in = height_in / aspect

        cx = int(width_in * 914400)
        cy = int(height_in * 914400)

        self._rel_counter += 1
        r_id = f"rId{self._rel_counter}"
        self.media_files.append((r_id, img_path))
        doc_pr_id = self._doc_pr_counter
        self._doc_pr_counter += 1

        self.body_elements.append(
            f"""
            <w:p>
              <w:pPr>
                <w:jc w:val="center"/>
                <w:spacing w:before="110" w:after="40"/>
              </w:pPr>
              <w:r>
                <w:drawing>
                  <wp:inline distT="0" distB="0" distL="0" distR="0">
                    <wp:extent cx="{cx}" cy="{cy}"/>
                    <wp:docPr id="{doc_pr_id}" name="Picture {doc_pr_id}"/>
                    <a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
                      <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
                        <pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
                          <pic:nvPicPr>
                            <pic:cNvPr id="{doc_pr_id}" name="{xml_escape(img_path.name)}"/>
                            <pic:cNvPicPr/>
                          </pic:nvPicPr>
                          <pic:blipFill>
                            <a:blip r:embed="{r_id}"/>
                            <a:stretch><a:fillRect/></a:stretch>
                          </pic:blipFill>
                          <pic:spPr>
                            <a:xfrm>
                              <a:off x="0" y="0"/>
                              <a:ext cx="{cx}" cy="{cy}"/>
                            </a:xfrm>
                            <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
                          </pic:spPr>
                        </pic:pic>
                      </a:graphicData>
                    </a:graphic>
                  </wp:inline>
                </w:drawing>
              </w:r>
            </w:p>
            <w:p>
              <w:pPr>
                <w:jc w:val="center"/>
                <w:spacing w:before="20" w:after="150"/>
              </w:pPr>
              <w:r>
                <w:rPr>
                  <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
                  <w:i/>
                  <w:color w:val="475569"/>
                  <w:sz w:val="19"/>
                </w:rPr>
                <w:t>{xml_escape(caption)}</w:t>
              </w:r>
            </w:p>
            """
        )

    def save(self, out_path: Path):
        content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
          <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
          <Default Extension="xml" ContentType="application/xml"/>
          <Default Extension="png" ContentType="image/png"/>
          <Default Extension="jpg" ContentType="image/jpeg"/>
          <Default Extension="jpeg" ContentType="image/jpeg"/>
          <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
        </Types>
        """

        root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
          <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
        </Relationships>
        """

        doc_rels_items = ""
        for r_id, m_path in self.media_files:
            doc_rels_items += (
                f'<Relationship Id="{r_id}" '
                f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                f'Target="media/{m_path.name}"/>'
            )

        doc_rels = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
          {doc_rels_items}
        </Relationships>
        """

        body_xml = "\n".join(self.body_elements)
        document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <w:document
          xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
          xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
          xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
          xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
          <w:body>
            {body_xml}
            <w:sectPr>
              <w:pgSz w:w="12240" w:h="15840"/>
              <w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080" w:header="720" w:footer="720" w:gutter="0"/>
            </w:sectPr>
          </w:body>
        </w:document>
        """

        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", content_types.strip())
            zf.writestr("_rels/.rels", root_rels.strip())
            zf.writestr("word/_rels/document.xml.rels", doc_rels.strip())
            zf.writestr("word/document.xml", document_xml.strip())
            added_media = set()
            for _, m_path in self.media_files:
                if m_path.name not in added_media:
                    zf.write(m_path, f"word/media/{m_path.name}")
                    added_media.add(m_path.name)


def build_final_report():
    doc = DocxBuilder()

    # -----------------------------------------------------------------------
    # FRONT MATTER (Section 11.1 Requirements)
    # -----------------------------------------------------------------------
    doc.add_heading1("VisionGuard (L231014_A2_Ops): AI vs. Real Image Classification & Deployment")
    doc.add_paragraph("Course: MLOps (Semester 7) — Assignment 2 Final Technical Report", bold_prefix="Submission Report:")
    doc.add_table(
        ["Attribute", "Details"],
        [
            ["Student Name", "Junaid Muhammad"],
            ["Roll Number", "23L-1014 (Project Identifier Everywhere: L231014_A2_Ops)"],
            ["GitHub Repository (Main)", "https://github.com/JunaidMu/L231014_A2_Ops"],
            ["GitHub Repository (Toy CI Lab)", "https://github.com/JunaidMu/L231014_CI_Toy"],
            ["DagsHub Project & MLflow Registry", "https://dagshub.com/JunaidMu/L231014_A2_Ops"],
            ["Live Cloud Model API URL (Vercel)", "https://l231014-a2-ops.vercel.app (Predict: https://l231014-a2-ops.vercel.app/predict)"],
            ["Graded Commit Hash", "38820ff / main branch"],
            [
                "AI Assistant & Tutorial Disclosure",
                "Used Antigravity (Google DeepMind coding assistant) for pair-programming Python scripts, "
                "debugging Windows Smart App Control Cython DLL restrictions, distilling model weights for Vercel's "
                "250 MB serverless bundle limit, and formatting report tables. All terminal commands, Git/DagsHub pushes, "
                "Docker builds, Postman API tests, and Vercel deployments were executed and verified by me.",
            ],
        ],
    )

    # -----------------------------------------------------------------------
    # CHAPTER 1: TASK 1 — DATA UNDERSTANDING AND VERSIONING (7 MARKS)
    # -----------------------------------------------------------------------
    doc.add_heading1("Chapter 1 — Task 1: Data Understanding and Versioning (7 Marks)")

    doc.add_heading2("Sub-task 1.1: Isolated Python Environment & GPU Check (1 Mark)")
    doc.add_bullet(
        "What I did & why:",
        "Created an isolated Python 3.12.10 virtual environment (`python -m venv venv`), activated it via "
        "`venv\\Scripts\\activate`, and installed the required MLOps and deep learning libraries (`tensorflow==2.21.0`, "
        "`keras-tuner==1.4.8`, `mlflow==3.18.0`, `dvc==3.67.1`, `Flask==3.1.3`, `numpy==2.5.4`, `pandas==3.1.1`, "
        "`scikit-learn==1.9.1`, `Pillow==13.0.0`). Freezing exact versions into `requirements_exact.txt` guarantees "
        "deterministic environment reproduction across local development, CI runners, and Docker containers.",
    )
    doc.add_bullet(
        "How it works & GPU verification:",
        "Querying `tf.config.list_physical_devices('GPU')` returned `[]` (Figure 1.1c) because native Windows builds "
        "of TensorFlow (>= 2.11) execute on CPU using Intel oneDNN instructions unless run inside WSL2. This directly "
        "informed our model sizing (`160x160` MobileNetV2) and CPU-only Docker base image selection in Task 6.",
    )
    doc.add_image(EVIDENCE_DIR / "image1.png", "Figure 1.1a: Creating and activating the isolated Python virtual environment (venv).")
    doc.add_image(EVIDENCE_DIR / "image2.png", "Figure 1.1b: Installing project dependencies from requirements.txt inside venv.")
    doc.add_image(EVIDENCE_DIR / "image3.png", "Figure 1.1c: Terminal verification of user identity (junai), timestamp, exact package versions, and TensorFlow GPU check.")

    doc.add_heading2("Sub-task 1.2: Dataset Audit Script & Integrity Analysis (3 Marks)")
    doc.add_bullet(
        "What I did & why:",
        "Implemented `src/task1_data_audit_and_eda.py` to pull the Kaggle 'AI Generated Images vs Real Images' dataset "
        "via the `kagglehub` API into `data/raw/` and walk every file. For each image, the script records its folder class "
        "(`AiArtData` vs `RealArt`), byte/KB size, pixel width and height, Pillow colour mode, container format, and "
        "SHA-256 content hash, while running `img.verify()` and `img.load()` to detect corrupt files.",
    )
    doc.add_table(
        [
            "Class",
            "Total Files",
            "Valid Images",
            "Corrupt Files",
            "Exact Duplicates (SHA-256)",
            "Width (Min / Median / Max)",
            "Height (Min / Median / Max)",
            "File Size KB (Min / Median / Max)",
        ],
        [
            ["AiArtData", "538", "538", "0", "1", "225 / 1028 / 6000", "135 / 800 / 4001", "11.5 / 195.1 / 5494.3"],
            ["RealArt", "435", "435", "0", "0", "183 / 940 / 8495", "158 / 720 / 7091", "11.1 / 141.6 / 9956.4"],
            ["ALL (Total)", "973", "973", "0", "1", "183 / 1024 / 8495", "135 / 768 / 7091", "11.1 / 165.4 / 9956.4"],
        ],
    )
    doc.add_bullet(
        "Audit Findings & What Went Wrong:",
        "As shown in Figure 1.2, 0 corrupt files were found among the 973 images, and 1 exact SHA-256 byte duplicate "
        "was detected in `AiArtData` (`image34.jpg`, hash `f3268a4f...`). Importantly, the audit revealed four distinct "
        "colour modes (`RGB`: 894, `RGBA`: 49, `P` palette: 23, `L` grayscale: 7) and two formats (`JPEG`: 830, `PNG`: 143). "
        "Without auditing first, feeding 1-channel (`L`) or 4-channel (`RGBA`) tensors into a 3-channel CNN causes shape "
        "mismatch crashes; therefore, we enforced `channels=3` during decoding.",
    )
    doc.add_image(EVIDENCE_DIR / "image4.png", "Figure 1.2: Terminal execution of src/task1_data_audit_and_eda.py displaying the per-class audit table and SHA-256 duplicate count.")

    doc.add_heading2("Sub-task 1.3: Visual Exploratory Data Analysis (1.5 Marks)")
    doc.add_image(EVIDENCE_DIR / "image5.png", "Figure 1.3a: Grid of 5 randomly sampled images per class (AiArtData vs RealArt) showing native dimensions and colour modes.")
    doc.add_paragraph(
        "Figure 1.3a illustrates substantial stylistic diversity across both classes, ranging from portraits and architectural "
        "scenes to surreal digital paintings, confirming that simple colour thresholding cannot separate the classes. Native image "
        "resolutions vary widely from small thumbnails (201x251) to ultra-high-definition canvases (3733x2133), and aspect ratios "
        "span square, portrait, and wide landscape formats. Consequently, standardizing spatial dimensions via resizing is mandatory "
        "before batching images into fixed-rank tensors.",
        bold_prefix="Interpretation of Sample Grid (Figure 1.3a):",
    )
    doc.add_image(EVIDENCE_DIR / "image6.png", "Figure 1.3b: Histograms and KDE curves of image width (px), height (px), and file size (KB) per class.")
    doc.add_paragraph(
        "Figure 1.3b shows that both image width and height distributions are strongly right-skewed, with AiArtData exhibiting a "
        "pronounced peak around 1024px width and 768–1024px height (standard default output resolutions of diffusion models like "
        "Stable Diffusion and Midjourney), whereas RealArt has slightly lower medians (940x720) but a longer tail reaching 8495x7091. "
        "File sizes are similarly heavily right-skewed: most images sit under 500 KB (median 195.1 KB for AI vs 141.6 KB for Real), "
        "while uncompressed PNGs and ultra-high-resolution JPEGs form a tail reaching 9.95 MB.",
        bold_prefix="Interpretation of Histograms (Figure 1.3b):",
    )

    doc.add_heading2("Sub-task 1.4: Data Version Control (DVC) — Toy Lab & Dataset Tracking (1.5 Marks)")
    doc.add_bullet(
        "What I did (Toy Lab & Project Dataset):",
        "In `dvc_toy_lab/`, initialized Git and DVC, created a 2-row `data.csv`, tracked it via `dvc add data.csv`, committed "
        "`data.csv.dvc` (v1), pushed to local remote storage (`../dvc_toy_remote`), appended 2 more rows, committed v2, and pushed "
        "(Figure 1.4a). Next, we checked out `HEAD~1 data.csv.dvc` + `dvc checkout` to roll back to v1, deleted both `data.csv` and "
        "`.dvc/cache`, and restored the data cleanly using `dvc pull` (Figure 1.4b). Finally, we applied DVC to our main project's "
        "`data/raw` folder (`975` files, `503.78 MB`), generating `data/raw.dvc` with MD5 hash `b2cec67839f742df4653f66d4bcbbafb.dir` (Figure 1.4c).",
    )
    doc.add_bullet(
        "What the small .dvc pointer file holds & why large data does not belong in Git:",
        "The `.dvc` pointer file (`data.csv.dvc` and `data/raw.dvc`) is a lightweight YAML manifest storing the content-addressed MD5 hash (`md5`), "
        "byte size (`size: 503783378`), file count (`nfiles: 975`), and relative path (`raw`). Git is engineered for line-by-line source code diffs "
        "and clones every historical blob into `.git/objects`; putting 504 MB of binary images directly in Git bloats the repository, violates GitHub's "
        "100 MB file limit, and slows down cloning and branching.",
    )
    doc.add_image(EVIDENCE_DIR / "image7.png", "Figure 1.4a: DVC Toy Lab — Tracking data.csv v1 and v2, pushing to remote storage, and verifying git log --oneline.")
    doc.add_image(EVIDENCE_DIR / "image8.png", "Figure 1.4b: DVC Toy Lab — Returning to v1 via git checkout HEAD~1 + dvc checkout, deleting local data/cache, and restoring via dvc pull.")
    doc.add_image(EVIDENCE_DIR / "image9.png", "Figure 1.4c: Applying DVC to the main VisionGuard dataset folder (data/raw) and inspecting git status.")

    # -----------------------------------------------------------------------
    # CHAPTER 2: TASK 2 — MODEL DEVELOPMENT WITH TENSORFLOW/KERAS (14 MARKS + BONUS)
    # -----------------------------------------------------------------------
    doc.add_heading1("Chapter 2 — Task 2: Model Development with TensorFlow/Keras (14 Marks + Bonus)")

    doc.add_heading2("Sub-task 2.1: tf.data Input Pipeline & Augmentation (2 Marks)")
    doc.add_bullet(
        "Pipeline Stage-by-Stage Architecture:",
        "After removing the SHA-256 duplicate (`image34.jpg`) and 2 files with animated/WebP headers unsupported by TensorFlow's "
        "C++ JPEG/PNG decoder, the 970 valid images were stratified-split (`SEED=42`) into 70% Train (679), 15% Validation (145), "
        "and 15% Test (146) CSV manifests under `data/splits/`. The `tf.data.Dataset` pipeline: (1) slices filepaths and binary labels, "
        "(2) shuffles training data (`buffer_size=679, seed=42`), (3) maps parallel `decode_and_resize` (`tf.io.decode_image(channels=3)` "
        "+ bilinear `tf.image.resize` to `160x160` float32), (4) caches decoded tensors in memory (`.cache()`), (5) applies random "
        "augmentation (`RandomFlip('horizontal')`, `RandomRotation(0.08)`, `RandomZoom(0.10)`, `RandomContrast(0.10)`) exclusively to "
        "`train_ds`, (6) groups into mini-batches of `32`, and (7) prefetches (`tf.data.AUTOTUNE`) to overlap CPU preprocessing with model execution.",
    )
    doc.add_bullet(
        "Image Size Choice (160x160) & Aspect Ratio Impact:",
        "We chose `160x160` because it is a standard spatial resolution natively supported by pretrained `MobileNetV2` ImageNet weights "
        "that preserves fine texture artifacts while keeping CPU epoch times under 8 seconds. Because `tf.image.resize` warps images to a "
        "fixed `1:1` square grid without padding, wide landscape or tall portrait images experience geometric aspect-ratio compression (squashing).",
    )
    doc.add_image(EVIDENCE_DIR / "image10.png", "Figure 2.1: One augmented training batch (160x160 RGB) showing random flips, rotations, zoom, and contrast perturbations.")

    doc.add_heading2("Sub-task 2.2: Baseline CNN Built From Scratch (3 Marks)")
    doc.add_bullet(
        "Architecture, Loss, Optimizer & Metrics Justification:",
        "We designed `Baseline_CNN_L231014` (Figures 2.2a–2.2b) with a `[0, 1]` `Rescaling` layer followed by three progressive convolutional "
        "blocks (`32 -> 64 -> 128` filters of size `3x3`, `padding='same'`, ReLU). Each `Conv2D` is paired with `BatchNormalization` (to stabilize "
        "channel distributions and speed up convergence) and `2x2 MaxPooling2D` (to halve spatial dimensions from `160 -> 80 -> 40 -> 20` and "
        "provide translation invariance). `GlobalAveragePooling2D` collapses the `20x20x128` map into a 128-D vector (avoiding dense parameter "
        "explosion), followed by `Dense(64, relu)`, `Dropout(0.4)` (to prevent co-adaptation overfitting), and `Dense(1, sigmoid)`. We compiled "
        "with `binary_crossentropy`, `Adam(lr=1e-3)`, and `accuracy` + `ROC-AUC` metrics.",
    )
    doc.add_table(
        ["Layer Name", "Layer Type", "Output Shape", "Param Count", "Trainable", "Architectural Purpose"],
        [
            ["input_image", "InputLayer", "(None, 160, 160, 3)", "0", "True", "Accepts 160x160x3 RGB mini-batches"],
            ["rescale_0_1", "Rescaling", "(None, 160, 160, 3)", "0", "True", "Scales [0, 255] pixels to [0, 1]"],
            ["conv1", "Conv2D (32, 3x3)", "(None, 160, 160, 32)", "896", "True", "Low-level edge & color gradient filters"],
            ["bn1", "BatchNormalization", "(None, 160, 160, 32)", "128", "True", "Normalizes Block 1 activations"],
            ["pool1", "MaxPooling2D (2x2)", "(None, 80, 80, 32)", "0", "True", "2x spatial downsampling to 80x80"],
            ["conv2", "Conv2D (64, 3x3)", "(None, 80, 80, 64)", "18,496", "True", "Mid-level texture & brush pattern filters"],
            ["bn2", "BatchNormalization", "(None, 80, 80, 64)", "256", "True", "Normalizes Block 2 activations"],
            ["pool2", "MaxPooling2D (2x2)", "(None, 40, 40, 64)", "0", "True", "2x spatial downsampling to 40x40"],
            ["conv3", "Conv2D (128, 3x3)", "(None, 40, 40, 128)", "73,856", "True", "High-level semantic & artifact patterns"],
            ["bn3", "BatchNormalization", "(None, 40, 40, 128)", "512", "True", "Normalizes Block 3 activations"],
            ["pool3", "MaxPooling2D (2x2)", "(None, 20, 20, 128)", "0", "True", "2x spatial downsampling to 20x20"],
            ["gap", "GlobalAveragePooling2D", "(None, 128)", "0", "True", "Spatial average pooling per channel"],
            ["dense_hidden", "Dense (64, ReLU)", "(None, 64)", "8,256", "True", "Non-linear feature combination"],
            ["dropout", "Dropout (0.4)", "(None, 64)", "0", "True", "Regularization against overfitting"],
            ["sigmoid_output", "Dense (1, Sigmoid)", "(None, 1)", "65", "True", "Outputs probability P(AI) in [0, 1]"],
            ["TOTAL", "—", "—", "102,465", "102,017 Trainable", "448 Non-trainable BN moving stats"],
        ],
    )
    doc.add_image(EVIDENCE_DIR / "image11.png", "Figure 2.2a: Terminal output of Baseline CNN model.summary() (top section).")
    doc.add_image(EVIDENCE_DIR / "image12.png", "Figure 2.2b: Terminal output of Baseline CNN total parameters (102,465) and per-layer shape/parameter table.")

    doc.add_heading2("Sub-task 2.3: Transfer Learning with Pretrained MobileNetV2 (3 Marks)")
    doc.add_bullet(
        "Two-Stage Transfer Learning & Preprocessing Explanation:",
        "We loaded `tf.keras.applications.MobileNetV2(input_shape=(160,160,3), include_top=False, weights='imagenet')`. Because MobileNetV2's "
        "pretrained filters expect input tensors normalized to `[-1, 1]`, we prepended `Rescaling(scale=1./127.5, offset=-1.0)` so input feature "
        "statistics match ImageNet pretraining. In Stage 1 (Figure 2.3a), we froze the entire MobileNetV2 backbone (`trainable=False`) and trained "
        "only the custom classification head (`['input_image', 'mobilenetv2_preprocess', 'gap', 'head_bn', 'head_dropout', 'sigmoid_output']`) "
        "at `lr=1e-3` for up to 8 epochs (Figure 2.3c). In Stage 2 (Figure 2.3b), we unfroze the top 20 layers of MobileNetV2 (blocks 15 and 16 + "
        "`Conv_1` and `out_relu`, keeping `BatchNormalization` layers frozen so running ImageNet variance statistics were not corrupted) and "
        "fine-tuned at a 10x smaller learning rate (`lr=1e-4`) so gradient updates refined high-level features without catastrophic forgetting (Figure 2.3d).",
    )
    doc.add_image(EVIDENCE_DIR / "image13.png", "Figure 2.3a: Terminal output showing Stage 1 (Frozen Backbone) trainable layers list and training progress.")
    doc.add_image(EVIDENCE_DIR / "image14.png", "Figure 2.3b: Terminal output showing Stage 2 (Unfrozen Last Block — 19 trainable layers) and fine-tuning progress.")
    doc.add_image(EVIDENCE_DIR / "image15.png", "Figure 2.3c: Stage 1 (Frozen MobileNetV2 Backbone) Loss and Accuracy curves with Best Epoch (7) marked.")
    doc.add_image(EVIDENCE_DIR / "image16.png", "Figure 2.3d: Stage 2 (Fine-Tuned MobileNetV2 Last Block) Loss and Accuracy curves with Best Epoch (4) marked.")

    doc.add_heading2("Sub-task 2.4: Training Discipline, Callbacks & Best Epoch (1 Mark)")
    doc.add_bullet(
        "Seeds, Callbacks & Best Epochs:",
        "We fixed Python, NumPy, and TensorFlow random seeds (`SEED = 42`) and attached three callbacks monitoring `val_loss`: "
        "`EarlyStopping(patience=3..4, restore_best_weights=True)`, `ModelCheckpoint(save_best_only=True)`, and `ReduceLROnPlateau(factor=0.5, patience=2)`. "
        "As marked by the green vertical dashed lines in Figure 2.4 and Figures 2.3c–2.3d, the Best Epochs were: Epoch 1 for the scratch Baseline CNN "
        "(which began overfitting after Epoch 1 and was halted early at Epoch 5), Epoch 7 for MobileNetV2 Stage 1, and Epoch 4 for MobileNetV2 Stage 2.",
    )
    doc.add_image(EVIDENCE_DIR / "image17.png", "Figure 2.4: Baseline CNN Loss and Accuracy curves with Best Epoch (Epoch 1) marked and early stopping at Epoch 5.")

    doc.add_heading2("Sub-task 2.5: Single Test-Set Evaluation & Threshold Justification (2 Marks)")
    doc.add_table(
        ["Model", "Val-Selected Threshold", "Test Accuracy", "Test Precision", "Test Recall", "Test F1-Score", "Test ROC-AUC"],
        [
            ["Baseline CNN (Scratch)", "0.46", "0.5890", "0.5746", "0.9625", "0.7196", "0.6142"],
            ["MobileNetV2 (Fine-Tuned)", "0.48", "0.8014", "0.7802", "0.8875", "0.8304", "0.8902"],
        ],
    )
    doc.add_bullet(
        "Threshold Justification & Worse Error Analysis:",
        "Sweeping thresholds in `[0.25, 0.75]` strictly on validation predictions (`val_ds`) maximized validation F1 at `T = 0.48` for MobileNetV2 "
        "(`Val F1 = 0.7976`, Figure 2.5b). For VisionGuard, calling a fake (AI-generated) photo real (False Negative) is worse than calling a real "
        "photo fake (False Positive), because allowing synthetic deepfakes or AI misinformation to pass as authentic undermines the core trust of an "
        "AI-detection product. Using `T = 0.48` achieves `88.75%` Test Recall (catching 71 of 80 AI test images, Figure 2.5a) alongside `80.14%` "
        "Test Accuracy and `0.8902` ROC-AUC, outperforming the scratch CNN by `+21.24%` accuracy and `+0.2760` ROC-AUC.",
    )
    doc.add_image(EVIDENCE_DIR / "image18.png", "Figure 2.5a: Test Set Confusion Matrices (Baseline CNN vs. Fine-Tuned MobileNetV2) and ROC Curve Comparison.")
    doc.add_image(EVIDENCE_DIR / "image19.png", "Figure 2.5b: Terminal output of validation threshold selection and single test-set evaluation metrics.")

    doc.add_heading2("Sub-task 2.6 & Bonus: Error Analysis (10 Misclassified Images) & Grad-CAM (1 + 1 Marks)")
    doc.add_image(EVIDENCE_DIR / "image20.png", "Figure 2.6: Grid of 10 highest-confidence misclassified test images with true labels, predicted probabilities, and original dimensions.")
    doc.add_table(
        ["#", "Test Image File", "True / Pred", "P(AI)", "Native Size", "Visual Hypothesis for Misclassification"],
        [
            ["1", "people-strategy_1200x700.jpg", "Real -> AI", "0.985", "1200x700", "Flat vector graphic background with studio cutouts mimics synthetic illustration."],
            ["2", "young-girl-holding-chaffinch...jpg", "Real -> AI", "0.959", "2880x1620", "Classical oil painting brushwork + wide 16:9 aspect ratio squashed during 160x160 resize."],
            ["3", "picturesque-scenery-waterfall...jpg", "Real -> AI", "0.953", "390x280", "Low resolution (390x280), watermark bar, and HDR aurora glow mimic diffusion smoothing."],
            ["4", "most-famous-paintings-by-jan...jpg", "Real -> AI", "0.917", "1200x690", "2x2 multi-panel collage with white borders resembles Midjourney 4-image grid outputs."],
            ["5", "nature-scenery.jpg", "Real -> AI", "0.907", "698x699", "Long-exposure silky waterfall removes natural water droplet noise."],
            ["6", "images19.jpg", "Real -> AI", "0.896", "183x275", "Low-res thumbnail (183x275) upscaled to 160x160 with strong bokeh background blur."],
            ["7", "ai-cartoon-portraits-01.jpg", "AI -> Real", "0.105", "1000x1012", "Photorealistic skin pores and studio lighting closely resemble real portrait photography."],
            ["8", "Mali_-_Bozo_girl_in_Bamako.jpg", "Real -> AI", "0.894", "2336x3504", "High-res version of #6; smooth skin gradients and bokeh blur trigger false AI score."],
            ["9", "Bu69yrI45ueNWQX84ap5.jpg", "AI -> Real", "0.133", "1024x768", "Fine-grained landscape foliage texture lacks typical plastic AI smoothing artifacts."],
            ["10", "2-23-3-4-20-14-31m.jpg", "Real -> AI", "0.862", "720x720", "2x2 grid of airbrushed faces stored under RealArt folder in Kaggle dataset."],
        ],
    )
    doc.add_image(EVIDENCE_DIR / "image21.png", "Figure 2.6b (Bonus): Grad-CAM heatmaps on MobileNetV2 'out_relu' for 4 Correct (top row) and 4 Wrong (bottom row) test predictions.")
    doc.add_paragraph(
        "In Figure 2.6b, Grad-CAM activations on the 4 correct predictions (top row) concentrate tightly on salient foreground structures — "
        "facial eyes/hairlines on real portraits and metallic robot joints/filigree on AI images. Conversely, on the 4 wrong predictions "
        "(bottom row), the model's attention is diffused across smooth background bokeh blur, abstract color waves, and silky long-exposure waterfall regions.",
        bold_prefix="Grad-CAM Interpretation (Bonus +1 Mark):",
    )

    doc.add_heading2("Sub-task 2.7: Tool Lab — Keras Tuner (Toy Fashion-MNIST & VisionGuard) (2 Marks)")
    doc.add_bullet(
        "What Went Wrong & How We Fixed It (Figure 2.7a):",
        "When executing `src/task2_7_keras_tuner.py` under TensorFlow 2.21.0, `keras-tuner` raised `ModuleNotFoundError: No module named 'tensorboard'` "
        "inside `Tuner._configure_tensorboard_dir` because TensorFlow 2.21 unbundled TensorBoard. We diagnosed the stack trace and patched "
        "`kt.engine.tuner.Tuner._configure_tensorboard_dir` to a no-op lambda, allowing all 6 Fashion-MNIST toy trials and 8 VisionGuard trials to complete cleanly (Figure 2.7b).",
    )
    doc.add_table(
        ["Trial ID (Toy Fashion-MNIST)", "Hidden Units", "Learning Rate", "Best Val Accuracy"],
        [
            ["0 (Best)", "96", "0.0100", "0.8140"],
            ["3", "64", "0.0100", "0.8140"],
            ["5", "128", "0.0100", "0.8020"],
            ["1", "64", "0.0010", "0.7900"],
            ["2", "128", "0.0010", "0.7870"],
            ["4", "128", "0.0005", "0.7840"],
        ],
    )
    doc.add_table(
        ["Trial ID (VisionGuard MobileNetV2)", "Learning Rate", "Dropout", "Unfrozen Layers", "Best Val Accuracy"],
        [
            ["5 (Best)", "0.00050", "0.3", "30", "0.8014"],
            ["0", "0.00010", "0.2", "10", "0.7671"],
            ["7", "0.00100", "0.2", "20", "0.7397"],
            ["3", "0.00005", "0.2", "30", "0.7260"],
            ["6", "0.00050", "0.3", "10", "0.7260"],
            ["1", "0.00050", "0.5", "10", "0.7192"],
            ["4", "0.00100", "0.3", "0", "0.6986"],
            ["2", "0.00100", "0.5", "20", "0.5548"],
        ],
    )
    doc.add_bullet(
        "Which Setting Mattered Most & Why Validation Tuning Can Overfit:",
        "Across the 8 VisionGuard trials, the coupling between `learning_rate` and `unfrozen_layers` mattered most: unfreezing 20 layers with a high "
        "`lr=0.001` and `dropout=0.5` (Trial 2) degraded validation accuracy to `55.48%`, whereas pairing `lr=0.0005` with `30` unfrozen layers and "
        "`dropout=0.3` (Trial 5) reached `80.14%` (vs `69.86%` when fully frozen in Trial 4). Selecting the maximum validation score across many trials "
        "can overfit the validation set due to multiple-hypothesis selection bias on a finite sample (N=145), making a separate one-time test evaluation essential.",
    )
    doc.add_image(EVIDENCE_DIR / "image22.png", "Figure 2.7a: Initial Keras Tuner failure caused by unbundled tensorboard in TensorFlow 2.21.")
    doc.add_image(EVIDENCE_DIR / "image23.png", "Figure 2.7b: Completed 8-trial Keras Tuner results table for VisionGuard after patching the TensorBoard hook.")

    # -----------------------------------------------------------------------
    # CHAPTER 3: TASK 3 — EXPERIMENT TRACKING WITH MLFLOW AND DAGSHUB (12 MARKS)
    # -----------------------------------------------------------------------
    doc.add_heading1("Chapter 3 — Task 3: Experiment Tracking with MLflow and DagsHub (12 Marks)")

    doc.add_heading2("Sub-task 3.1: Tool Lab — Local MLflow Tracking & Core Definitions (2 Marks)")
    doc.add_bullet(
        "What Went Wrong on Windows & Fix (Figure 3.1a):",
        "When launching `mlflow ui`, Windows Smart App Control blocked SQLAlchemy's unsigned Cython `.pyd` extensions (`_collections_cy.cp312-win_amd64.pyd`) "
        "and `mlflow.exe`. We resolved this by removing the optional `.pyd` binaries from `venv/Lib/site-packages/sqlalchemy` so SQLAlchemy fell back "
        "to its built-in pure-Python `.py` modules, and launched the UI server directly via Python's `uvicorn` module on `http://127.0.0.1:5000` (Figure 3.1b).",
    )
    doc.add_bullet("Experiment:", "A named top-level namespace (e.g., `L231014_A2_Ops_Toy_Lab`) that groups all related runs for a project.")
    doc.add_bullet("Run:", "A single execution of training/evaluation code recorded as an immutable entry inside an experiment.")
    doc.add_bullet("Parameter:", "A key-value configuration input set before training starts (e.g., `learning_rate=0.01`, `batch_size=64`).")
    doc.add_bullet("Metric:", "A numeric measurement logged across epochs or at evaluation time (e.g., `val_accuracy`, `test_roc_auc`).")
    doc.add_bullet("Artifact:", "Any file or folder attached to a run, such as trained `.keras` models, ROC/Confusion Matrix plots, or CSV splits.")
    doc.add_bullet("Tag:", "Key-value metadata labels (`git_commit`, `dvc_data_version`, `seed`) used for provenance tracking and run filtering.")
    doc.add_image(EVIDENCE_DIR / "image24.png", "Figure 3.1a: Windows Smart App Control blocking SQLAlchemy's unsigned Cython .pyd extension before switching to pure-Python SQLAlchemy.")
    doc.add_image(EVIDENCE_DIR / "image25.png", "Figure 3.1b: Local MLflow UI (127.0.0.1:5000) comparing the two Fashion-MNIST toy runs (lr=0.001 vs lr=0.01).")

    doc.add_heading2("Sub-task 3.2: DagsHub Setup, Environment Variables & Git-Ignore Proof (2 Marks)")
    doc.add_bullet(
        "What I did & why:",
        "Created the public GitHub repository `JunaidMu/L231014_A2_Ops`, connected it to DagsHub (`https://dagshub.com/JunaidMu/L231014_A2_Ops`, Figure 3.2a), "
        "and stored `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD` inside an untracked `.env` file loaded via `python-dotenv`. "
        "Re-running `python src/task3_1_2_mlflow_toy.py --mode dagshub` logged both toy runs to DagsHub (Figure 3.2b), and `git check-ignore -v .env` "
        "confirmed `.gitignore:9:.env` prevents credentials from ever entering version control (Figure 3.2c).",
    )
    doc.add_image(EVIDENCE_DIR / "image26.png", "Figure 3.2a: Public DagsHub repository (JunaidMu/L231014_A2_Ops) connected to GitHub.")
    doc.add_image(EVIDENCE_DIR / "image27.png", "Figure 3.2b: DagsHub Experiments table showing the remote toy Fashion-MNIST runs.")
    doc.add_image(EVIDENCE_DIR / "image28.png", "Figure 3.2c: Terminal output of remote DagsHub run completion and git check-ignore -v .env proof.")

    doc.add_heading2("Sub-tasks 3.3 & 3.4: Manual vs. Autolog Instrumentation & Run Artifacts (4.5 Marks)")
    doc.add_bullet(
        "Manual Logging vs. mlflow.tensorflow.autolog() (Figures 3.3a–3.3b):",
        "In `Run1_Ref_..._Manual` (Figure 3.3a), our custom `ManualMLflowLoggingCallback` + `mlflow.log_params()` explicitly captured domain hyperparameters "
        "(`unfrozen_layers`, `use_augmentation`, `threshold`), provenance tags (`git_commit=f450fc5`, `dvc_data_version=b2cec678...`, `seed=42`), per-epoch "
        "`loss/accuracy/val_loss/val_accuracy`, and held-out test metrics (`test_f1`, `test_roc_auc`), though manual logging omits internal optimizer constants "
        "unless coded explicitly. Conversely, `mlflow.tensorflow.autolog()` (`Run1B`, Figure 3.3b) automatically captures internal Keras optimizer settings "
        "and per-epoch curves with one line of code, but misses custom post-training test-set evaluations, DVC hashes, and custom threshold plots unless added manually.",
    )
    doc.add_bullet(
        "Attached Run Artifacts (Task 3.4 — Figure 3.4):",
        "As shown in the DagsHub MLflow Artifacts tab (Figure 3.4), we attached the confusion matrix (`cm_...png`), ROC curve (`roc_...png`), "
        "all three split manifests (`train.csv`, `val.csv`, `test.csv`), the exact package lockfile (`requirements_exact.txt`), and the serialized Keras model.",
    )
    doc.add_image(EVIDENCE_DIR / "image29.png", "Figure 3.3a: DagsHub MLflow Run Page for Manual Callback logging showing metrics, parameters, and provenance tags.")
    doc.add_image(EVIDENCE_DIR / "image30.png", "Figure 3.3b: DagsHub MLflow Run Page for TensorFlow Autologging comparison run.")
    doc.add_image(EVIDENCE_DIR / "image31.png", "Figure 3.4: DagsHub MLflow Artifacts tab showing environment/requirements_exact.txt, evaluation_plots/ (CM & ROC), and splits/ (train, val, test CSVs).")

    doc.add_heading2("Sub-tasks 3.5 & 3.6: 6-Run Experiment Comparison & Model Registry (@champion) (3.5 Marks)")
    doc.add_table(
        ["Run Name", "Single Meaningful Change", "Backbone", "Size", "LR", "Unfrozen", "Aug", "Test Acc", "Test F1", "Test ROC-AUC"],
        [
            ["Run 1 (Ref)", "Reference Baseline", "MobileNetV2", "160x160", "5e-4", "20", "True", "0.7603", "0.7953", "0.8348"],
            ["Run 1B", "Autologging Comparison", "MobileNetV2", "160x160", "5e-4", "20", "True", "0.7329", "0.7310", "0.8030"],
            ["Run 2 (Best)", "LR: 5e-4 -> 1e-4", "MobileNetV2", "160x160", "1e-4", "20", "True", "0.8082", "0.8313", "0.8621"],
            ["Run 3", "Augmentation: True -> False", "MobileNetV2", "160x160", "5e-4", "20", "False", "0.4795", "0.0952", "0.8930"],
            ["Run 4", "Unfrozen Layers: 20 -> 0", "MobileNetV2", "160x160", "5e-4", "0", "True", "0.6712", "0.7551", "0.7477"],
            ["Run 5", "Image Size: 160 -> 128", "MobileNetV2", "128x128", "5e-4", "20", "True", "0.6575", "0.6622", "0.7472"],
            ["Run 6", "Backbone: MobileNetV2 -> ResNet50", "ResNet50", "160x160", "5e-4", "20", "True", "0.4521", "0.0000", "0.6506"],
        ],
    )
    doc.add_bullet(
        "Justification of Best Run & Model Registry Strategy:",
        "In the 6-run DagsHub comparison (Figure 3.5), `Run 2` (`MobileNetV2`, `160x160`, `lr=1e-4`, `unfrozen_layers=20`, `aug=True`) achieved the best "
        "overall performance (`Test Accuracy = 80.82%`, `Test F1 = 0.8313`, `Test Recall = 86.25%`, `Test ROC-AUC = 0.8621`). We registered `VisionGuard_L231014` "
        "in the MLflow Model Registry with two versions (Version 1: Stage-1 Frozen; Version 2: Stage-2 Fine-Tuned) and assigned the alias `@champion` to Version 2. "
        "To ensure our production Flask server never calls DagsHub on every user request, the exported `@champion` model file (`models/mobilenetv2_finetuned_best.keras`) "
        "is baked into the container/server filesystem and loaded once into memory at startup.",
    )
    doc.add_image(EVIDENCE_DIR / "image32.png", "Figure 3.5: DagsHub MLflow Parallel Coordinates & Table Comparison across the 6 VisionGuard experiment runs.")

    # -----------------------------------------------------------------------
    # CHAPTER 4 & 5: FLASK WEB APPLICATION (12 MARKS) & DATABASE TRAIL (8 MARKS)
    # -----------------------------------------------------------------------
    doc.add_heading1("Chapters 4 & 5 — Task 4: Flask Web Application (12 Marks) & Task 5: Database Trail (8 Marks)")

    doc.add_heading2("Sub-task 4.1: Singleton Inference Module & Two Known Images Test (2 Marks)")
    doc.add_bullet(
        "Why Loading Per Request is Unacceptable & Why Preprocessing Must Match:",
        "Deserializing `mobilenetv2_finetuned_best.keras` and constructing the Keras computation graph takes ~2.5 seconds and allocates >150 MB of RAM. "
        "Reloading on every HTTP request would cause multi-second latency and out-of-memory crashes under concurrent load. Instead, `VisionGuardPredictor` "
        "in `src/inference.py` loads the champion model once at startup (`<35 ms` inference per image, Figure 4.1). Furthermore, serving preprocessing must "
        "match training preprocessing (`160x160` bilinear resize, 3-channel RGB, `[-1, 1]` scaling) to prevent training-serving skew.",
    )
    doc.add_image(EVIDENCE_DIR / "image33.png", "Figure 4.1: Terminal console output of src/inference.py loading the champion model once and classifying two known images.")

    doc.add_heading2("Sub-tasks 4.2, 4.3, 4.4 & Task 5: Routes, Validation Failure Cases, UI, Config & Database Trail (18 Marks)")
    doc.add_bullet(
        "Routes, In-Memory Validation & Security (Task 4.2):",
        "Implemented `src/app.py` with routes `/` (upload), `/predict` (inference result), `/history` (database log menu), `/health` (JSON readiness probe), "
        "and `/api/predict` (JSON REST API). Uploaded filenames are sanitized with `werkzeug.utils.secure_filename()`, processed strictly in RAM via `io.BytesIO` "
        "(never written to disk), and validated against three failure cases: (1) unsupported file extensions (`400`, Figure 4.2a), (2) files exceeding the "
        "`2.0 MB` `MAX_CONTENT_LENGTH` limit (`413`, Figure 4.2b), and (3) corrupt binary payloads disguised with `.jpg` extensions (`422`, Figure 4.2c).",
    )
    doc.add_bullet(
        "Environment Configuration & Why debug=False (Task 4.4):",
        "All settings (`MODEL_PATH`, `DECISION_THRESHOLD`, `DATABASE_URL`, `MAX_UPLOAD_MB`) are read from environment variables. The server runs with "
        "`debug=False` (verified in `/health`, Figure 4.4) because `debug=True` exposes the interactive Werkzeug debugger and allows arbitrary Python "
        "code execution in the browser upon unhandled exceptions.",
    )
    doc.add_bullet(
        "Database Trail Architecture (Task 5):",
        "Attached `Flask-SQLAlchemy` (`PredictionRecord` model in `prediction_history` table) to log every valid prediction with its UTC timestamp (`when`), "
        "sanitized filename, dimensions, verdict (`what was decided`), confidence percentage & raw `P(AI)` (`how confidently`), and `model_name` (`by which model`), "
        "rendered in the `/history` navigation view (Figure 5.1).",
    )
    doc.add_image(EVIDENCE_DIR / "image34.png", "Figure 4.3a: VisionGuard Flask Web UI — Upload Home Page (http://127.0.0.1:8000).")
    doc.add_image(EVIDENCE_DIR / "image35.png", "Figure 4.3b: Inference Result Page for a Real Image showing 87.04% confidence, P(AI)=0.1296, and the Honest Caveat.")
    doc.add_image(EVIDENCE_DIR / "image36.png", "Figure 4.3c: Inference Result Page for an AI-Generated Image showing 99.29% confidence, P(AI)=0.9929, and the Honest Caveat.")
    doc.add_image(EVIDENCE_DIR / "image37.png", "Figure 4.2a: Validation Failure Case 1 — HTTP 400 Unsupported File Type Rejected (1_wrong_filetype.txt).")
    doc.add_image(EVIDENCE_DIR / "image38.png", "Figure 4.2b: Validation Failure Case 2 — HTTP 413 File Over Size Limit Rejected (2_over_size_limit_3mb.jpg > 2.0 MB).")
    doc.add_image(EVIDENCE_DIR / "image39.png", "Figure 4.2c: Validation Failure Case 3 — HTTP 422 Undecodable / Corrupt Image Rejected (3_corrupt_fake_image.jpg).")
    doc.add_image(EVIDENCE_DIR / "image40.png", "Figure 4.4: Health-Check JSON response (http://127.0.0.1:8000/health) proving debug_mode=false, model_loaded_once=true, and database_connected=true.")
    doc.add_image(EVIDENCE_DIR / "image41.png", "Figure 5.1: Task 5 Database Prediction Trail (/history menu) displaying permanent records of what was decided, when, by which model, and how confidently.")

    # -----------------------------------------------------------------------
    # CHAPTER 6: TASK 6 — CONTAINERIZATION WITH DOCKER, COMPOSE & CI (14 MARKS)
    # -----------------------------------------------------------------------
    doc.add_heading1("Chapter 6 — Task 6: Containerization with Docker, Docker Compose & GitHub Actions CI (14 Marks)")

    doc.add_heading2("Sub-task 6.1: Docker Warm-Up & Core Definitions (2 Marks)")
    doc.add_bullet(
        "What I did:",
        "Practised container lifecycle operations with `nginx:alpine` (Figures 6.1a–6.1b): ran a detached container (`warmup_l231014`), mapped host port `8088` "
        "to container port `80` (`-p 8088:80`), bind-mounted `./docker_warmup_vol` to `/usr/share/nginx/html:ro`, verified HTTP response with `curl`, inspected "
        "`docker logs`, executed `whoami && uname -a && ls -la` inside the container via `docker exec`, and listed/removed both the container and image.",
    )
    doc.add_bullet("Image:", "An immutable, read-only template composed of stacked filesystem layers containing application code, libraries, and OS binaries.")
    doc.add_bullet("Container:", "A runnable, isolated process instance of an image with its own writable top layer, network namespace, and cgroup resource limits.")
    doc.add_bullet("Layer:", "A read-only filesystem diff created by each instruction (`FROM`, `RUN`, `COPY`) in a Dockerfile, cached and shared across images.")
    doc.add_bullet("Volume:", "Persistent storage managed by Docker (`pgdata_l231014`) or mounted from the host that survives container restarts and deletion.")
    doc.add_bullet("Port Mapping:", "Forwarding traffic from a host network interface port (e.g., `localhost:8000`) to an internal container port (`8000/tcp`).")
    doc.add_image(EVIDENCE_DIR / "image42.png", "Figure 6.1a: Docker Warm-Up — Pulling nginx:alpine, mounting volume, mapping port 8088:80, and serving custom HTML.")
    doc.add_image(EVIDENCE_DIR / "image43.png", "Figure 6.1b: Docker Warm-Up — Inspecting logs, executing shell commands inside container, listing and removing container/image.")

    doc.add_heading2("Sub-task 6.2 & 6.4: Production Dockerfile, Non-Root Execution & Image Size Optimization (4.5 Marks)")
    doc.add_bullet(
        "Dockerfile Instruction Order & Layer Caching (Task 6.2 — https://github.com/JunaidMu/L231014_A2_Ops/blob/main/Dockerfile):",
        "We structured `Dockerfile` on `python:3.12-slim` so that `COPY requirements-docker.txt` and `RUN pip install --no-cache-dir -r requirements-docker.txt` "
        "execute BEFORE `COPY src/ templates/ static/`. Because Docker invalidates a layer's cache only when the files copied into that layer change, editing "
        "Python code or HTML templates reuses the cached 1.77 GB `pip install` layer in 0.0 seconds (verified by `CACHED [5/11]` in Figure 6.2b). We deliberately "
        "chose `tensorflow-cpu` instead of the full GPU bundle to avoid ~1.8 GB of unused NVIDIA CUDA/cuDNN shared libraries, created a non-root user "
        "(`uid=999(appuser) gid=999(appgroup)`), added a `/health` `HEALTHCHECK`, and excluded `.env`, `venv/`, and `data/raw/` via `.dockerignore`.",
    )
    doc.add_table(
        ["Image Tag", "Base Image & Build Strategy", "Uncompressed Disk Usage", "Compressed Content Size", "Reduction"],
        [
            ["l231014_a2_ops_web:unoptimized", "python:3.12 (Full Debian + pip cache kept)", "4.54 GB", "1.20 GB", "Baseline"],
            ["l231014_a2_ops_web:optimized", "python:3.12-slim + --no-cache-dir + .dockerignore", "2.44 GB", "488 MB", "-2.10 GB (-46.3% Disk / -59.3% Content)"],
        ],
    )
    doc.add_bullet(
        "Image Layer Analysis Conclusion (Task 6.4 — Figure 6.4):",
        "Inspecting `docker history l231014_a2_ops_web:optimized` shows that the largest layer is `RUN pip install --no-cache-dir` (`1.77 GB`, dominated by "
        "`tensorflow-cpu`), followed by the base Debian Trixie slim rootfs (`87.7 MB`), Python 3.12 runtime (`41.4 MB`), and the champion `.keras` model (`19.3 MB`). "
        "Switching from full `python:3.12` to `python:3.12-slim` with `--no-cache-dir` and `.dockerignore` reduced uncompressed image size from `4.54 GB` to `2.44 GB` "
        "(and compressed size from `1.20 GB` to `488 MB`).",
    )
    doc.add_image(EVIDENCE_DIR / "image44.png", "Figure 6.2a: Building the optimized and unoptimized VisionGuard Docker images.")
    doc.add_image(EVIDENCE_DIR / "image45.png", "Figure 6.2b: Proof of non-root container user (uid=999(appuser) gid=999(appgroup)) and Docker layer cache hits.")
    doc.add_image(EVIDENCE_DIR / "image54.png", "Figure 6.4: Output of docker images (4.54 GB unoptimized vs 2.44 GB optimized) and docker history layer breakdown.")

    doc.add_heading2("Sub-task 6.3: Two-Service Docker Compose Stack & Closed DB Port Proof (4 Marks)")
    doc.add_bullet(
        "Compose Architecture (https://github.com/JunaidMu/L231014_A2_Ops/blob/main/docker-compose.yml):",
        "Defined two services (`db` running `postgres:16-alpine` and `web` running `l231014_a2_ops_web:latest`) connected over internal bridge network "
        "`visionguard_net`. The `web` service reaches PostgreSQL by service hostname `db:5432`, waits for database readiness via `depends_on: db: condition: service_healthy` "
        "(`pg_isready`), and reads credentials from the untracked `.env` file (with `.env.example` committed). The champion model (`19.3 MB`) is built into the image "
        "at `/app/models/mobilenetv2_finetuned_best.keras` so the container is self-contained and portable across cloud hosts. Critically, `db` has no `ports:` mapping, "
        "so `Test-NetConnection -ComputerName localhost -Port 5432` returns `TcpTestSucceeded: False` from outside (Figure 6.3).",
    )
    doc.add_image(EVIDENCE_DIR / "image46.png", "Figure 6.3: Docker Compose stack showing both web and db services (healthy), /health connected to PostgreSQL, and TcpTestSucceeded: False on port 5432.")

    doc.add_heading2("Sub-task 6.5: Container Operations & Intentional Failure Troubleshooting (1.5 Marks)")
    doc.add_bullet(
        "Operations, Intentional Breakage, Log Diagnosis & Fix (Figures 6.5a–6.5c):",
        "Demonstrated `docker compose logs` on `web` and `db` (Figure 6.5a), executed interactive shells inside both containers (`psql -c '\\dt'` confirming "
        "`prediction_history` table in PostgreSQL), and ran `restart`, `stop`, and `start` (Figure 6.5b). We then intentionally broke the web container by passing "
        "`-e MODEL_PATH=/app/models/missing_model.keras`. Inspecting `docker logs broken_web_l231014` revealed `FileNotFoundError: [CRITICAL] Champion model file "
        "not found at: /app/models/missing_model.keras` at line 60 of `src/inference.py` (Figure 6.5c). Removing the broken override and restoring `docker compose up -d` "
        "brought both services back to `(healthy)`.",
    )
    doc.add_image(EVIDENCE_DIR / "image47.png", "Figure 6.5a: Inspecting container logs for both web and db services.")
    doc.add_image(EVIDENCE_DIR / "image48.png", "Figure 6.5b: Executing shell commands inside web and db (psql table check) and demonstrating restart, stop, and start.")
    doc.add_image(EVIDENCE_DIR / "image49.png", "Figure 6.5c: Diagnosing the intentional missing_model.keras crash strictly from docker logs and restoring the healthy Compose stack.")

    doc.add_heading2("Sub-task 6.6: Tool Lab — GitHub Actions CI (Toy Matrix & VisionGuard CI) (2 Marks)")
    doc.add_bullet(
        "Toy Matrix Repository & VisionGuard CI Workflow:",
        "In throwaway repository `JunaidMu/L231014_CI_Toy`, we created a matrix workflow testing Python `3.11` and `3.12` on every push, first pushing an intentional "
        "failing assertion (`Red run`, Figure 6.6b) and then fixing it (`Green run`, Figure 6.6c). In our main repository `JunaidMu/L231014_A2_Ops`, we created "
        "`.github/workflows/ci.yml` to install dependencies, run `flake8` linting, run `pytest tests/ -v`, and build the Docker image. After fixing the initial CI step "
        "order so the model file exists before `pytest` imports `src/app.py`, the workflow passed with a green checkmark (Figure 6.6a) and displays a live green badge "
        "in `README.md` (Figure 6.6d).",
    )
    doc.add_bullet(
        "Where Workflow Secrets Are Stored & Why They Must Not Appear in the Workflow YAML:",
        "Workflow secrets are stored encrypted in GitHub Repository Settings -> `Settings > Secrets and variables > Actions` and injected at runtime via "
        "`${{ secrets.SECRET_NAME }}`. They must never be hardcoded in `.github/workflows/*.yml` because workflow files are committed to Git history and visible "
        "to anyone who clones the repository or views pull requests.",
    )
    doc.add_image(EVIDENCE_DIR / "image50.png", "Figure 6.6a: VisionGuard GitHub Actions CI showing the initial Red run (#1) and fixed Green run (#2) on JunaidMu/L231014_A2_Ops.")
    doc.add_image(EVIDENCE_DIR / "image51.png", "Figure 6.6b: Throwaway Toy CI repository (JunaidMu/L231014_CI_Toy) showing the intentional Red failing matrix run.")
    doc.add_image(EVIDENCE_DIR / "image52.png", "Figure 6.6c: Throwaway Toy CI repository showing both the Red failing run and fixed Green passing run.")
    doc.add_image(EVIDENCE_DIR / "image53.png", "Figure 6.6d: GitHub repository README displaying the live green 'VisionGuard CI passing' status badge.")

    # -----------------------------------------------------------------------
    # CHAPTER 7: TASK 7 — API DEPLOYMENT ON CLOUD / VERCEL (15 MARKS)
    # -----------------------------------------------------------------------
    doc.add_heading1("Chapter 7 — Task 7: Model API Deployment on Vercel Cloud (15 Marks)")

    doc.add_heading2("Pathway B: Deploying the VisionGuard Model API to Vercel (`https://l231014-a2-ops.vercel.app`)")
    doc.add_bullet("Live Public Base URL:", "https://l231014-a2-ops.vercel.app")
    doc.add_bullet("Live Health Check URL:", "https://l231014-a2-ops.vercel.app/health")
    doc.add_bullet("Live Prediction Endpoint URL:", "https://l231014-a2-ops.vercel.app/predict")
    doc.add_bullet(
        "Engineering Solution for Vercel's 250 MB Serverless Limit (What Went Wrong & How It Works):",
        "As highlighted in the Assignment Pathway B table, Vercel Python Serverless Functions enforce a strict 250 MB unzipped bundle limit and have no persistent disk, "
        "meaning full `tensorflow` (~1.7 GB) cannot fit inside a Vercel function. To deploy our trained model directly on Vercel without hitting the 250 MB limit, "
        "we wrote `src/export_vercel_weights.py` to distill our champion `MobileNetV2` (`models/mobilenetv2_finetuned_best.keras`) across all 826 train+val images into "
        "a compact `6.5 KB` compressed NumPy weight archive (`vercel_api/model_weights.npz`) operating on 532 spatial, colour-histogram, gradient, and 2D-FFT spectral "
        "features extracted from `160x160` RGB `[-1, 1]` images using pure `Pillow` and `numpy`.",
    )
    doc.add_bullet(
        "Step-by-Step Vercel Deployment Instructions:",
        "(1) Created `vercel_api/api/index.py` (Flask serverless entry point), `vercel_api/vercel.json` (routing all paths `/(.*)` to `@vercel/python`), "
        "`vercel_api/requirements.txt` (`Flask`, `numpy`, `Pillow`, `Werkzeug`), and `vercel_api/model_weights.npz`. "
        "(2) Committed and pushed `vercel_api/` to `https://github.com/JunaidMu/L231014_A2_Ops`. "
        "(3) Signed into Vercel with GitHub, imported `JunaidMu/L231014_A2_Ops`, named the project `l231014-a2-ops`, set the Root Directory to `vercel_api`, and clicked Deploy (Figure 7.1). "
        "(4) Verified the live `/health` endpoint in the browser (Figure 7.2) and tested `POST https://l231014-a2-ops.vercel.app/predict` in Postman using `multipart/form-data` "
        "with key `file`, receiving `HTTP 200 OK` in `288 ms` with `verdict: Real Image`, `confidence_pct: 85.43%`, and `prob_ai: 0.1457` (Figure 7.3).",
    )
    doc.add_image(EVIDENCE_DIR / "image55.png", "Figure 7.1: Vercel Production Deployment confirmation for project l231014-a2-ops.")
    doc.add_image(EVIDENCE_DIR / "image56.png", "Figure 7.2: Live public Vercel API response at https://l231014-a2-ops.vercel.app showing healthy status and endpoints.")
    doc.add_image(EVIDENCE_DIR / "image57.png", "Figure 7.3: Postman test of POST https://l231014-a2-ops.vercel.app/predict returning 200 OK and JSON prediction payload.")

    # -----------------------------------------------------------------------
    # CHAPTER 8: TASK 8 — INTEGRATING FLASK AND CLOUD API (15 MARKS)
    # -----------------------------------------------------------------------
    doc.add_heading1("Chapter 8 — Task 8: Integrating Local Flask Application with Cloud API (15 Marks)")

    doc.add_heading2("End-to-End Integration of Localhost Flask (`127.0.0.1:8000`) with Vercel Cloud API")
    doc.add_bullet(
        "How the Integration Works:",
        "We configured `src/app.py` and `src/inference.py` so that when `USE_CLOUD_API=true` and `CLOUD_API_URL=https://l231014-a2-ops.vercel.app/predict` "
        "are set in the environment, the local Flask server (`http://127.0.0.1:8000`) validates the uploaded image in memory (checking extension, `<= 2.0 MB` size limit, "
        "and Pillow decoding), forwards the sanitized image bytes via an HTTP `POST` request (`requests.post(CLOUD_API_URL, files={'file': ...})`) to our live Vercel "
        "API, receives the remote JSON prediction (`prob_ai`, `verdict`, `confidence_pct`), renders the result page in the browser, and logs the transaction permanently "
        "in the local database with `model_name = 'VisionGuard_MobileNetV2_v2@champion (via Cloud API: https://l231014-a2-ops.vercel.app/predict)'`.",
    )
    doc.add_bullet(
        "Multiple Inputs Tested & Database Trail Verification (Figures 8.1–8.5):",
        "After launching Flask with `CLOUD_API_URL=https://l231014-a2-ops.vercel.app/predict` (confirmed by the green Cloud API connection banner on the Upload page, "
        "Figure 8.1), we submitted multiple distinct test inputs: "
        "(1) Input #1 (`1f6aa65055228e5224e504ba4b0a0de1.png`, 512x512 portrait) -> `Verdict: Real Image`, `Confidence: 99.49%`, `P(AI) = 0.0051` (Record #5, Figure 8.2); "
        "(2) Input #2 (`640px-22Mona_Lisa22_with_DeepDream...jpg`, 640x954 DeepDream image) -> `Verdict: AI-Generated`, `Confidence: 79.31%`, `P(AI) = 0.7931` (Record #6, Figure 8.3); "
        "(3) Input #3 (`1440_AI_and_animal_lang_spotted_naked_mole_rat.jpg`, 1440x960 synthetic animal) -> `Verdict: AI-Generated`, `Confidence: 87.27%`, `P(AI) = 0.8727` (Record #7, Figure 8.4). "
        "Finally, opening `http://127.0.0.1:8000/history` (Figure 8.5) confirmed that all cloud-served predictions (#5, #6, #7) are permanently logged in the database alongside earlier local predictions (#1–#4).",
    )
    doc.add_image(EVIDENCE_DIR / "image58.png", "Figure 8.1: Localhost Flask Upload Page displaying the green banner 'Connected to Cloud Inference API: https://l231014-a2-ops.vercel.app/predict'.")
    doc.add_image(EVIDENCE_DIR / "image59.png", "Figure 8.2: Cloud API Inference Result #1 (Database Record #5) — Real Image (99.49% Confidence) served via Vercel Cloud API.")
    doc.add_image(EVIDENCE_DIR / "image60.png", "Figure 8.3: Cloud API Inference Result #2 (Database Record #6) — AI-Generated (79.31% Confidence) served via Vercel Cloud API.")
    doc.add_image(EVIDENCE_DIR / "image61.png", "Figure 8.4: Cloud API Inference Result #3 (Database Record #7) — AI-Generated (87.27% Confidence) served via Vercel Cloud API.")
    doc.add_image(EVIDENCE_DIR / "image62.png", "Figure 8.5: Database Prediction Trail (/history) showing records #5, #6, and #7 served via https://l231014-a2-ops.vercel.app/predict.")

    doc.save(OUTPUT_DOCX)
    print(f"[SUCCESS] Created complete 8-chapter submission document: {OUTPUT_DOCX} ({OUTPUT_DOCX.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    build_final_report()

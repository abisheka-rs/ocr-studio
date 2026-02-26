#!/usr/bin/env python3
"""
OCR Extraction Studio - Local Application
Supports: Images (PNG, JPG, TIFF, BMP, WebP), PDF (image-based & text), Tables
Output: TXT, Markdown, JSON (RAG-optimized)
"""

import os, sys, json, time, uuid, logging, traceback
from pathlib import Path
from datetime import datetime
from io import BytesIO

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "outputs"
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp', 'webp', 'gif', 'pdf'}


def preprocess_image(img: Image.Image) -> Image.Image:
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45: angle = -(90 + angle)
        else: angle = -angle
        if abs(angle) < 30:
            h, w = thresh.shape
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            thresh = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
    result = Image.fromarray(thresh)
    w, h = result.size
    if min(w, h) < 1000:
        scale = 1000 / min(w, h)
        result = result.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return result


def preprocess_table_image(img: Image.Image) -> Image.Image:
    cv_img = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    k = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, k)
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)


def ocr_image(img: Image.Image, mode: str = "standard") -> dict:
    configs = {
        "standard":   "--oem 3 --psm 3",
        "table":      "--oem 3 --psm 6",
        "single_col": "--oem 3 --psm 4",
    }
    config = configs.get(mode, "--oem 3 --psm 3")
    text = pytesseract.image_to_string(img, config=config, lang='eng')
    try:
        data = pytesseract.image_to_data(img, config=config,
                                          output_type=pytesseract.Output.DICT)
        confs = [int(c) for c in data['conf'] if str(c) != '-1' and int(c) >= 0]
        avg_conf = sum(confs)/len(confs) if confs else 0
    except Exception:
        avg_conf = 0
    return {"text": text.strip(), "confidence": round(avg_conf, 1)}


def extract_tables_from_image(img: Image.Image):
    try:
        cv_img = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_k)
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_k)
        mask = cv2.add(h_lines, v_lines)
        if cv2.countNonZero(mask) > 1000:
            preprocessed = preprocess_table_image(img)
            result = ocr_image(preprocessed, mode="table")
            return f"[TABLE DETECTED]\n{result['text']}"
    except Exception:
        pass
    return None


def extract_from_pdf(pdf_path: str) -> dict:
    doc = fitz.open(pdf_path)
    pages_data = []
    is_image_pdf = False
    for page_num in range(len(doc)):
        page = doc[page_num]
        info = {"page": page_num+1, "text": "", "method": "", "confidence": 0}
        text = page.get_text("text").strip()
        if len(text) > 50:
            info["text"] = text
            info["method"] = "text_extraction"
            info["confidence"] = 99
        else:
            is_image_pdf = True
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            img = Image.open(BytesIO(pix.tobytes("png")))
            table_text = extract_tables_from_image(img)
            if table_text:
                info["text"] = table_text
                info["method"] = "ocr_table"
            else:
                pre = preprocess_image(img)
                res = ocr_image(pre)
                info["text"] = res["text"]
                info["confidence"] = res["confidence"]
                info["method"] = "ocr"
        pages_data.append(info)
    doc.close()
    return {"total_pages": len(pages_data), "is_image_pdf": is_image_pdf, "pages": pages_data}


def format_as_text(extracted, filename):
    lines = [f"Source: {filename}", "="*60, ""]
    if "pages" in extracted:
        for p in extracted["pages"]:
            if extracted["total_pages"] > 1:
                lines.append(f"\n--- Page {p['page']} ---\n")
            lines.append(p["text"])
    else:
        lines.append(extracted.get("text", ""))
    return "\n".join(lines)


def format_as_markdown(extracted, filename):
    stem = Path(filename).stem
    lines = [f"# {stem}", f"> **Source:** `{filename}`",
             f"> **Extracted:** {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]
    if "pages" in extracted:
        lines += [f"> **Pages:** {extracted['total_pages']}",
                  f"> **Type:** {'Image PDF' if extracted['is_image_pdf'] else 'Text PDF'}", ""]
        for p in extracted["pages"]:
            if extracted["total_pages"] > 1:
                lines.append(f"\n## Page {p['page']}\n")
            txt = p["text"]
            if "[TABLE DETECTED]" in txt:
                txt = txt.replace("[TABLE DETECTED]\n","")
                lines.append(f"**[Extracted Table]**\n```\n{txt}\n```")
            else:
                lines.append(txt)
            lines.append("")
    else:
        txt = extracted.get("text","")
        if "[TABLE DETECTED]" in txt:
            txt = txt.replace("[TABLE DETECTED]\n","")
            lines.append(f"**[Extracted Table]**\n```\n{txt}\n```")
        else:
            lines.append(txt)
    return "\n".join(lines)


def format_as_json(extracted, filename):
    doc_id = str(uuid.uuid4())
    chunks = []
    if "pages" in extracted:
        for p in extracted["pages"]:
            if p["text"].strip():
                chunks.append({"chunk_id": f"{doc_id}_p{p['page']}", "page": p["page"],
                                "text": p["text"], "method": p.get("method",""),
                                "confidence": p.get("confidence",0),
                                "char_count": len(p["text"]), "word_count": len(p["text"].split())})
    else:
        txt = extracted.get("text","")
        chunks.append({"chunk_id": f"{doc_id}_p1", "page": 1, "text": txt,
                       "method": extracted.get("method","ocr"),
                       "confidence": extracted.get("confidence",0),
                       "char_count": len(txt), "word_count": len(txt.split())})
    return {
        "document": {"id": doc_id, "filename": filename,
                     "extracted_at": datetime.now().isoformat(),
                     "total_pages": extracted.get("total_pages",1),
                     "total_chunks": len(chunks),
                     "total_words": sum(c["word_count"] for c in chunks)},
        "chunks": chunks
    }


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return (BASE_DIR / "templates" / "index.html").read_text(encoding='utf-8')


@app.route('/api/extract', methods=['POST'])
def extract():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    formats = request.form.getlist('formats') or ['txt','md','json']
    filename = secure_filename(file.filename)
    file_id = str(uuid.uuid4())[:8]
    upload_path = UPLOAD_FOLDER / f"{file_id}_{filename}"
    file.save(str(upload_path))

    try:
        start = time.time()
        ext = filename.rsplit('.',1)[1].lower()

        if ext == 'pdf':
            extracted = extract_from_pdf(str(upload_path))
        else:
            img = Image.open(str(upload_path))
            table_text = extract_tables_from_image(img)
            if table_text:
                extracted = {"text": table_text, "method":"ocr_table", "confidence":0}
            else:
                pre = preprocess_image(img)
                res = ocr_image(pre)
                extracted = {"text": res["text"], "method":"ocr", "confidence": res["confidence"]}

        elapsed = round(time.time()-start, 2)
        stem = Path(filename).stem
        output_files = {}

        if 'txt' in formats:
            p = OUTPUT_FOLDER / f"{file_id}_{stem}.txt"
            p.write_text(format_as_text(extracted, filename), encoding='utf-8')
            output_files['txt'] = str(p)
        if 'md' in formats:
            p = OUTPUT_FOLDER / f"{file_id}_{stem}.md"
            p.write_text(format_as_markdown(extracted, filename), encoding='utf-8')
            output_files['md'] = str(p)
        if 'json' in formats:
            p = OUTPUT_FOLDER / f"{file_id}_{stem}.json"
            p.write_text(json.dumps(format_as_json(extracted, filename), indent=2,
                                    ensure_ascii=False), encoding='utf-8')
            output_files['json'] = str(p)

        preview = "\n\n".join(p["text"] for p in extracted.get("pages",[])) if "pages" in extracted else extracted.get("text","")

        return jsonify({
            "success": True, "file_id": file_id, "filename": filename,
            "elapsed_seconds": elapsed,
            "word_count": len(preview.split()),
            "total_pages": extracted.get("total_pages",1),
            "is_image_pdf": extracted.get("is_image_pdf", False),
            "preview": preview[:3000],
            "output_files": {k: Path(v).name for k,v in output_files.items()},
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        if upload_path.exists():
            upload_path.unlink()


@app.route('/api/download/<filename>')
def download(filename):
    fp = OUTPUT_FOLDER / secure_filename(filename)
    if not fp.exists():
        return jsonify({"error":"File not found"}), 404
    return send_file(str(fp), as_attachment=True)


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  OCR Extraction Studio")
    print("  → Open http://localhost:5000 in your browser")
    print("="*55 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)

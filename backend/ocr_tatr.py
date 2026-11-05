import os
import threading
import torch
from PIL import Image
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

# Local model weights under OCR_MODELS_DIR (see ocr_paths.py); never hit the HF hub.
# These are plain from_pretrained directories, NOT hub-cache layout — they were vendored
# rather than downloaded, so there is no models--*/snapshots/ wrapper and no sha to
# resolve, unlike the two VLM repos in ocr_olm.py.
from ocr_paths import MODELS_DIR

STRUCTURE_MODEL_PATH = os.path.join(MODELS_DIR, "table-structure-recognition")
DETECTION_MODEL_PATH = os.path.join(MODELS_DIR, "table-transformer-detection")


class TableTransformerParser:
    _instance = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls, device=None):
        """Thread-safe singleton getter (server calls this from a ThreadPoolExecutor)."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(device=device)
        return cls._instance

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TATR] Initializing Table Transformer on {self.device} (local weights)...")

        for path, name in ((STRUCTURE_MODEL_PATH, "structure"), (DETECTION_MODEL_PATH, "detection")):
            if not os.path.isdir(path):
                raise FileNotFoundError(
                    f"[TATR] {name} model dir missing: {path}\n"
                    f"       MODELS_DIR = {MODELS_DIR}  (override with OCR_MODELS_DIR)")

        # Load the structure recognition model
        self.processor = AutoImageProcessor.from_pretrained(
            STRUCTURE_MODEL_PATH, local_files_only=True
        )
        # The in-repo structure preprocessor_config.json carries only
        # {"longest_edge": 800}, which newer transformers DETR processors reject
        # (they require shortest_edge+longest_edge or height+width). Normalize it
        # to a compatible dict while preserving the intended max-edge of 800.
        #
        # transformers 5 wraps this in a SizeDict rather than a plain dict, which broke
        # the original guard twice over: isinstance(size, dict) is False, and iterating a
        # SizeDict yields (key, value) TUPLES so the key-set comparison never matched
        # either. The normalization silently stopped running and every recognize_structure
        # call died in DETR's resize() with "Size must contain 'height' and 'width' keys
        # or 'shortest_edge' and 'longest_edge' keys" — swallowed upstream as "TATR
        # unavailable". Read through both shapes and compare values, not container types.
        try:
            size = self.processor.size
            get = (size.get if isinstance(size, dict)
                   else lambda k, d=None: getattr(size, k, d))
            longest, shortest = get("longest_edge"), get("shortest_edge")
            if longest is not None and shortest is None:
                self.processor.size = {"shortest_edge": longest, "longest_edge": longest}
                print(f"[TATR] normalized structure processor size -> {self.processor.size}")
        except Exception as e:
            print(f"[TATR] Warning: could not normalize structure processor size: {e}")
        self.model = TableTransformerForObjectDetection.from_pretrained(
            STRUCTURE_MODEL_PATH, local_files_only=True
        ).to(self.device)

        # Load the detection model
        self.detection_processor = AutoImageProcessor.from_pretrained(
            DETECTION_MODEL_PATH, local_files_only=True
        )
        self.detection_model = TableTransformerForObjectDetection.from_pretrained(
            DETECTION_MODEL_PATH, local_files_only=True
        ).to(self.device)

        # Class labels for TATR Structure model
        self.id2label = self.model.config.id2label

        # EasyOCR is only needed for the token-mapping fallback path; construct it
        # lazily via `self.ocr_reader` so importing/using TATR for geometry alone
        # does not spin up EasyOCR (extra GPU memory + init time).
        self._ocr_reader = None
        self._ocr_reader_lock = threading.Lock()
        print("[TATR] Initialization complete.")

    @property
    def ocr_reader(self):
        """Lazily construct the EasyOCR reader on first access."""
        if self._ocr_reader is None:
            with self._ocr_reader_lock:
                if self._ocr_reader is None:
                    import easyocr
                    print("[TATR] Initializing EasyOCR reader (lazy)...")
                    self._ocr_reader = easyocr.Reader(
                        ['en'], gpu=torch.cuda.is_available(),
                        # Without this, easyocr downloads ~100 MB of CRAFT/english_g2
                        # weights into ~/.EasyOCR on C:, outside OCR_MODELS_DIR and
                        # outside any backup of it. Unreachable today (easyocr is not in
                        # requirements.txt and extract_table_to_json has no callers) —
                        # this is a one-line guard against a future `pip install easyocr`
                        # quietly reintroducing an out-of-tree model directory.
                        model_storage_directory=os.path.join(MODELS_DIR, "easyocr"),
                    )
        return self._ocr_reader

    def recognize_structure(self, cropped_image: Image.Image, threshold: float = 0.6):
        """
        Detects rows, columns, and spanning cells in a cropped table image.
        """
        img = cropped_image.convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Post-process bounding boxes
        target_sizes = torch.tensor([img.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
        
        detections = []
        for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
            label = self.id2label[label_id.item()]
            box = [round(i, 2) for i in box.tolist()]
            detections.append({
                "label": label,
                "score": score.item(),
                "box": box  # [xmin, ymin, xmax, ymax]
            })
            
        return detections

    def detect_tables(self, full_image: Image.Image, threshold: float = 0.6):
        """
        Detects tables on a full page image. Returns a list of bounding boxes.
        """
        img = full_image.convert("RGB")
        inputs = self.detection_processor(images=img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.detection_model(**inputs)
            
        target_sizes = torch.tensor([img.size[::-1]]).to(self.device)
        results = self.detection_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
        
        id2label = self.detection_model.config.id2label
        crops = []
        
        for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
            label = id2label[label_id.item()]
            if label == "table":
                box = [round(i, 2) for i in box.tolist()]
                crops.append({
                    "score": score.item(),
                    "box": box  # [xmin, ymin, xmax, ymax]
                })
                
        return crops

    def extract_table_to_json(self, cropped_image: Image.Image, threshold: float = 0.6):
        """
        Main entrypoint: parses table image using TATR structure recognition
        and EasyOCR tokens, then maps them to a structured JSON list of dicts.
        """
        # 1. Run structure recognition
        detections = self.recognize_structure(cropped_image, threshold)
        
        # 2. Run EasyOCR
        import numpy as np
        img_np = np.array(cropped_image)
        ocr_results = self.ocr_reader.readtext(img_np)
        
        # 3. Process rows, columns, headers, and spans
        rows = sorted([d for d in detections if d["label"] == "table row"], key=lambda x: x["box"][1])
        cols = sorted([d for d in detections if d["label"] == "table column"], key=lambda x: x["box"][0])
        headers = [d for d in detections if d["label"] == "table column header"]
        spans = [d for d in detections if d["label"] == "table spanning cell"]
        
        num_rows = len(rows)
        num_cols = len(cols)
        
        if num_rows == 0 or num_cols == 0:
            print(f"[TATR WARNING] No rows ({num_rows}) or columns ({num_cols}) detected. Returning empty list.")
            return []
            
        # Construct an empty grid of cells (each cell will hold a list of (x_mid, text))
        grid = [[[] for _ in range(num_cols)] for _ in range(num_rows)]
        
        # 4. Map OCR tokens to the rows/columns grid using bounding box containment
        for bbox, text, conf in ocr_results:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x_mid = sum(xs) / len(xs)
            y_mid = sum(ys) / len(ys)
            
            # Find column by midpoint
            col_idx = -1
            for idx, col in enumerate(cols):
                xmin, _, xmax, _ = col["box"]
                if xmin <= x_mid <= xmax:
                    col_idx = idx
                    break
            
            # Find row by midpoint
            row_idx = -1
            for idx, r in enumerate(rows):
                _, ymin, _, ymax = r["box"]
                if ymin <= y_mid <= ymax:
                    row_idx = idx
                    break
                    
            if col_idx != -1 and row_idx != -1:
                grid[row_idx][col_idx].append((x_mid, text))
                
        # 5. Format grid cell texts
        formatted_grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]
        for r in range(num_rows):
            for c in range(num_cols):
                # Sort horizontally by x_mid coordinates before joining
                tokens = sorted(grid[r][c], key=lambda x: x[0])
                formatted_grid[r][c] = " ".join([t[1] for t in tokens]).strip()
                
        # 6. Apply spanning cell logic
        for span in spans:
            s_xmin, s_ymin, s_xmax, s_ymax = span["box"]
            
            # Find covered rows
            covered_rows = []
            for idx, r in enumerate(rows):
                r_xmin, r_ymin, r_xmax, r_ymax = r["box"]
                overlap_y = min(s_ymax, r_ymax) - max(s_ymin, r_ymin)
                r_height = r_ymax - r_ymin
                if r_height > 0 and (overlap_y / r_height) > 0.5:
                    covered_rows.append(idx)
                    
            # Find covered columns
            covered_cols = []
            for idx, col in enumerate(cols):
                c_xmin, _, c_xmax, _ = col["box"]
                overlap_x = min(s_xmax, c_xmax) - max(s_xmin, c_xmin)
                c_width = c_xmax - c_xmin
                if c_width > 0 and (overlap_x / c_width) > 0.5:
                    covered_cols.append(idx)
                    
            if covered_rows and covered_cols:
                r_start, r_end = min(covered_rows), max(covered_rows)
                c_start, c_end = min(covered_cols), max(covered_cols)
                
                # Combine texts in the span
                combined_text_parts = []
                for r in range(r_start, r_end + 1):
                    for c in range(c_start, c_end + 1):
                        val = formatted_grid[r][c]
                        if val and val not in combined_text_parts:
                            combined_text_parts.append(val)
                combined_text = " ".join(combined_text_parts).strip()
                
                # Assign to top-left of span, clear other cells covered by span
                for r in range(r_start, r_end + 1):
                    for c in range(c_start, c_end + 1):
                        if r == r_start and c == c_start:
                            formatted_grid[r][c] = combined_text
                        else:
                            formatted_grid[r][c] = ""
                            
        # 7. Identify Header rows based on overlap with predicted headers
        header_row_indices = set()
        for h in headers:
            h_xmin, h_ymin, h_xmax, h_ymax = h["box"]
            for idx, r in enumerate(rows):
                r_xmin, r_ymin, r_xmax, r_ymax = r["box"]
                overlap = min(h_ymax, r_ymax) - max(h_ymin, r_ymin)
                r_height = r_ymax - r_ymin
                if r_height > 0 and (overlap / r_height) > 0.5:
                    header_row_indices.add(idx)
                    
        if not header_row_indices:
            header_row_indices.add(0) # fallback: first row is header
            
        num_headers = max(header_row_indices) + 1
        
        # 8. Reconstruct headers
        col_headers = ["" for _ in range(num_cols)]
        for r_idx in range(num_headers):
            for c_idx in range(num_cols):
                val = formatted_grid[r_idx][c_idx]
                if val:
                    if col_headers[c_idx]:
                        col_headers[c_idx] += " - " + val
                    else:
                        col_headers[c_idx] = val
                        
        # Clean header names
        final_headers = []
        seen = {}
        for idx, h in enumerate(col_headers):
            base_h = h.strip() if h.strip() else f"Column_{idx+1}"
            if base_h not in seen:
                seen[base_h] = 0
                final_headers.append(base_h)
            else:
                seen[base_h] += 1
                final_headers.append(f"{base_h}_{seen[base_h]}")
                
        # 9. Extract data rows
        extracted = []
        for r_idx in range(num_headers, num_rows):
            row_dict = {}
            for c_idx in range(num_cols):
                row_dict[final_headers[c_idx]] = formatted_grid[r_idx][c_idx]
            if any(str(v).strip() for v in row_dict.values()):
                extracted.append(row_dict)
                
        return extracted

import base64
import io
import subprocess
import tempfile
import os

from PIL import Image


def _get_page_longest_dim(pdf_bytes: bytes, page_number: int) -> float:
    """Get the longest dimension (in points) of a PDF page using pdfinfo."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            ["pdfinfo", "-f", str(page_number), "-l", str(page_number), "-box", "-enc", "UTF-8", tmp_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "MediaBox" in line:
                    parts = line.split(":")[1].strip().split()
                    w = abs(float(parts[0]) - float(parts[2]))
                    h = abs(float(parts[3]) - float(parts[1]))
                    return max(w, h)
    except Exception:
        pass
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    return 792.0  # Default: letter-size page height in points


def render_pdf_to_base64png(pdf_stream, page_number=1, resolution=1288):
    """
    Render a single PDF page to a base64-encoded PNG.

    Args:
        pdf_stream: BytesIO or file-like object containing PDF bytes.
        page_number: 1-indexed page number to render.
        resolution:  Target longest image dimension in pixels.
                     Internally converted to the correct DPI.
    """
    try:
        # Read bytes from stream
        if hasattr(pdf_stream, 'read'):
            pdf_stream.seek(0)
            pdf_bytes = pdf_stream.read()
        else:
            # Already bytes
            pdf_bytes = pdf_stream

        # Compute DPI so that the longest dim = resolution pixels
        longest_dim_pts = _get_page_longest_dim(pdf_bytes, page_number)
        dpi = int(resolution * 72.0 / longest_dim_pts)
        dpi = max(dpi, 50)  # Safety floor

        # Write to temp file for pdftoppm
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [
                    "pdftoppm", "-png",
                    "-f", str(page_number),
                    "-l", str(page_number),
                    "-r", str(dpi),
                    tmp_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(f"pdftoppm failed: {result.stderr.decode()}")

            return base64.b64encode(result.stdout).decode("utf-8")

        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    except Exception as e:
        print(f"PDF render error: {str(e)}")
        # Return a small pink placeholder so the caller can detect the failure
        img = Image.new('RGB', (800, 1000), color=(255, 200, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode('utf-8')

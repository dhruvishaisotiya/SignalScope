"""
Optional deep-detector inference (EfficientNet-B0 exported to ONNX by
training/train_deep_colab.py). When model/cnn_model.onnx exists, predict.py
uses this as the primary visual scorer; otherwise the classic two-branch
model is used. Requires: pip install onnxruntime
"""
import os

import cv2
import numpy as np

CNN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_model.onnx")
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_session = None


def cnn_available() -> bool:
    if not os.path.exists(CNN_PATH):
        return False
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        _session = ort.InferenceSession(CNN_PATH, providers=["CPUExecutionProvider"])
    return _session


def _prep(img_bgr, size=224):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    s = 256 / min(h, w)
    img = cv2.resize(img, (max(size, int(round(w * s))), max(size, int(round(h * s)))),
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    y0, x0 = (h - size) // 2, (w - size) // 2
    img = img[y0:y0 + size, x0:x0 + size].astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    return img.transpose(2, 0, 1)[None]


def cnn_p_ai(img_bgr) -> float:
    """P(AI-generated) from the deep model. Averages the whole image and a
    center crop for a little scale robustness."""
    sess = _get_session()
    views = [_prep(img_bgr)]
    h, w = img_bgr.shape[:2]
    if min(h, w) >= 448:  # add a native-resolution center crop view
        cy, cx = h // 2, w // 2
        views.append(_prep(img_bgr[cy - 224:cy + 224, cx - 224:cx + 224]))
    ps = []
    for v in views:
        logit = sess.run(None, {"input": v.astype(np.float32)})[0].ravel()[0]
        ps.append(1.0 / (1.0 + np.exp(-float(logit))))
    return float(np.mean(ps))

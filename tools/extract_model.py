"""One-time build step: copy ddddocr's OCR model into the extension, trimmed to what SRM needs.

    pip install ddddocr onnx
    python tools/extract_model.py

ddddocr's common_old.onnx scores 8210 characters (mostly Chinese) at every time step, but the
SRM captcha only uses 62 letters/digits. The final layer (1024 x 8210 int8 weights, ~8.4 MB) is
cut down to the CTC blank + those 62 columns. Weights, scale and zero point are per-tensor, so
the kept scores are bit-identical to the full model; only scores we never looked at are removed.

Writes model/ocr.onnx: output 0 is the CTC blank, output i is ALPHABET[i - 1] (the same
ALPHABET constant is in solver.js). Not needed to use the extension - only to rebuild model/.
"""
import os

import ddddocr
import numpy as np
import onnx
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "..", "model")
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

charset = ddddocr.DdddOcr(show_ad=False).ocr_engine.charset_manager.get_charset()
assert charset[0] == "", "index 0 must be the CTC blank"
keep = [0] + [charset.index(c) for c in ALPHABET]

model = onnx.load(os.path.join(os.path.dirname(ddddocr.__file__), "common_old.onnx"))
graph = model.graph

# The classifier is the graph's only MatMulInteger, followed by its bias Add.
inits = {t.name: t for t in graph.initializer}
matmul = next(n for n in graph.node if n.op_type == "MatMulInteger")
weight_name = matmul.input[1]
bias_name = next(i for n in graph.node if n.op_type == "Add" for i in n.input
                 if i in inits and list(inits[i].dims) == [len(charset)])

for name, axis in ((weight_name, 1), (bias_name, 0)):
    array = numpy_helper.to_array(inits[name])
    assert array.shape[axis] == len(charset), (name, array.shape)
    inits[name].CopyFrom(numpy_helper.from_array(np.ascontiguousarray(np.take(array, keep, axis=axis)), name))
for name in (matmul.input[3],):                  # zero point must be per-tensor for slicing to be exact
    assert numpy_helper.to_array(inits[name]).ndim == 0, name

del graph.value_info[:]                         # drop stale 8210-wide shape hints
# The original declares the output as [1, seqlen] but it is [time, 1, classes]; ONNX Runtime
# logs a warning on every run about the mismatch, so declare the real shape.
output_dims = graph.output[0].type.tensor_type.shape
del output_dims.dim[:]
for dim in ("time", 1, len(keep)):
    output_dims.dim.add().CopyFrom(onnx.TensorShapeProto.Dimension(
        **({"dim_param": dim} if isinstance(dim, str) else {"dim_value": dim})))
onnx.checker.check_model(model)

os.makedirs(MODEL_DIR, exist_ok=True)
onnx.save(model, os.path.join(MODEL_DIR, "ocr.onnx"))
print(f"model/ocr.onnx written ({os.path.getsize(os.path.join(MODEL_DIR, 'ocr.onnx')):,} bytes)")

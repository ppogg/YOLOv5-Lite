#!/usr/bin/env python3
"""Verify NCNN export fixes for YOLOv5-Lite (issue #53).

Checks:
  1. ncnn_forward produces three heads with the expected layout.
  2. ONNX export with --ncnn writes three outputs and dynamic head Reshape nodes.
  3. fix_ncnn_param.py patches fixed-grid Reshape lines in .param files.
"""

import re
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.yolo import Model
from scripts.fix_ncnn_param import fix_ncnn_param_lines, patch_onnx_reshape_for_ncnn

RESHAPE_HEAD_PATTERN = re.compile(r'Reshape\s+\S+.*\s0=(-?\d+)\s+1=(\d+)\s+2=3')


def check_ncnn_forward():
    model = Model(ROOT / 'models' / 'v5Lite-s.yaml')
    detect = model.model[-1]
    detect.eval()

    x = [
        torch.randn(1, 64, 40, 40),
        torch.randn(1, 128, 20, 20),
        torch.randn(1, 256, 10, 10),
    ]
    outputs = detect.ncnn_forward(x)
    assert len(outputs) == 3, 'expected three detection heads'

    grids = [40 * 40, 20 * 20, 10 * 10]
    for out, grid in zip(outputs, grids):
        assert out.shape == (1, grid, 85, 3), f'unexpected head shape: {out.shape}'
    print('PASS: ncnn_forward head shapes')


def check_onnx_export():
    import onnx
    from onnx import numpy_helper

    with tempfile.TemporaryDirectory() as tmp:
        cfg = ROOT / 'models' / 'v5Lite-s.yaml'
        model = Model(cfg)
        model.eval()
        detect = model.model[-1]
        detect.forward = detect.ncnn_forward

        onnx_path = Path(tmp) / 'verify-ncnn.onnx'
        img = torch.zeros(1, 3, 320, 320)
        torch.onnx.export(
            model,
            img,
            str(onnx_path),
            opset_version=12,
            input_names=['images'],
            output_names=['output', 'stride16', 'stride32'],
            dynamic_axes={
                'images': {0: 'batch', 2: 'height', 3: 'width'},
                'output': {0: 'batch', 1: 'stride8_grid'},
                'stride16': {0: 'batch', 1: 'stride16_grid'},
                'stride32': {0: 'batch', 1: 'stride32_grid'},
            },
            dynamo=False,
        )

        patched = patch_onnx_reshape_for_ncnn(onnx_path, nc=80)
        onnx_model = onnx.load(str(onnx_path))
        output_names = {out.name for out in onnx_model.graph.output}
        assert output_names == {'output', 'stride16', 'stride32'}, f'unexpected outputs: {output_names}'

        constants = {}
        for node in onnx_model.graph.node:
            if node.op_type == 'Constant':
                for attr in node.attribute:
                    if attr.name == 'value':
                        constants[node.output[0]] = numpy_helper.to_array(attr.t)

        head_reshapes = []
        for node in onnx_model.graph.node:
            if node.op_type != 'Reshape' or len(node.input) < 2:
                continue
            shape = constants.get(node.input[1])
            if shape is None:
                continue
            flat = list(shape.reshape(-1))
            if flat[-2:] == [85, 3] and (len(flat) == 3 or (len(flat) == 4 and flat[0] == 1)):
                head_reshapes.append(flat[0] if len(flat) == 3 else flat[1])

        assert len(head_reshapes) == 3, f'expected 3 head Reshape nodes, found {head_reshapes}'
        assert all(dim == -1 for dim in head_reshapes), f'head Reshape dims must be -1: {head_reshapes}'
        assert patched == 3, f'expected 3 patched ONNX reshapes, got {patched}'
    print('PASS: ONNX NCNN export outputs and dynamic Reshape nodes')


def check_param_patcher():
    sample = """Reshape                  Reshape_468              1 1 632 650 0=6400 1=85 2=3
Permute                  Transpose_469            1 1 650 output 0=1
Reshape                  Reshape_484              1 1 652 670 0=1600 1=85 2=3
Permute                  Transpose_485            1 1 670 671 0=1
Reshape                  Reshape_500              1 1 672 690 0=400 1=85 2=3
Permute                  Transpose_501            1 1 690 691 0=1
Reshape                  Reshape_other            1 1 100 101 0=6 1=-1
"""
    updated, patched = fix_ncnn_param_lines(sample.splitlines(keepends=True), nc=80)
    text = ''.join(updated)
    head_dims = [int(m.group(1)) for m in RESHAPE_HEAD_PATTERN.finditer(text) if int(m.group(2)) == 85]
    assert patched == 3, f'expected 3 param lines patched, got {patched}'
    assert head_dims == [-1, -1, -1], f'head reshape dims should be -1: {head_dims}'
    assert '0=6 1=-1' in text, 'non-head reshape lines must be unchanged'
    print('PASS: NCNN param Reshape patcher')


def check_reference_param():
    ref = ROOT / 'cpp_demo' / 'ncnn' / 'model_zoo' / 'yolov5-lite.param'
    if not ref.exists():
        print('SKIP: reference param file not found')
        return

    text = ref.read_text()
    head_dims = [int(m.group(1)) for m in RESHAPE_HEAD_PATTERN.finditer(text) if int(m.group(2)) == 85]
    assert len(head_dims) == 3, f'reference param missing 3 head reshapes: {head_dims}'
    assert all(dim == -1 for dim in head_dims), f'reference param must use -1: {head_dims}'
    print('PASS: bundled reference NCNN param uses dynamic head reshapes')


def main():
    checks = [
        check_ncnn_forward,
        check_onnx_export,
        check_param_patcher,
        check_reference_param,
    ]
    failed = 0
    for check in checks:
        try:
            check()
        except Exception as exc:
            failed += 1
            print(f'FAIL: {check.__name__}: {exc}', file=sys.stderr)

    if failed:
        print(f'\n{failed} check(s) failed', file=sys.stderr)
        return 1

    print('\nAll NCNN export verification checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

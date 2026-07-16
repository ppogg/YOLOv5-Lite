#!/usr/bin/env python3
"""Regression checks for the YOLOv5-Lite ONNX-to-NCNN export contract."""

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.yolo import Model
from scripts.fix_ncnn_param import (
    _onnx_constants,
    fix_ncnn_param_lines,
    patch_onnx_reshape_for_ncnn,
)


def verify_head_values():
    """Check shape and channel semantics, which a shape-only test can miss."""
    torch.manual_seed(53)
    model = Model(ROOT / 'models' / 'v5Lite-s.yaml')
    detect = model.model[-1].eval()
    features = [
        torch.randn(1, 64, 40, 24),
        torch.randn(1, 128, 20, 12),
        torch.randn(1, 256, 10, 6),
    ]

    with torch.no_grad():
        raw = [conv(feature) for conv, feature in zip(detect.m, features)]
        actual = detect.ncnn_forward([feature.clone() for feature in features])

    for raw_head, exported in zip(raw, actual):
        bs, _, ny, nx = raw_head.shape
        expected = (
            raw_head.view(bs, detect.na, detect.no, ny * nx)
            .permute(0, 1, 3, 2)
            .contiguous()
        )
        assert exported.shape == (bs, detect.na, ny * nx, detect.no)
        assert torch.equal(exported, expected), 'anchor/output channels were reordered'

    print('PASS: NCNN heads preserve anchor-major detector values')


def verify_onnx_graph():
    import onnx

    model = Model(ROOT / 'models' / 'v5Lite-s.yaml').eval()
    detect = model.model[-1]
    detect.forward = detect.ncnn_forward

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'v5lite-ncnn.onnx'
        kwargs = {}
        if 'dynamo' in __import__('inspect').signature(torch.onnx.export).parameters:
            kwargs['dynamo'] = False
        torch.onnx.export(
            model,
            torch.zeros(1, 3, 320, 192),
            str(path),
            opset_version=12,
            input_names=['images'],
            output_names=['output', 'stride16', 'stride32'],
            **kwargs
        )

        patched = patch_onnx_reshape_for_ncnn(path, nc=detect.nc, na=detect.na)
        graph = onnx.load(str(path))
        onnx.checker.check_model(graph)
        assert patched == detect.nl == 3
        assert [output.name for output in graph.graph.output] == [
            'output', 'stride16', 'stride32'
        ]

        constants = _onnx_constants(graph)
        shapes = []
        for node in graph.graph.node:
            if node.op_type == 'Reshape' and len(node.input) > 1:
                shape = constants.get(node.input[1])
                if shape is not None:
                    shape = [int(value) for value in shape.reshape(-1)]
                    if len(shape) == 4 and shape[1:3] == [detect.na, detect.no]:
                        shapes.append(shape)
        assert len(shapes) == detect.nl, shapes
        assert all(shape[3] == -1 for shape in shapes), shapes

    print('PASS: ONNX heads use named outputs and inferred NCNN grid sizes')


def verify_param_rewrite():
    original = [
        'Reshape head8 1 1 in8 out8 0=960 1=85 2=3\n',
        'Reshape head16 1 1 in16 out16 0=240 1=85 2=3\n',
        'Reshape head32 1 1 in32 out32 0=60 1=85 2=3\n',
        'Reshape unrelated 1 1 in out 0=960 1=84 2=3\n',
    ]
    fixed, patched = fix_ncnn_param_lines(original)
    assert patched == 3
    assert all('0=-1' in line for line in fixed[:3])
    assert fixed[3] == original[3]

    reference = (
        ROOT / 'cpp_demo' / 'ncnn' / 'model_zoo' / 'yolov5-lite.param'
    ).read_text()
    head_lines = [
        line for line in reference.splitlines()
        if line.startswith('Reshape') and '1=85' in line and '2=3' in line
    ]
    assert len(head_lines) == 3
    assert all('0=-1' in line for line in head_lines)

    print('PASS: NCNN param rewrite is targeted and resolution-independent')


def verify_consumer_output_names():
    consumers = [
        ROOT / 'cpp_demo' / 'ncnn' / 'v5lite-s.cpp',
        ROOT / 'android_demo' / 'ncnn-android-v5lite' / 'app' / 'src' / 'main'
        / 'jni' / 'yolov5.cpp',
    ]
    for consumer in consumers:
        source = consumer.read_text()
        for output in ('output', 'stride16', 'stride32'):
            assert 'extract("%s"' % output in source, (consumer, output)
        assert 'extract("1111"' not in source
        assert 'extract("2222"' not in source

    print('PASS: NCNN consumers use the exported output names')


def main():
    verify_head_values()
    verify_onnx_graph()
    verify_param_rewrite()
    verify_consumer_output_names()
    print('All Issue #53 regression checks passed.')


if __name__ == '__main__':
    main()

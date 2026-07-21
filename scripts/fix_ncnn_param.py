#!/usr/bin/env python3
"""Fix dynamic detection-head reshapes in ONNX and NCNN model files."""

import argparse
import re
from pathlib import Path


_GRID_PARAM = re.compile(r'(?<!\S)0=(\d+)(?!\S)')


def fix_ncnn_param_lines(lines, nc=80, na=3):
    """Replace fixed detection grid sizes with NCNN's inferred dimension."""
    no = nc + 5
    fixed = []
    patched = 0
    for line in lines:
        tokens = line.split()
        is_head = (
            tokens
            and tokens[0] == 'Reshape'
            and '1=%d' % no in tokens
            and '2=%d' % na in tokens
        )
        if is_head:
            line, count = _GRID_PARAM.subn('0=-1', line, count=1)
            patched += count
        fixed.append(line)
    return fixed, patched


def fix_ncnn_param(path, nc=80, na=3, output=None):
    """Patch an onnx2ncnn .param file and return the number of changes."""
    path = Path(path)
    lines = path.read_text().splitlines(keepends=True)
    fixed, patched = fix_ncnn_param_lines(lines, nc=nc, na=na)
    target = Path(output) if output else path
    target.write_text(''.join(fixed))
    return patched


def _onnx_constants(model):
    from onnx import numpy_helper

    constants = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    for node in model.graph.node:
        if node.op_type == 'Constant':
            value = next((attr.t for attr in node.attribute if attr.name == 'value'), None)
            if value is not None:
                constants[node.output[0]] = numpy_helper.to_array(value)
    return constants


def patch_onnx_reshape_for_ncnn(path, nc=80, na=3):
    """Make exported Detect reshape grid dimensions inferable by onnx2ncnn."""
    import numpy as np
    import onnx
    from onnx import numpy_helper

    path = Path(path)
    model = onnx.load(str(path))
    constants = _onnx_constants(model)
    changed = {}
    no = nc + 5

    for node in model.graph.node:
        if node.op_type != 'Reshape' or len(node.input) < 2:
            continue
        name = node.input[1]
        if name not in constants:
            continue
        shape = np.asarray(constants[name])
        flat = shape.reshape(-1)
        # ncnn_forward reshapes to (batch, anchors, outputs, grid) before
        # transposing to the final (batch, anchors, grid, outputs) tensor.
        if (
            len(flat) == 4
            and int(flat[1]) == na
            and int(flat[2]) == no
            and int(flat[3]) > 0
        ):
            updated = shape.copy()
            updated.reshape(-1)[3] = -1
            constants[name] = updated
            changed[name] = updated

    for node in model.graph.node:
        if node.op_type != 'Constant' or node.output[0] not in changed:
            continue
        name = node.output[0]
        del node.attribute[:]
        node.attribute.append(
            onnx.helper.make_attribute(
                'value', numpy_helper.from_array(changed[name], name=name)
            )
        )

    for initializer in model.graph.initializer:
        if initializer.name in changed:
            initializer.CopyFrom(
                numpy_helper.from_array(changed[initializer.name], name=initializer.name)
            )

    if changed:
        onnx.checker.check_model(model)
        onnx.save(model, str(path))
    return len(changed)


def main():
    parser = argparse.ArgumentParser(
        description='Replace fixed YOLOv5-Lite detection grid sizes with -1'
    )
    parser.add_argument('path', type=Path, help='ONNX model or onnx2ncnn .param file')
    parser.add_argument('--nc', type=int, default=80, help='number of classes')
    parser.add_argument('--na', type=int, default=3, help='anchors per detection head')
    parser.add_argument('--output', type=Path, help='output path (.param files only)')
    args = parser.parse_args()

    if args.path.suffix == '.onnx':
        if args.output:
            parser.error('--output is only supported for .param files')
        patched = patch_onnx_reshape_for_ncnn(args.path, nc=args.nc, na=args.na)
    elif args.path.suffix == '.param':
        patched = fix_ncnn_param(
            args.path, nc=args.nc, na=args.na, output=args.output
        )
    else:
        parser.error('path must end in .onnx or .param')

    print('patched %d detection-head Reshape layer(s)' % patched)


if __name__ == '__main__':
    main()

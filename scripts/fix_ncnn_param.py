"""Patch NCNN .param files for YOLOv5-Lite ONNX exports (issue #53).

After onnx2ncnn conversion, detection-head Reshape layers must use a dynamic
first dimension (0=-1) instead of a fixed grid size so inference works at any
input resolution.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np


RESHAPE_HEAD_PATTERN = re.compile(
    r'^(Reshape\s+\S+\s+\d+\s+\d+\s+\S+\s+\S+\s+)0=(\d+)(\s+1=(\d+)\s+2=(\d+))'
)


def _is_head_reshape_shape(shape, no):
    shape = list(int(x) for x in np.array(shape).reshape(-1))
    if shape[-2:] != [no, 3]:
        return False, None
    if len(shape) == 3:
        return shape[0] > 0, 0
    if len(shape) == 4 and shape[0] == 1:
        return shape[1] > 0, 1
    return False, None


def _patch_shape_array(shape, no):
    shape = np.array(shape).copy()
    flat = shape.reshape(-1)
    should_patch, index = _is_head_reshape_shape(flat, no)
    if not should_patch:
        return None
    flat[index] = -1
    return flat.reshape(shape.shape)


def _collect_constants(model):
    from onnx import numpy_helper

    constants = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != 'Constant':
            continue
        for attr in node.attribute:
            if attr.name == 'value':
                constants[node.output[0]] = numpy_helper.to_array(attr.t)
    return constants


def fix_ncnn_param_lines(lines, nc=80):
    """Return updated lines and the number of head Reshape layers patched."""
    no = nc + 5
    patched = 0
    out = []
    for line in lines:
        match = RESHAPE_HEAD_PATTERN.match(line)
        if match and int(match.group(4)) == no and int(match.group(5)) == 3:
            prefix, _, suffix = match.group(1), match.group(2), match.group(3)
            if match.group(2) != '-1':
                line = f'{prefix}0=-1{suffix}\n'
                patched += 1
        out.append(line if line.endswith('\n') else line + '\n')
    return out, patched


def fix_ncnn_param(param_path, nc=80, inplace=True):
    param_path = Path(param_path)
    lines = param_path.read_text().splitlines(keepends=True)
    updated, patched = fix_ncnn_param_lines(lines, nc=nc)
    if patched == 0:
        return 0

    target = param_path if inplace else param_path.with_suffix('.ncnn.param')
    target.write_text(''.join(updated))
    return patched


def patch_onnx_reshape_for_ncnn(onnx_path, nc=80, inplace=True):
    """Set detection-head ONNX Reshape grid dimensions to -1 before onnx2ncnn."""
    import onnx
    from onnx import numpy_helper

    no = nc + 5
    model = onnx.load(str(onnx_path))
    constants = _collect_constants(model)
    patched_names = set()

    for node in model.graph.node:
        if node.op_type != 'Reshape' or len(node.input) < 2:
            continue
        shape_name = node.input[1]
        if shape_name not in constants:
            continue
        updated = _patch_shape_array(constants[shape_name], no)
        if updated is None:
            continue
        constants[shape_name] = updated
        patched_names.add(shape_name)

    if not patched_names:
        return 0

    for node in model.graph.node:
        if node.op_type != 'Constant' or node.output[0] not in patched_names:
            continue
        name = node.output[0]
        while len(node.attribute):
            node.attribute.pop()
        node.attribute.extend([onnx.helper.make_attribute('value', numpy_helper.from_array(constants[name], name=name))])

    for idx, init in enumerate(model.graph.initializer):
        if init.name in patched_names:
            model.graph.initializer[idx].CopyFrom(numpy_helper.from_array(constants[init.name], name=init.name))

    onnx.checker.check_model(model)
    target = Path(onnx_path) if inplace else Path(onnx_path).with_suffix('.ncnn.onnx')
    onnx.save(model, str(target))
    return len(patched_names)


def main():
    parser = argparse.ArgumentParser(description='Fix YOLOv5-Lite NCNN export artifacts (issue #53)')
    parser.add_argument('path', help='.param file (after onnx2ncnn) or .onnx file (before onnx2ncnn)')
    parser.add_argument('--nc', type=int, default=80, help='number of classes (default: 80)')
    parser.add_argument('--output', type=str, default='', help='optional output path')
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f'error: file not found: {path}', file=sys.stderr)
        return 1

    if path.suffix == '.onnx':
        patched = patch_onnx_reshape_for_ncnn(path, nc=args.nc, inplace=not args.output)
        kind = 'ONNX Reshape'
    elif path.suffix == '.param':
        if args.output:
            text, patched = fix_ncnn_param_lines(path.read_text().splitlines(keepends=True), nc=args.nc)
            Path(args.output).write_text(''.join(text))
        else:
            patched = fix_ncnn_param(path, nc=args.nc, inplace=True)
        kind = 'NCNN param Reshape'
    else:
        print('error: path must end with .onnx or .param', file=sys.stderr)
        return 1

    print(f'{kind} nodes patched: {patched}')
    return 0 if patched > 0 else 2


if __name__ == '__main__':
    sys.exit(main())

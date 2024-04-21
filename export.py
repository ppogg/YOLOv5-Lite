"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse
import sys
import time

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn

import models
from models.common import NMS, NMS_Export
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device

def export_onnx(model, img, dynamic, output_names=None):
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                          output_names=['classes', 'boxes'] if y is None else output_names,
                          dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                                        'output': {0: 'batch', 2: 'y', 3: 'x'}} if opt.dynamic else None)

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model

        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='model_zoo/v5lite-e.pt', help='weights path')  # from yolov5/models/
    parser.add_argument('--img-size', nargs='+', type=int, default=[320, 320], help='image size')  # height, width
    parser.add_argument('--concat', action='store_true', help='concat or not')
    parser.add_argument('--mnnd', action='store_true', help='mnn decode or not')
    parser.add_argument('--mnne', action='store_true', help='mnn end2end or not')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--end2end', action='store_true', help='export the nms part in ONNX model')  # ONNX-only, #opt.grid has to be set True for nms export to work
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, models.yolo.Detect):
            if opt.concat:
                m.forward = m.cat_forward
            elif opt.mnnd:
                m.forward = m.mnnd_forward
            elif opt.mnne:
                m.forward = m.mnne_forward
            elif opt.end2end:
                m.forward = m.end2end_forward
            else:
                m.forward

    model.model[-1].export = not opt.grid  # set Detect() layer grid export
    print(model.model[-1])
    y = model(img)  # dry run

    if opt.end2end:
        #nms = NMS(conf=0.001)
        #nms_export = NMS_Export(conf=0.001)
        ##y_export = nms_export(y)
        ##y = nms(y)
        ##assert (torch.sum(torch.abs(y_export[0]-y[0]))<1e-6)
        #model_nms = torch.nn.Sequential(model, nms_export)
        #model_nms.eval()
        output_names = ['outputs']
    elif opt.concat or opt.mnnd or opt.mnne:
        output_names = ['outputs']

    dynamic = opt.dynamic

    if opt.end2end:
        # print(model_nms)
        export_onnx(model, img, dynamic, output_names)
    elif opt.concat:
        # print(model)
        export_onnx(model, img, dynamic, output_names)
    elif opt.mnnd:
        export_onnx(model, img, dynamic, output_names)
    elif opt.mnne:
        export_onnx(model, img, dynamic, output_names)
    else:
        export_onnx(model, img, dynamic)





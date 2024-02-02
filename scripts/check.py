import onnxruntime
import cv2
import time
import torch
import random
from torchvision import transforms
import numpy as np
import torchvision


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def letterbox(img, new_shape=(320, 320), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def non_max_suppression_end2end(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, nc=None):
    output = []
    print(prediction.shape)

    xc = prediction[:, 4] > conf_thres  # candidates
    output = prediction[xc]
    print(output)

    return output

def non_max_suppression_mnne(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, nc=None):
    output = []

    xc = prediction[:, 4] > conf_thres  # candidates
    output = prediction[xc]

    boxes, scores = output[:, :4], output[:, 4]  # boxes (offset by class), scores
    print(output.shape)

    i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS

    return output[i]

def non_max_suppression_mnnd(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, nc=None):
    output = []
    print(prediction.shape)

    min_wh, max_wh = 2, 4096  
    xc = prediction[..., 4] > conf_thres  # candidates
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    print(type(output))
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        # Compute conf
        cx, cy, w, h = x[:,0:1], x[:,1:2], x[:,2:3], x[:,3:4]
        obj_conf = x[:,4:5]
        cls_conf = x[:,5:]
        cls_conf = obj_conf * cls_conf  # conf = obj_conf * cls_conf
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        conf, j = cls_conf.max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] +c , x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        output[xi] = x[i].view(-1, 6)

    return output[0]


def plot_one_box(x, im, color=None, line_thickness=3):
    # print(x)
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, (255, 0, 0), thickness=tl * 1 // 3, lineType=cv2.LINE_AA)
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(str(round(float(x[4]), 6)), 0, fontScale=tl / 6, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(im, str(round(float(x[4]), 6)), (c1[0], c1[1] - 2), 0, tl / 6, [225, 255, 255], thickness=tf // 6,
                lineType=cv2.LINE_AA)

def process(weight_path, img_path):
    image = cv2.imread(img_path)
    image = letterbox(image, 320, stride=32, auto=True)[0]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = np.ascontiguousarray(image)
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))

    sess = onnxruntime.InferenceSession(weight_path)
    out = sess.run(['outputs'], {'images': image.numpy()})[0]
    out = torch.from_numpy(out)

    # 如果使用的是end2end的导出方式，则使用以下后处理
    # output = non_max_suppression_end2end(out, 0.50, 0.50, nc=80)

    # 如果使用的是mnnd的导出方式，则使用以下后处理
    output = non_max_suppression_mnnd(out, 0.50, 0.50, nc=80)

    # 如果使用的是mnne的导出方式，则使用以下后处理
    # output = non_max_suppression_mnne(out, 0.50, 0.50, nc=80)
   
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    for idx in range(output.shape[0]):
        plot_one_box(output[idx], nimg)
    return nimg

if __name__ == '__main__':
    img_path = '/home/chenxr/Racherry/YOLOv5-Lite/sample/000000001000.jpg'
    weight_path = '/home/chenxr/Racherry/YOLOv5-Lite/weights/v5lite-e_end2end.onnx'
    nimg = process(weight_path, img_path)
    cv2.imwrite('tmp.png', nimg)

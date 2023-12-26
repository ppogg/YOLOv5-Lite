import xml.etree.ElementTree as ET
import os
from os import getcwd

sets = ['train', 'test', 'val']

classes = ["person"]  # 我们只检测person这个类别


def convert(size, box):#对图片进行归一化处理
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    in_file = open('data\Person1K\Annotations/%s.xml' % (image_id))
    out_file = open('data\Person1K/labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()
print(wd)
for image_set in sets:
    if not os.path.exists('data\Person1K/labels/'):
        os.makedirs('data\Person1K/labels/')
    image_ids = open('data\Person1K/ImageSets/%s.txt' % (image_set)).read().strip().split()
    list_file = open('data\Person1K/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write('data\Person1K/images/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()

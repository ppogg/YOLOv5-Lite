from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import random

headstr = """\
<annotation>
    <folder>VOC</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>COCO</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""

tailstr = '''\
</annotation>
'''


# 如果目录不存在则创建它，存在则删除后创建
def mkr(path):
    if os.path.exists(path):
        # 删除后创建
        shutil.rmtree(path)
        # os.mkdir(path)
        # os.mkdir创建单层目录；os.makedirs创建多层目录
        os.makedirs(path)
    else:
        os.makedirs(path)


# 通过coco数据集的id，得到它的类别名
def id2name(coco):
    classes = dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']] = cls['name']
    return classes


# 写xml文件
def write_xml(anno_path, head, objs, tail):
    f = open(anno_path, "w")
    f.write(head)
    for obj in objs:
        f.write(objstr % (obj[0], obj[1], obj[2], obj[3], obj[4]))
    f.write(tail)


#
def save_annotations_and_imgs(coco, dataset, filename, objs):
    # eg:COCO_train2014_000000196610.jpg-->COCO_train2014_000000196610.xml
    anno_path = anno_dir + filename[:-3] + 'xml'
    print(anno_path)
    img_path=dataDir+dataset+'/'+filename
    #img_path = dataset + '//' + filename
    print(img_path)
    dst_imgpath = img_dir + filename
    # print(dst_imgpath)
    img = cv2.imread(img_path)
    # print("***********")
    # print(img1.shape)
    try:
        if (img.shape[2] == 1):
            print(filename + " not a RGB image")
            return
    except:
        print(img_path)
    shutil.copy(img_path, dst_imgpath)

    head = headstr % (filename, img.shape[1], img.shape[0], img.shape[2])
    tail = tailstr
    write_xml(anno_path, head, objs, tail)


def showimg(coco, dataset, img, classes, cls_id, show=True):
    global dataDir
    I = Image.open('%s/%s/%s' % (dataDir, dataset, img['file_name']))
    # print(I)#
    # 通过id，得到注释的信息
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    # print(annIds)
    anns = coco.loadAnns(annIds)
    # print(anns)
    # coco.showAnns(anns)
    objs = []
    for ann in anns:
        class_name = classes[ann['category_id']]
        # print(class_name)
        if class_name in classes_names:
            # print(class_name)
            if 'bbox' in ann:
                bbox = ann['bbox']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                obj = [class_name, xmin, ymin, xmax, ymax]
                objs.append(obj)
                draw = ImageDraw.Draw(I)
                draw.rectangle([xmin, ymin, xmax, ymax])
    if show:
        plt.figure()
        plt.axis('off')
        plt.imshow(I)
        plt.show()
    return objs


def generate():
    for dataset in datasets_list:
        # ./COCO/annotations/instances_train2014.json
        annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataset)

        # COCO API for initializing annotated data
        coco = COCO(annFile)
        '''
        COCO 对象创建完毕后会输出如下信息:
        loading annotations into memory...
        Done (t=0.81s)
        creating index...
        index created!
        至此, json 脚本解析完毕, 并且将图片和对应的标注数据关联起来.
        '''
        # show all classes in coco
        classes = id2name(coco)
        print(classes)
        # [1, 2, 3, 4, 6, 8]
        classes_ids = coco.getCatIds(catNms=classes_names)
        print(classes_ids)
        # print("**********")
        print(classes_names)
        for cls in classes_names:
            # Get ID number of this class
            cls_id = coco.getCatIds(catNms=[cls])
            img_ids = coco.getImgIds(catIds=cls_id)
            # print(cls,len(img_ids))
            # imgIds=img_ids[0:10]
            for imgId in tqdm(img_ids):
                img = coco.loadImgs(imgId)[0]
                # print("##########")
                # print(img)
                # print("##########")
                filename = img['file_name']
                # print(filename)
                objs = showimg(coco, dataset, img, classes, classes_ids, show=False)
                # print(objs)
                save_annotations_and_imgs(coco, dataset, filename, objs)


def split_traintest(trainratio=0.7, valratio=0.2, testratio=0.1):
    dataset_dir = dataDir
    files = os.listdir(img_dir)
    trains = []
    vals = []
    trainvals = []
    tests = []
    random.shuffle(files)
    for i in range(len(files)):
        filepath = img_dir + "/" + files[i][:-3] + "jpg"  # 找到以上images文件夹下的图片
        # print(filepath)
        if (i < trainratio * len(files)):
            trains.append(files[i])
            trainvals.append(files[i])
        elif i < (trainratio + valratio) * len(files):
            vals.append(files[i])
            trainvals.append(files[i])
        else:
            tests.append(files[i])
    # 生成yolo所用的txt
    with open(dataset_dir + "/trainval.txt", "w")as f:
        for line in trainvals:
            line = img_dir + "/" + line
            f.write(line + "\n")
    with open(dataset_dir + "/test.txt", "w") as f:
        for line in tests:
            line = img_dir + "/" + line
            f.write(line + "\n")

    # 生成voc所用的txt
    maindir = dataset_dir + "ImageSets/Main"
    mkr(maindir)
    with open(maindir + "/train.txt", "w") as f:
        for line in trains:
            line = line[:line.rfind(".")]
            f.write(line + "\n")
    with open(maindir + "/val.txt", "w") as f:
        for line in vals:
            line = line[:line.rfind(".")]
            f.write(line + "\n")
    with open(maindir + "/trainval.txt", "w") as f:
        for line in trainvals:
            line = line[:line.rfind(".")]
            f.write(line + "\n")
    with open(maindir + "/test.txt", "w") as f:
        for line in tests:
            line = line[:line.rfind(".")]
            f.write(line + "\n")

    print("spliting done")


if __name__ == "__main__":
    # 下载的coco数据集存放路径，annotations和train2014/val2014/...
    dataDir = r'D:\Dataset\coco/'
    # 转化后的文件保存路径
    savepath = r"D:\Dataset\person2"
    # 转化后图片保存路径
    img_dir = savepath + 'images/'
    # 转化后xml保存路径
    anno_dir = savepath + 'Annotations/'
    # 选用的数据集列表
    # datasets_list=['train2014', 'val2014']
    datasets_list = ['val2017']
    # 想要从coco数据集中提取的类别
    # coco数据集目标检测中有90个id，但实际只有80类
    classes_names = ['person']
    """classes_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                     'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                     'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                     'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                     'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                     'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                     'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
                     'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                     'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                     'toothbrush']"""

    # 创建所需要的目录
    mkr(savepath)
    mkr(img_dir)
    mkr(anno_dir)

    # 生成voc格式的数据集
    generate()
    # 生成yolo及voc所使用的txt文件
    split_traintest()
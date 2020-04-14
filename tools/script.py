import os
from os.path import join
from glob import glob
import torch
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from metric.BoundingBox import BoundingBox
from metric.BoundingBoxes import BoundingBoxes
from metric.Evaluator import Evaluator
from metric.utils import BBFormat, BBType, CoordinatesType

YMAX = "ymax"

YMIN = "ymin"

XMAX = "xmax"

XMIN = "xmin"

PERSON = "person"

BNDBOX = "bndbox"

IMAGE_ = "image_"

TRAIN = "train"

TEST = "test"

VID = "VID"

DATA = "Data"

ANNOTATIONS = "Annotations"

def create_image_set_test(data_dir, test_name):
    image_sets = "ImageSets"
    if not os.path.exists(join(data_dir, image_sets)):
        os.makedirs(join(data_dir, image_sets))

    with open(join(data_dir, image_sets, f"VID_val_videos_{test_name}.txt"), "w") as f:
        images = glob(join(data_dir, DATA, VID,TEST,  test_name, "*"))
        images.sort()
        #random.shuffle(images)
        #images = images[: max_data]
        #print "total images %d" % len(images)
        for i, dire in enumerate(images):
            file_name = dire.replace(join(data_dir, DATA, VID) + "/", "").replace(".JPEG", "").replace(".jpg", "")
            #line = '%s %s %s\n' % (file_name, '1')
            line = f"{os.path.dirname(file_name)} {i+1} {i} {len(images)}\n"
            f.write(line)



def draw_all_detection(im, detections, class_names, threshold=0.1):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """


#    color_white = (255, 255, 255)
    color_white = (0, 0, 0)
    color = (255, 255, 0)
    #im = image.transform_inverse(im_array, cfg.network.PIXEL_MEANS)
    # change to bgr
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    size = detections.size
    im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
    bbox = detections.bbox
    scores = detections.extra_fields["scores"]
    labels = detections.extra_fields["labels"]
    for bb, score, label in zip(bbox, scores, labels):
        score = score.item()
        label = label.item()
        bb = bb.numpy()
        if score < threshold:
            continue
        cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), color=color, thickness=2)
        text = '%s %.3f' % (class_names[label], score)
        cv2.putText(im, text, (int(bb[0]), int(bb[1]) + 10), color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    # for j, name in enumerate(class_names):
    #     if name == '__background__':
    #         continue
        #color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        #
        # dets = detections[j]
        # for det in dets:
        #     bbox = det[:4] * scale
        #     score = det[-1]
        #     if score < threshold:
        #         continue
        #     bbox = map(int, bbox)
        #     cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
        #     cv2.putText(im, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
        #                 color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im

def draw_bb(prediction_path, files_path, output_dir):
    #files_path = "/home/indoordesk/PycharmProjects/mega/mega.pytorch/datasets/ILSVRC2015/Data/VID/ILSVRC2015_val_00007010"
    #prediction_path = "/home/indoordesk/PycharmProjects/mega/mega.pytorch/inference/VID_val_frames2"
    prediction_pth = torch.load(os.path.join(prediction_path, "predictions.pth"))
    images_path = os.listdir(files_path)
    images_path = list(map(lambda x: join(files_path, x), images_path))
    images_path.sort()
    print(len(images_path), len(prediction_pth))
    #assert len(images_path) == len(prediction_pth)
    for box_list, image_path in zip(prediction_pth, images_path):
        print(image_path)
        im = cv2.imread(image_path)
        out_im = draw_all_detection(im, box_list, ["i", "person"], threshold = 0.5)
        cv2.imwrite(os.path.join(output_dir, os.path.basename(image_path)), out_im)

def createBoundingBoxes_pred(allBoundingBoxes, img_ids, obj_labels, obj_confs, obj_bboxes):
    for img_id, obj_label, obj_conf, obj_bbox in zip(img_ids, obj_labels, obj_confs, obj_bboxes):
        bb = BoundingBox(
            img_id,
            obj_label,
            obj_bbox[0],
            obj_bbox[1],
            obj_bbox[2],
            obj_bbox[3],
            CoordinatesType.Absolute,
            None,
            BBType.Detected,
            obj_conf,
            format=BBFormat.XYX2Y2)
        allBoundingBoxes.addBoundingBox(bb)

        return allBoundingBoxes


def createBoundingBoxes_gt(path_xml):
    allBoundingBoxes = BoundingBoxes()


    for file in os.listdir(path_xml):

        target = ET.parse(join(path_xml,file)).getroot()
        objs = target.findall("object")
        for obj in objs:
            # if not obj.find("name").text in self.classes_to_ind:
            #   continue
            id = file.replace(".xml", "")
            size = target.find("size")
            im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
            bbox = obj.find("bndbox")
            box = [
                np.maximum(float(bbox.find("xmin").text), 0),
                np.maximum(float(bbox.find("ymin").text), 0),
                np.minimum(float(bbox.find("xmax").text), im_info[1] - 1),
                np.minimum(float(bbox.find("ymax").text), im_info[0] - 1)
            ]
            bb = BoundingBox(
                id,
                1,
                box[0],
                box[1],
                box[2],
                box[3],
                CoordinatesType.Absolute,
                None,
                BBType.GroundTruth,
                format=BBFormat.XYX2Y2)
            allBoundingBoxes.addBoundingBox(bb)

    return allBoundingBoxes



def calculate_map(prediction_path, files_path):

    prediction_pth = torch.load(os.path.join(prediction_path, "predictions.pth"))
    images_path = os.listdir(files_path)
    images_path = list(map(lambda x: join(files_path, x), images_path))
    images_path.sort()
    print(len(images_path), len(prediction_pth))
    #assert len(images_path) == len(prediction_pth)
    allBoundingBoxes = createBoundingBoxes_gt(files_path.replace(DATA, ANNOTATIONS))
    for box_list, image_path in zip(prediction_pth, images_path):
        bbox = box_list.bbox
        scores = box_list.extra_fields["scores"]
        labels = box_list.extra_fields["labels"]
        for bb, score, label in zip(bbox, scores, labels):
            score = score.item()
            label = label.item()
            bb = bb.numpy()
            bb = BoundingBox(os.path.basename(image_path).replace(".jpg", ""),
                             label,bb[0],bb[1],bb[2],bb[3],
                CoordinatesType.Absolute,
                None,
                BBType.Detected,
                score,
                format=BBFormat.XYX2Y2)
            allBoundingBoxes.addBoundingBox(bb)

    evaluator = Evaluator()
    detections = evaluator.GetPascalVOCMetrics(allBoundingBoxes)
    print(detections)
    aps = list(map(lambda x: x["AP"], detections))
    print("ron", aps)
    return aps







#p = "/home/indoordesk/PycharmProjects/mega/mega.pytorch/datasets/ILSVRC2015/"
#draw_bb(None, None, "/tmp")
#create_image_set_test(p)
if __name__ == '__main__':
    p = "/media/indoordesk/653ce34c-0c14-4427-8029-be7afe6d1989/test_sets"
    #for test_file in os.listdir("/media/indoordesk/653ce34c-0c14-4427-8029-be7afe6d1989/test_sets/Data/VID/test"):
    #    create_image_set_test(p, test_file)
    prediction_path = "/home/indoordesk/PycharmProjects/mega/mega.pytorch/inference/VID_val_videos_c"
    file_path = "/media/indoordesk/653ce34c-0c14-4427-8029-be7afe6d1989/test_sets/Data/VID/test/c"
    #calculate_map(prediction_path, file_path)
    draw_bb(prediction_path, file_path, "/home/indoordesk/Desktop/images")
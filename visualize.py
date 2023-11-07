import PIL.Image as pilimg
import numpy as np
import matplotlib.pyplot as plt
import json
import bbox_visualizer as bbv
from PIL import ImageFont, ImageDraw, Image
from PaddleOCR.ppocr.utils.utility import check_and_read, get_image_file_list

fontpath = "fonts/gulim.ttc"
ImageFont.truetype(fontpath, 15)

def change_box_representation(left_top, right_top, right_bottom, left_bottom):
    res = [left_top[0], left_top[1], right_top[0], left_bottom[1]]
    return [int(x) for x in res]

def load_data(root_dir, image_name):
    
    image_path = f"{root_dir}/images/{image_name}"
    result_path = f"{root_dir}/predicts/predict_({image_name.split('.')[0]}).json"

    img = np.array(pilimg.open(image_path))


    with open(result_path, "r") as f:
        result = json.loads(f.readline())[0]

    boxes = [change_box_representation(*x[0]) for x in result]
    labels = [x[1][0] for x in result]
    return [img, boxes, labels]

def draw_multiple_labels(img, boxes, labels, font_size = 10):

    img_pil = Image.fromarray(img)
    fontpath = "fonts/gulim.ttc"
    font = ImageFont.truetype(fontpath, font_size)
    draw = ImageDraw.Draw(img_pil)

    for box, label in zip(boxes, labels):
        box[1] -= font_size
        draw.text(box, label, font=font, fill=(0,0,255))
    return img_pil


def draw_multiple_boxes(img, boxes):

    return bbv.draw_multiple_rectangles(img, boxes, thickness=2, bbox_color=(0,0,0))

def get_image_file_name_list(root_dir):
    image_dir = f"{root_dir}/images"
    image_path_list = get_image_file_list(image_dir)
    image_file_name_list = [x.split("\\")[-1] for x in image_path_list]
    return image_file_name_list


if __name__=="__main__":
    root_dir = "test1"
    
    for image_file_name in get_image_file_name_list(root_dir):

        img, boxes, labels = load_data(root_dir, image_file_name)
        img = draw_multiple_boxes(img, boxes)
        img = draw_multiple_labels(img, boxes, labels)
        plt.imshow(img)
        plt.savefig(f"{root_dir}/results/result_({image_file_name.split('.')[0]}).png", dpi=1200)


    
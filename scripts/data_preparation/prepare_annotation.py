import argparse
import os
from os.path import join
from shutil import copyfile
import json

from pathlib import Path

from cityscapesscripts.preparation.json2instanceImg import json2instanceImg
from cityscapesscripts.preparation.json2labelImg import json2labelImg

parser = argparse.ArgumentParser(description="Script to convert folder containing json annotation to target folder witch cityscapes dataset style")
parser.add_argument("--root_dir", type=str, help="root directory of annotations")
parser.add_argument("--target_dir", type=str, help="target folder in which the jsons are stored")


def _ensure_dir(path):
    if not os.path.exists(path):
        Path(path).mkdir(parents=True, exist_ok=True)

def transform(source_path, target_path):
    """transform _backgroudn_ to cityscapes known void label"""
    lookup_table = {
        '_background_': 'out of roi',
        'points': 'polygon',
        'imageHeight': 'imgHeight',
        'imageWidth': 'imgWidth',
        'shapes': 'objects',
    }
    # Read in data file
    with open(source_path) as in_file:
        data = json.load(in_file)

    # Iterate through keys and transform them
    for shape in range(len(data["shapes"])):
        if data["shapes"][shape]["label"] == "_background_":
            data["shapes"][shape].update({"label": lookup_table["_background_"]})
        # Rename points to polygon and image height, width
        data["shapes"][shape][lookup_table["points"]] = data["shapes"][shape].pop("points")
    data[lookup_table["imageHeight"]] = data.pop("imageHeight")
    data[lookup_table["imageWidth"]] = data.pop("imageWidth")
    data[lookup_table["shapes"]] = data.pop("shapes")

    # Write new file to target directory
    with open(target_path, "w") as out_file:
        json.dump(data, out_file)

def copy_from_dirs(source_dir, target_dir, split):
    target_base_name_img = "{}_{}_000019_leftImg8bit"
    target_base_name_gt = "{}_{}_000019_gtFine_{}"
    counter = 0
    for file_name in filter(lambda x: os.path.isdir(join(source_dir, x)), os.listdir(source_dir)):
        dir_list = os.listdir(join(source_dir, file_name))
        json_list = list(filter(lambda x: 'json' in x, dir_list))

        for json_file in json_list:
            json_name = json_file.split(".")[0]

            # Copy json file
            source_path = join(source_dir, file_name, json_name)

            target_path_img = join(target_dir, "leftImg8bit", split, file_name)
            _ensure_dir(target_path_img)
            target_image_name = target_base_name_img.format(file_name, f"{counter:06d}")
            target_path_img = join(target_path_img, target_image_name)

            target_path_gt = join(target_dir, "gtFine", split, file_name)
            _ensure_dir(target_path_gt)
            target_annot_name = target_base_name_gt.format(file_name, f"{counter:06d}", "polygons")
            target_path_gt = join(target_path_gt, target_annot_name)

            transform(source_path + ".json", target_path_gt + ".json")
            copyfile(source_path + ".png", target_path_img + ".png")

            counter += 1

    return target_image_name

def augment_data_to_cityscapes(target_dir, split):
    folder_root = join(target_dir, "gtFine", split)
    for folder in list(filter(lambda x: os.path.isdir(join(folder_root, x)), os.listdir(folder_root))):
        json_root_dir = join(folder_root, folder)
        for json_file in os.listdir(json_root_dir):
            name, _ = json_file.split(".")
            base_name = name[:name.rfind("_")]
            json2labelImg(join(json_root_dir, name + ".json"), join(json_root_dir, f"{base_name}_labelIds.png"))
            json2instanceImg(join(json_root_dir, name + ".json"), join(json_root_dir, f"{base_name}_instanceIds.png"))


def main(args):
    source_dir = args.root_dir
    target_dir = args.target_dir
    copy_from_dirs(source_dir, target_dir, split="val")
    augment_data_to_cityscapes(target_dir, split="val")

if __name__ == '__main__':
    main(parser.parse_args())
#!/usr/bin/python

# pip install lxml

import sys
import os
import json
import xml.etree.ElementTree as ET


START_BOUNDING_BOX_ID = 1
#PRE_DEFINE_CATEGORIES = {}
# If necessary, pre-define category and its id
PRE_DEFINE_CATEGORIES = {"car": 1, "bus": 2, "van": 3, "others": 4}


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return int(filename)
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.'%(filename))

def convert(xml_list, xml_dir, json_file):
    list_fp = open(xml_list, 'r')
    json_dict = {"images":[], "annotations": [], "categories": []}
    #categories = PRE_DEFINE_CATEGORIES
    #bnd_id = START_BOUNDING_BOX_ID
    id_count = 0

    for line in list_fp:
        line = line.strip()
        print("Processing %s"%(line))
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        folder = root.attrib
        frames = []
        tracks = []
        bboxes = []
        category_ids = []

        for frame in root.iter('frame'):
            id_count = id_count + 1
            frame_name = int(frame.attrib['num'])
            #print("Folder/Frame No. : %s/%s" % (folder['name'], frame_name))
            if frame_name < 10:
                filename = folder['name'] + "/img" + "0000" + str(frame_name) + ".jpg"
            elif frame_name >= 10 and frame_name < 100:
                filename = folder['name'] + "/img" + "000" + str(frame_name) + ".jpg"
            elif frame_name >= 100 and frame_name < 1000:
                filename = folder['name'] + "/img" + "00" + str(frame_name) + ".jpg"
            elif frame_name >= 1000 and frame_name < 10000:
                filename = folder['name'] + "/img" + "0" + str(frame_name) + ".jpg"
            else:
                filename = folder['name'] + "/img" + str(frame_name) + ".jpg"
            image = {'file_name': filename, 'id': id_count, 'frame_id': frame_name}
            json_dict['images'].append(image)

            for target in frame.iter('target'):
                track_id = int(target.attrib['id'])
                tracks.append(track_id)
                frames.append(id_count)

            for box in frame.iter('box'):
                bbox = [float(box.attrib["left"]), float(box.attrib["top"]), float(box.attrib["width"]), float(box.attrib["height"])]
                bboxes.append(bbox)

            for attribute in root.iter('attribute'):
                category_name = attribute.attrib['vehicle_type']
                category_id = PRE_DEFINE_CATEGORIES[category_name]
                category_ids.append(category_id)

        for i in range(len(tracks)):
            ann = {'image_id': frames[i], 'id': i + 1, 'category_id': category_ids[i], 'bbox': bboxes[i], "track_id": tracks[i]}
            json_dict['annotations'].append(ann)

    for class_name in PRE_DEFINE_CATEGORIES:
        json_dict['categories'].append({'name': class_name, 'id': PRE_DEFINE_CATEGORIES[class_name]})
    
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    list_fp.close()


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('3 auguments are need.')
        print('Usage: %s XML_LIST.txt XML_DIR OUTPU_JSON.json'%(sys.argv[0]))
        exit(1)

    convert(sys.argv[1], sys.argv[2], sys.argv[3])

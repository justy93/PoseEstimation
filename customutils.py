import json

def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_id_by_name(labels, name):
    return labels.index(name)


def writeJson(val,fname):
  with open(fname, 'w') as data_file:
    json.dump(val, data_file)

def get_bb_coco(points):
    x_values = [item["x"] for item in points]
    y_values = [item["y"] for item in points]

    # Get the minimum x and y values
    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)

    box_width = max_x - min_x
    box_height = max_y - min_y

    return [min_x, min_y, box_width, box_height]

    
def get_bb(points):
    x_values = [item["x"] for item in points]
    y_values = [item["y"] for item in points]

    # Get the minimum x and y values
    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)

    return [min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]

        
        
def compute_area(bounding_box):
    box_width = bounding_box[2][0] - bounding_box[0][0]
    box_height = bounding_box[2][1] - bounding_box[0][1]

    return box_width*box_height

def compute_area_coco(bounding_box):
    return bounding_box[2]*bounding_box[3]

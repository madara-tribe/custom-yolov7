import sys
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET


# Dictionary that maps class names to IDs
class_name_to_id_mapping = {"trafficlight": 0,
                           "stop": 1,
                           "speedlimit": 2,
                           "crosswalk": 3}
                           

def extract_info_from_xml(xml_file):
    '''
    Function to get the data from XML Annotation
    '''
    root = ET.parse(xml_file).getroot()
    
    # Initialise the info dict 
    info_dict = {}
    info_dict['bboxes'] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name 
        if elem.tag == "filename":
            info_dict['filename'] = elem.text
            
        # Get the image size
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))
            
            info_dict['image_size'] = tuple(image_size)
        
        # Get details of the bounding box 
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text
                    
                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)            
            info_dict['bboxes'].append(bbox)
    
    return info_dict





def convert_to_yolov5(bbox_dict, dir_path, xml_file):
    '''
    Convert the info dict to the required yolo format and write it to disk
    '''
    print_buffer = []
    
    # For each bounding box
    for b in bbox_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
        
        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2 
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])
        
        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = bbox_dict["image_size"]  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        
        #Write the bbox details to the file 
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
    
    fname = bbox_dict["filename"].replace("png", "txt")  
    # Save the annotation to disk
    print("\n".join(print_buffer), file= open(os.path.join(dir_path, fname), "w"))
    os.system("rm {}".format(xml_file))
    
def main(dir_path):
    # Get the annotations
    annotations = [os.path.join(dir_path, x) for x in os.listdir(dir_path) if x[-3:] == "xml"]
    annotations.sort()

    # Convert and save the annotations
    for xml_file in tqdm(annotations):
        bbox_dict = extract_info_from_xml(xml_file)
        convert_to_yolov5(bbox_dict, dir_path, xml_file)
        
if __name__=="__main__":
    if len(sys.argv[1])>0:
        dir_path = str(sys.argv[1])
    else:
        print("python3 ~.py <folder_path>")
        sys.exit(1)
    main(dir_path)


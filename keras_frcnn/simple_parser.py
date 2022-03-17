import cv2
import numpy as np
import os

def get_data(input_path):
	found_bg = False
	all_imgs = {}

	classes_count = {}

	class_mapping = {}

	visualise = True
	
	for filename in os.listdir(input_path):

		with open(os.path.join(input_path, filename),'r') as f:

			filename_no_end = filename.split('.')[0] 

			for line in f:
				line_split = line.strip().split(' ')
				path_list = input_path.strip().split('/')
				file_path = 'Keras-frcnn/data/' + path_list[len(path_list)-1].split('_')[0] + '_' + path_list[len(path_list)-1].split('_')[1] + '/' + filename_no_end + '.jpg'
				class_name = line_split.pop(0)
				line_split.insert(0, file_path)
				line_split.append(class_name)
				(filename,x1,y1,x2,y2,class_name) = line_split
				# print(line_split)

				if class_name not in classes_count:
					classes_count[class_name] = 1
				else:
					classes_count[class_name] += 1

				if class_name not in class_mapping:
					if class_name == 'bg' and found_bg == False:
						print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
						found_bg = True
					class_mapping[class_name] = len(class_mapping)

				if filename not in all_imgs:
					all_imgs[filename] = {}
					
					img = cv2.imread(filename)
					# if img is None:
					# 	print(filename)
					# 	print(img)
					(rows,cols) = img.shape[:2]
					# all_imgs[filename]['filepath'] = filename
					all_imgs[filename]['width'] = cols
					all_imgs[filename]['height'] = rows
					all_imgs[filename]['bboxes'] = []
					all_imgs[filename]['imageset'] = 'test'

				all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})


			all_data = []
			for key in all_imgs:
				all_data.append(all_imgs[key])
			
			# make sure the bg class is last in the list
			if found_bg:
				if class_mapping['bg'] != len(class_mapping) - 1:
					key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
					val_to_switch = class_mapping['bg']
					class_mapping['bg'] = len(class_mapping) - 1
					class_mapping[key_to_switch] = val_to_switch
	
	return all_data, classes_count, class_mapping

# all_data, classes_count, class_mapping = get_data('../data/haze_images_labels')

# print()
# print("all data ", all_data)
# print("classes count ", classes_count)
# print("class mapping ", class_mapping)


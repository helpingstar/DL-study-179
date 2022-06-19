import os

from numpy import source
from translate import translate
import random
import cv2


def save_img(save_path, folder_name, image_list):
    new_name = translate[folder_name]
    folder_path = os.path.join(save_path, new_name)
    
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    
    for i, image in enumerate(image_list):
        img = cv2.imread(image)
        
        image_path = os.path.join(folder_path, new_name + "_" + str(i) + ".jpg")
        cv2.imwrite(image_path, img)


if __name__ == "__main__":
    random.seed(100)
    
    BASE_PATH = "."
    BASE_PATH = os.path.abspath(BASE_PATH)  
    source_path = os.path.join(BASE_PATH, "raw-img")
    assert os.path.isdir(source_path)
    
    train_path = os.path.join(BASE_PATH, "train_img")
    test_path = os.path.join(BASE_PATH, "test_img")
    
    if not os.path.isdir(train_path):
        os.mkdir(train_path)
    if not os.path.isdir(test_path):
        os.mkdir(test_path)

    folder_list = os.listdir(source_path)

    if '.DS_Store' in folder_list:
        folder_list.remove('.DS_Store')


    img_set = {}
    for folder in folder_list:
        folder_path = os.path.join(source_path, folder)
        image_list = os.listdir(folder_path)
        
        image_path_list = []
        for image in image_list:
            image_path_list.append(os.path.join(folder_path, image))
        img_set[folder] = image_path_list


    for folder in img_set:
        random.shuffle(img_set[folder])
        train_length = int(len(img_set[folder]) * 0.8)
        
        train_list = img_set[folder][:train_length]
        test_list = img_set[folder][train_length:]

        save_img(train_path, folder, train_list)
        save_img(test_path, folder, test_list)
    
    


    
    
    
    

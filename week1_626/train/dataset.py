import cv2, os, sys
from matplotlib.pyplot import cla
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random

from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2    
import torch
from class_number import CLASS_NUMBER, SMALL_CLASS_NUMBER


train_transform = A.Compose([
                            A.RandomBrightnessContrast(brightness_limit = 0.3, contrast_limit = 0.3, p=1),
                            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=180, p=1),
                            A.ColorJitter(brightness=0, contrast=0, saturation=0.1, hue=0.1, p=1),
                            A.GaussNoise(var_limit = (0.0, 0.05), p=1),
                            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                            ToTensorV2()
                            ])

test_transform = A.Compose([
                            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                            ToTensorV2()
                            ])

class CustomDataset(Dataset):
    def __init__(self, base_path, type, class_dict):
        self.base_path = base_path
        base_path = os.path.abspath(base_path)
        class_list = os.listdir(base_path)
        
        self.type = type
        
        
        data_collection = {}
        for class_name in class_list:
            if class_name in class_dict:
                data_collection[class_name] = []
        
        for class_name in data_collection:
            img_path_list = []
            
            class_path = os.path.join(base_path, class_name)
            img_list = os.listdir(class_path)
            img_list.sort()
            
            for img_ in img_list:
                if "jpg" in img_:
                    img_path_list.append(os.path.join(class_path, img_))
            
            data_collection[class_name] = img_path_list
        
        if type == "train" or type == "valid":
            self.trainset = {}
            self.trainset["path"] = []
            self.trainset["label"] = []
            self.validset = {}
            self.validset["path"] = []
            self.validset["label"] = []
            
            img_path_set = {}
            label_set = {}
            max_length = 0
            
            for class_name in data_collection:
                img_length = len(data_collection[class_name])
                train_img_length = int(img_length * 0.8)
                
                train_img_path = data_collection[class_name][:train_img_length]
                valid_img_path = data_collection[class_name][train_img_length:]
                
                one_hot_vector = np.zeros((1, 10), dtype=np.int32)
                one_hot_vector[0][CLASS_NUMBER[class_name]] = 1
                
                train_label = np.empty((0,10), dtype=np.int32)
                for i in range(len(train_img_path)):
                    train_label = np.append(train_label, one_hot_vector, axis = 0)
                
                img_path_set[class_name] = train_img_path
                label_set[class_name] = train_label
                max_length = max(max_length, len(train_img_path))
                
                valid_label = np.empty((0,10), dtype=np.int32)
                for i in range(len(valid_img_path)):
                    valid_label = np.append(valid_label, one_hot_vector, axis = 0)
                
                # self.trainset["path"].extend(train_img_path)
                # self.trainset["label"].extend(train_label)
                self.validset["path"].extend(valid_img_path)
                self.validset["label"].extend(valid_label)
            
            for class_name in data_collection:
                train_img_path = img_path_set[class_name]
                
                if len(train_img_path) < max_length:
                    over_sample_count = max_length - len(train_img_path)
                    over_sample_list = [random.choice(train_img_path) for i in range(over_sample_count)]
                    
                    img_path_set[class_name].extend(over_sample_list)

                    one_hot_vector = np.zeros((1, 10), dtype=np.int32)
                    one_hot_vector[0][CLASS_NUMBER[class_name]] = 1
                    
                    for i in range(over_sample_count):
                        label_set[class_name] = np.append(label_set[class_name], one_hot_vector, axis = 0)
                    
                self.trainset["path"].extend(img_path_set[class_name])
                self.trainset["label"].extend(label_set[class_name])

            if self.type == "train":
                print("------------")
                for class_name in data_collection:
                    print("| {} -> {} ".format(class_name, len(data_collection[class_name])))
                print("------------")
                print("| Train Set : {}".format(len(self.trainset["path"])))
                print("| Valid Set : {}".format(len(self.validset["path"])))

            self.trainset["path"], self.trainset["label"] = shuffle(self.trainset["path"], self.trainset["label"], random_state=10)
            self.validset["path"], self.validset["label"] = shuffle(self.validset["path"], self.validset["label"], random_state=10)
        else:
            self.testset = {}
            self.testset["path"] = []
            self.testset["label"] = []
            
            for class_name in data_collection:
                test_img_path = data_collection[class_name]
                test_label = [CLASS_NUMBER[class_name]] * len(test_img_path)
                
                one_hot_vector = np.zeros((1, 10), dtype=np.int32)
                one_hot_vector[0][CLASS_NUMBER[class_name]] = 1
                
                test_label = np.empty((0,10), dtype=np.int32)
                for i in range(len(test_img_path)):
                    test_label = np.append(test_label, one_hot_vector, axis = 0)

                self.testset["path"].extend(test_img_path)
                self.testset["label"].extend(test_label)
                
            print("------------")
            for class_name in data_collection:
                print("| {} -> {} ".format(class_name, len(data_collection[class_name])))
            print("------------")
            print("| Test Set : {}".format(len(self.testset["path"])))
            
            self.testset["path"], self.testset["label"] = shuffle(self.testset["path"], self.testset["label"], random_state=10)
            
    def __getitem__(self, index):
        dataset = None
        if self.type == "train":
            dataset = self.trainset
        elif self.type == "valid":
            dataset = self.validset
        elif self.type == "test":
            dataset = self.testset
            
        img_path = dataset["path"][index]
        
        label = dataset["label"][index]
        images = cv2.imread(img_path)
        if self.type == "train":
            transformed = train_transform(image = images)
            image = transformed["image"]
        else:
            transformed = test_transform(image = images)
            image = transformed["image"]
        return image, label


    def __len__(self):
        if self.type == "train":
            return len(self.trainset["path"])
        elif self.type == "valid":
            return len(self.validset["path"])
        else:
            return len(self.testset["path"])


class DataManger():
    def __init__(self, config) -> None:
        self.dataloader = {}
        self.config = config
        self.class_dict = None
        self.all_class_init()
        
    def small_class_init(self):
        self.class_dict = SMALL_CLASS_NUMBER
        
        train_dataset = CustomDataset(self.config["train_image_path"], "train", SMALL_CLASS_NUMBER)
        self.dataloader["train"] = DataLoader(train_dataset, batch_size = self.config["train_batch_size"], shuffle=True, num_workers=4, drop_last=True)
    
        valid_dataset = CustomDataset(self.config["train_image_path"], "valid", SMALL_CLASS_NUMBER)
        self.dataloader["valid"] = DataLoader(valid_dataset, batch_size = self.config["train_batch_size"], shuffle=False, num_workers=4, drop_last=True)
    
        test_dataset = CustomDataset(self.config["test_image_path"], "test", CLASS_NUMBER)
        self.dataloader["test"] = DataLoader(test_dataset, batch_size = self.config["test_batch_size"], shuffle=False, num_workers=4,  drop_last=False)

    def all_class_init(self):
        self.class_dict = CLASS_NUMBER
        train_dataset = CustomDataset(self.config["train_image_path"], "train", CLASS_NUMBER)
        self.dataloader["train"] = DataLoader(train_dataset, batch_size = self.config["train_batch_size"], shuffle=True, num_workers=4, drop_last=True)
    
        valid_dataset = CustomDataset(self.config["train_image_path"], "valid", CLASS_NUMBER)
        self.dataloader["valid"] = DataLoader(valid_dataset, batch_size = self.config["train_batch_size"], shuffle=False, num_workers=4, drop_last=True)
    
        test_dataset = CustomDataset(self.config["test_image_path"], "test", CLASS_NUMBER)
        self.dataloader["test"] = DataLoader(test_dataset, batch_size = self.config["test_batch_size"], shuffle=False, num_workers=4,  drop_last=False)
        

    def get_loader(self, type):
        if type == "train":
            return self.dataloader["train"]
        elif type == "valid":
            return self.dataloader["valid"]
        elif type == "test":
            return self.dataloader["test"]
        else:
            return None
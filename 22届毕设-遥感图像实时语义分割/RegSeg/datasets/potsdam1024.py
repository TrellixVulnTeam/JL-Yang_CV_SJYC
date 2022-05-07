import os
from torch.utils import data
from PIL import Image


class Potsdam1024(data.Dataset):
    def __init__(self, root, split="train", mode="1024", target_type="semantic", transforms=None,
                 class_uniform_pct=0.5):
        assert target_type == "semantic"
        self.root = root
        self.transforms = transforms
        self.mode = mode
        self.image_root = os.path.join(self.root, "2_Ortho_RGB_split_1024")
        self.label_root = os.path.join(self.root, "5_Labels_all_split_1024_fine")
        self.split_root = os.path.join(self.root, "split_1024")
        self.split = split
        self.num_classes = 6

        if split == "train":
            self.split_path = os.path.join(self.split_root, "train_full_c100.txt")
        elif split == "val":
            self.split_path = os.path.join(self.split_root, "val.txt")
        elif split == "test":
            self.split_path = os.path.join(self.split_root, "test.txt")
        else:
            raise RuntimeError("split param not fit.")
        assert os.path.isfile(self.split_path), "{} not a file".format(self.split_path)
        with open(self.split_path, 'r') as file:
            ims = file.readlines()
            file.close()
        self.image_label_list = []
        for i in ims:
            i = i.rstrip('\n')
            i = i.split(',')
            self.image_label_list.append((i[0], i[1]))
        # check
        for image_name, label_name in self.image_label_list:
            assert os.path.isfile(os.path.join(self.image_root, image_name)), "{} not exists".format(image_name)
            assert os.path.isfile(os.path.join(self.label_root, label_name)), "{} not exists".format(label_name)

    def __getitem__(self, index):
        image_name = self.image_label_list[index][0]
        label_name = self.image_label_list[index][1]
        image_path = os.path.join(self.image_root, image_name)
        label_path = os.path.join(self.label_root, label_name)
        assert os.path.isfile(image_path), "image: {} not exists".format(image_path)
        assert os.path.isfile(label_path), "label: {} not exists".format(label_path)
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)
        image, label = self.transforms(image, label)
        return image, label

    def __len__(self):
        return len(self.image_label_list)


if __name__ == '__main__':
    import transforms as T

    transform = T.Compose([
        T.RandomResize(1024, 1024, "uniform"),
        T.ToTensor()
    ])
    dataset = Potsdam1024("root/", transforms=transform)
    image, label = dataset[1]
    print(image.shape)
    print(label.shape)

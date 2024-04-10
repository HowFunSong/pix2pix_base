import torch
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "maps/train"
VAL_DIR = "maps/val"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 4
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 250
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc_maps.pth.tar"
CHECKPOINT_GEN = "gen_maps.pth.tar"

both_transform = transforms.Compose([
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.Resize(size=(256, 256))
                        ])

transform_only_input = transforms.Compose([
                            # transforms.RandomCrop(32, padding=4),
                            # transforms.CenterCrop((256, 256)),
                            transforms.RandomHorizontalFlip(p=0.5),
                            # transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1, hue=0.02),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                        ])

transform_only_mask = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                        ])
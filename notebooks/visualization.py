import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from data_setup import train_image_paths, test_image_paths, transform
from PIL import Image
import matplotlib.pyplot as plt 

# Visualize one image from the training dataset
def visualize_one_image(train_image_paths, index):
    image_path = train_image_paths[index]
    image = Image.open(image_path)
    image = transform(image).permute(1,2,0).numpy()
    plt.imshow(image)
    plt.axis(False)
    plt.show()

visualize_one_image(train_image_paths=train_image_paths, index=10)

    
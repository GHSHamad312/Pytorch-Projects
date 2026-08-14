from torchvision.io import read_image
from torchvision import transforms
import matplotlib.pyplot as plt

image=read_image("./computer_vision/cat.png")
transform=transforms.Resize((255,255))
image=transform(image)
plt.imshow(image.permute(1,2,0))
plt.axis("off")
plt.show()
print(image.shape)
print(image.min())
print(image.max())
from platformer_gen import generator
import cv2
from urllib.request import urlopen
import numpy as np

image = "platform-generation/platform-constraint/images/SuperMarioBros2(J)-World2-3.png"

gnrtr = generator(
    image,
    n=3
)

generated_level = gnrtr.propagate()

cv2.imshow("fixed", generated_level)

cv2.waitKey(0)
cv2.destroyAllWindows()
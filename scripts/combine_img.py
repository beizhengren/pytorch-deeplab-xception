import sys
import os.path
from PIL import Image
def main():
    print("start")


def blend_two_images(img1_path, img2_path ):
    img1 = Image.open( img1_path)
#    if not img1.exists()
    img1 = img1.convert('RGBA')
 
    img2 = Image.open( img2_path)
    img2 = img2.convert('RGBA')
    
    r, g, b, alpha = img2.split()
    alpha = alpha.point(lambda i: i>0 and 204)
 
    img = Image.composite(img2, img1, alpha)
 
    img.show()
    img.save("blend.png")
 
    return

if __name__ == "__main__":
    blend_two_images(sys.argv[1], sys.argv[2])

from PIL import Image
import sys


def main(argv):
    img = Image.open(argv[0])
    img = img.convert("RGB")
    pixdata = img.load()


    for y in range(img.size[1]):
        for x in range(img.size[0]):
            r, g, b = pixdata[x, y]
            pixdata[x, y] = (r//2, g//2, b//2)

    img.save("./Q2.png")


if __name__ == "__main__":
    main(sys.argv[1:])

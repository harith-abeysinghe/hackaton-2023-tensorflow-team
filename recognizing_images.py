import pytesseract as tess
from PIL import Image
from pytesseract import pytesseract

pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

path = "test.jpg"
img = Image.open(path)
text = pytesseract.image_to_string(img)
print(text)

'''
print(recognize("test.jpg"))
print(recognize("test2.jpg"))
print(recognize("test3.jpg"))
print(recognize("test4.jpg"))
print(recognize("test5.jpg"))
print(recognize("test6.jpg"))
print(recognize("cameraphoto.jpg"))
print(recognize("test7.jpg"))
'''


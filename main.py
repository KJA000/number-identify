#이미지 컬러있는 버전으로 나오는 거

'''

import cv2

import numpy as np

img = cv2.imread('./images/meat.jpeg', cv2.IMREAD_GRAYSCALE)


#hist,bins = np.histogram(img.ravel(),256,[0,256])


cv2.imshow('color', img)


cv2.waitKey()

cv2.destroyAllwindows()

'''


#히스토그램

'''import cv2

from matplotlib import pyplot as plt

img = cv2.imread('./images/meat.jpeg')

color = ('b','g','r')

for i,col in enumerate(color):

    histr = cv2.calcHist([img],[i],None,[256],[0,256])

    plt.plot(histr,color = col)

    plt.xlim([0,256])

plt.show()

'''

'''

import cv2

import numpy as np


img = cv2.imread('./images/meat.jpeg')


x=320; y=150; w=50; h=50        # roi 좌표

roi = img[y:y+h, x:x+w]         # roi 지정        ---①


print(roi.shape)                # roi shape, (50,50,3)

cv2.rectangle(roi, (0,0), (h-10, w+100), (0,0,255)) # roi 전체에 사각형 그리기 ---②

cv2.imshow("img", img)


key = cv2.waitKey(0)

print(key)

cv2.destroyAllWindows()

'''

'''

import cv2

import numpy as np

import matplotlib.pylab as plt

# haarcascade 불러오기

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')



# 이미지 불러오기

img = cv2.imread('./images2/face2.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('./images2/face2.jpg')



# 얼굴 찾기

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

k = 0

loc = []

for (x, y, w, h) in faces:

    #face = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    loc.append([x, y])


    #print(loc)

    # 눈 찾기

    roi_color = img[y:y + h, x:x + w]   # jooseok

    roi_gray = gray[y:y + h, x:x + w]


    if k==0:

        roi_first = gray[y:y + h, x:x + w]


    #print(roi_gray)

    cv2.imwrite('./images/result' + str(k) + '.jpg', roi_color)

    k += 1

    for i in range(len(roi_first)):

        for j in range(len(roi_first[i])):

            img[i+y][j+x] = roi_first[i][j]            # 얼굴부분을 한 사람 얼굴로 전부 바꾸기


    eyes = eye_cascade.detectMultiScale(gray)

    for (ex, ey, ew, eh) in eyes:

        roi_color1 = img2[ey:ey+eh, ex:ex+ew]

        for l in range(len(roi_color1)):

            for t in range(len(roi_color1[l])):

                img[l+ey][t+ex] = roi_color1[l][t]    #눈부분만 컬러로 (각얼굴에 맞게)


        #cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)




# 영상 출력

cv2.imshow('image', img)

from matplotlib import pyplot as plt


img = cv2.imread('./images2/face2.jpg',0)

'''

'''

color = ('b','g','r')

for i,col in enumerate(color):

    histr = cv2.calcHist([img],[i],None,[256],[0,256])

    plt.plot(histr,color = col)

    plt.xlim([0,256])

plt.show()

'''

'''

img2 = cv2.equalizeHist(img)

img = cv2.resize(img,(400,400))

img2 = cv2.resize(img2,(400,400))


dst = np.hstack((img, img2))

cv2.imshow('img',dst)

cv2.waitKey()

cv2.destroyAllWindows()

'''

'''


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

img2 = clahe.apply(img)


img = cv2.resize(img,(400,400))

img2 = cv2.resize(img2,(400,400))


dst = np.hstack((img, img2))

cv2.imshow('img',dst)

#print(loc)


key = cv2.waitKey(0)

cv2.destroyAllWindows()



import glob

target_dir = './images/'

files = glob.glob(target_dir+'*.*')

#print(files)


from PIL import Image


def imgresize(files,x_size,y_size):

    file_list = []

    for file in files:

        image = Image.open(file)

        resized_file = image.resize((x_size,y_size))

        file_list.append(resized_file)


thumb_sizeX = 100

thumb_sizeY = 100

file_list = imgresize(files, thumb_sizeX,thumb_sizeY)


def imgMerge(file_list,sizeX,sizeY,numX):

    new_image = Image.new("RGB",(sizeX*numX,sizeY*(len(file_list)//numX+1)),(256,256,256))

    count_X = 0

    count_Y = 0

    for index in range(len(file_list)):

        area = ((count_X*sizeX),count_Y*sizeY,(count_X+1)*sizeX,(count_Y+1)*sizeY)

        new_image.paste(file_list[index],area)


        if(index+1)%numX ==0:

            count_Y+=1

            count_X =0

        else:

            count_X +=1

    new_image.show()

    new_image.save('./images2/face2','jpg')


numX = 3

imgMerge(file_list,thumb_sizeX, thumb_sizeY, numX)

'''

import tensorflow as tf

'''



# 1. MNIST 데이터셋 임포트

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# 2. 데이터 전처리

x_train, x_test = x_train/255.0, x_test/255.0


# 3. 모델 구성

model = tf.keras.models.Sequential([

    tf.keras.layers.Flatten(input_shape=(28, 28)),

    tf.keras.layers.Dense(512, activation=tf.nn.relu),

    tf.keras.layers.Dense(10, activation=tf.nn.softmax)

])


# 4. 모델 컴파일

model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])


# 5. 모델 훈련

model.fit(x_train, y_train, epochs=20)


# 6. 정확도 평가

test_loss, test_acc = model.evaluate(x_test, y_test)

print('테스트 정확도:', test_acc)

'''


'''

import sys

import tensorflow as tf

from PyQt5.QtWidgets import *

from PyQt5.QtGui import *

from PyQt5.QtCore import *

import numpy as np

class MyApp(QMainWindow):


    def __init__(self):

        super().__init__()

        self.image = QImage(QSize(400, 400), QImage.Format_RGB32)

        self.image.fill(Qt.white)

        self.drawing = False

        self.brush_size = 30

        self.brush_color = Qt.black

        self.last_point = QPoint()

        self.loaded_model = None

        self.initUI()


    def initUI(self):

        menubar = self.menuBar()

        menubar.setNativeMenuBar(False)

        filemenu = menubar.addMenu('File')


        load_model_action = QAction('Load model', self)

        load_model_action.setShortcut('Ctrl+L')

        load_model_action.triggered.connect(self.load_model)


        save_action = QAction('Save', self)

        save_action.setShortcut('Ctrl+S')

        save_action.triggered.connect(self.save)


        clear_action = QAction('Clear', self)

        clear_action.setShortcut('Ctrl+C')

        clear_action.triggered.connect(self.clear)


        filemenu.addAction(load_model_action)

        filemenu.addAction(save_action)

        filemenu.addAction(clear_action)


        self.statusbar = self.statusBar()


        self.setWindowTitle('MNIST Classifier')

        self.setGeometry(300, 300, 400, 400)

        self.show()


    def paintEvent(self, e):

        canvas = QPainter(self)

        canvas.drawImage(self.rect(), self.image, self.image.rect())


    def mousePressEvent(self, e):

        if e.button() == Qt.LeftButton:

            self.drawing = True

            self.last_point = e.pos()


    def mouseMoveEvent(self, e):

        if (e.buttons() & Qt.LeftButton) & self.drawing:

            painter = QPainter(self.image)

            painter.setPen(QPen(self.brush_color, self.brush_size, Qt.SolidLine, Qt.RoundCap))

            painter.drawLine(self.last_point, e.pos())

            self.last_point = e.pos()

            self.update()


    def mouseReleaseEvent(self, e):

        if e.button() == Qt.LeftButton:

            self.drawing = False


            arr = np.zeros((28, 28))

            for i in range(28):

                for j in range(28):

                    arr[j, i] = 1 - self.image.scaled(28, 28).pixelColor(i, j).getRgb()[0] / 255.0

            arr = arr.reshape(-1, 28, 28)


            if self.loaded_model:

                pred = self.loaded_model.predict(arr)[0]

                pred_num = str(np.argmax(pred))

                self.statusbar.showMessage('숫자 ' + pred_num + '입니다.')


    def load_model(self):

        fname, _ = QFileDialog.getOpenFileName(self, 'Load Model', '')


        if fname:

            self.loaded_model = tf.keras.models.load_model(fname)

            self.statusbar.showMessage('Model loaded.')


    def save(self):

        fpath, _ = QFileDialog.getSaveFileName(self, 'Save Image', '', "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ")


        if fpath:

            self.image.scaled(28, 28).save(fpath)


    def clear(self):

        self.image.fill(Qt.white)

        self.update()

        self.statusbar.clearMessage()



if __name__ == '__main__':

    app = QApplication(sys.argv)

    ex = MyApp()

    sys.exit(app.exec_())

'''


import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

import matplotlib.pyplot as plt


#from google.colab import auth

#from google.colab import drive


import cv2

import numpy as np



#drive.mount('/content/gdrive/') # google drive address


#import glob

import os

scr = os.listdir('./numbers/')


'''

plt.title('The number I wrote')

plt.imshow(myNum, cmap='gray')

print(myNum.shape)

'''



# data input

mnist = tf.keras.datasets.mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train/255.0, x_test/255.0

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

#model

model = tf.keras.models.Sequential([

  tf.keras.layers.Flatten(input_shape=(28, 28)),

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(10, activation='softmax')

])


model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])


model.summary()


model.fit(x_train, y_train, epochs=1)

model.evaluate(x_test, y_test)


# test my Image data


#myNum = np.array(myNum)


#v = model.predict(np.reshape(myNum, (1,28,28)))

#print(v)

#print(max(v[0]))


sumplus = 0
sumgop =1

dataVV = []
lista = []
#for i in range(3):

for i in range(len(scr)):

    #path = os.path.join('./number/'+str(scr[i]))

    src = cv2.imread('./numbers/'+str(scr[i]), cv2.IMREAD_GRAYSCALE)

    dataVV.append(src)


for i in range(len(dataVV)):

    ret, binary = cv2.threshold(dataVV[i], 170, 255, cv2.THRESH_BINARY_INV)

    scr = np.asarray(cv2.resize(binary,dsize=(28, 28), interpolation=cv2.INTER_AREA))/255

    myNum = np.array(scr)

    v = model.predict(np.reshape(myNum, (1, 28, 28)))

    s = list(v[0]).index(max(v[0]))
    lista.append(s)

for i in range(len(lista)):
    print(lista[i],sep='\n')
    sumplus += lista[i]
    sumgop *= lista[i]

print(sumplus,sumgop, sep = ',', end="!!!!")

#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Sencilla clase de lectura genérica de secuencias de imágenes de entrada.
Puede leer de una sóla imagen de un archivo, o bien de una secuencia de éstas,
o bien de un vídeo, o finalmente de una cámara. Se pueden especificar todos
estos tipos de fuentes en la cadena de inicialización, así como un reescalado
deseado (rows x cols), o si se desea que la fuente de imágenes cicle (vuelva
al comienzo) al terminar, o bien directamente termine (en cuyo caso el método
read simplemente devolverá None).

Nota 1: Si los parámetros de inicialización no se ponen (o bien se ponen, pero
explícitamente se hacen cero) entonces la imagen de entrada no se escala, y
se proporciona según su tamaño original.

Nota 2: La implementación prioriza la sencillez y legibilidad del código, de
manera que apenas se realiza control de errores.

Ejemplos posibles de cadenas válidas de inicialización:

Lectura de archivo de vídeo sin escalar, y repitiendo al acabar (bucle):
"/path/to/file.mp4:loop"

Equivalente al anterior:
"/path/to/file.mp4:rows=0:cols=0:loop"

Similar, pero con otro tipo de contenedor (se soporta casi cualquier tipo de
vídeo):
"/path/to/file.avi"

Lectura de secuencia de imágenes escalando sólo la anchura, y bucle al acabar:
"/path/to/img-???.jpg:rows=0:cols=200:loop"

Lectura de una sola imagen, escalada. Si se intenta una segunda lectura,
devolverá None (al no haber bucle):
"/path/to/img-001.jpg:rows=300:cols=400"

Lectura de secuencia de imágenes, cada una a su tamaño original, y sin bucle
(al acabarse las mismas el método read devolverá None):
"/path/to/img-*.jpg"

Lectura de la cámara numerada como 0:
"0"

Lectura de la cámara numerada como 1, con reescalado:
"1:rows=100:cols=200"


Para ver ejemplo de utilización, consultar el programa ejemplosimple.py:
Uso:
  ./ejemplosimple.py [<fuente de video>]
'''
import glob
import cv2


class VideoInput(object):
    def __init__(self, str):
        largs = map(lambda x: tuple(x.split('=')), str.split(':'))
        self.cols, self.rows = 0, 0
        self.loop = False
        self.type = 'camera'
        self.camera = 0
        for arg in largs:
            if len(arg) == 1:
                if arg[0] == 'loop':
                    self.loop = True
                elif arg[0].lower().endswith(('.jpg', '.jpeg',
                                              '.png', '.gif')):
                    self.type = 'imgfiles'
                    self.imgfiles = glob.glob(arg[0])
                    self.imgfiles.sort()
                    self.curframe = 0
                elif arg[0].lower().endswith(('.mpeg', '.mpg', '.dv', '.wmv',
                                              '.avi', '.mp4', '.webm', '.mkv')):
                    self.type = 'videofile'
                    self.videofile = arg[0]
                    self.cap = cv2.VideoCapture(self.videofile)
                elif arg[0].isdigit():
                    self.type = 'camera'
                    self.camera = int(arg[0])
                    self.cap = cv2.VideoCapture(self.camera)
            else:
                var, val = arg
                if(var == 'cols'):
                    self.cols = int(val)
                elif(var == 'rows'):
                    self.rows = int(val)
        if self.cols != 0 and self.type == 'camera':
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cols)
        if self.rows != 0 and self.type == 'camera':
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.rows)

    def read(self):
        if self.type in ('camera', 'videofile'):
            flag, frame = self.cap.read()
            if self.loop and frame is None:
                self.cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
                flag, frame = self.cap.read()
        elif self.type in ('imgfiles', ):
            if self.curframe == len(self.imgfiles):
                if self.loop:
                    self.curframe = 0
                else:
                    return None
            frame = cv2.imread(self.imgfiles[self.curframe])
            self.curframe += 1
        if frame is not None:  # Posible escalado
            if self.cols != 0 and self.rows != 0:
                frame = cv2.resize(frame, (self.cols, self.rows))
            elif self.cols != 0:
                frame = cv2.resize(frame, (self.cols, frame.shape[0]))
            elif self.rows != 0:
                frame = cv2.resize(frame, (frame.shape[1], self.rows))
        return frame

    def close(self):
        if self.type in ('videofile', 'camera'):
            # try:
                self.cap.release()

# -*- coding: utf-8 -*-
"""
Created on Sun May 10 11:17:13 2015

@author: José María Sola Durán
"""

import cv2
import os
import numpy as np
import utilscv

# Se ha creado una clase Python, llamada ImageFeature
# que contendrá para cada una de las imágenes de la base de datos,
# la información necesaria para computar el reconocimiento de objetos.
class ImageFeature(object):
    def __init__(self, nameFile, shape, imageBinary, kp, desc):
        #Nombre del fichero
        self.nameFile = nameFile
        #Shape de la imagen
        self.shape = shape
        #Datos binarios de la imagen
        self.imageBinary = imageBinary
        #KeyPoints de la imagen una vez aplicado el algoritmo de detección de features
        self.kp = kp
        #Descriptores de las features detectadas
        self.desc = desc
        #Matchings de la imagen de la base de datos con la imagen de la webcam
        self.matchingWebcam = []
        #Matching de la webcam con la imagen actual de la base de datos.
        self.matchingDatabase = []
    #Permite vaciar los matching calculados con anterioridad, para una nueva imagen
    def clearMatchingMutuos(self):
        self.matchingWebcam = []
        self.matchingDatabase = []

#Funcion encargada de calcular, para cada uno de los métodos de calculo de features,
#las features de cada una de las imagenes del directorio "modelos"
def loadModelsFromDirectory():
    #El método devuelve un diccionario. La clave es el algoritmo de features
    #mientras que el valor es una lista con objetos del tipo ImageFeature
    #donde se almacenan todos los datos de las features de las imagenes de la
    #Base de datos.
    dataBase = dict([('SIFT', []), ('AKAZE', []), ('SURF', []), 
                     ('ORB', []), ('BRISK', [])])
    #Se ha limitado el número de features a 250, para que el algoritmo vaya fluido.
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=250)
    akaze = cv2.AKAZE_create()
    surf = cv2.xfeatures2d.SURF_create(800)
    orb = cv2.ORB_create(400)
    brisk = cv2.BRISK_create()
    for imageFile in os.listdir("modelos"):
        #Se carga la imagen con la OpenCV
        colorImage = cv2.imread("modelos/" + str(imageFile))
        #Pasamos la imagen a escala de grises
        currentImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)
        #Realizamos un resize de la imagen, para que la imagen comparada sea igual
        kp, desc = sift.detectAndCompute(currentImage, None)
        #Se cargan las features con SIFT
        dataBase["SIFT"].append(ImageFeature(imageFile, currentImage.shape, colorImage, kp, desc))
        #Se cargan las features con AKAZE
        kp, desc = akaze.detectAndCompute(currentImage, None)
        dataBase["AKAZE"].append(ImageFeature(imageFile, currentImage.shape, colorImage, kp, desc))
        #Se cargan las features con SURF
        kp, desc = surf.detectAndCompute(currentImage, None)
        dataBase["SURF"].append(ImageFeature(imageFile, currentImage.shape, colorImage, kp, desc))
        #Se cargan las features con ORB
        kp, desc = orb.detectAndCompute(currentImage, None)
        dataBase["ORB"].append(ImageFeature(imageFile, currentImage.shape, colorImage, kp, desc))
         #Se cargan las features con BRISK
        kp, desc = brisk.detectAndCompute(currentImage, None)
        dataBase["BRISK"].append(ImageFeature(imageFile, currentImage.shape, colorImage, kp, desc))
    return dataBase
    
#Función encargada de calcular los Matching mutuos, pero anidando bucles
#Es una solución muy lenta porque no se aprovecha potencia de Numpy
#Ni siquiera ponemos un slider para utilizar este método ya que es muy lento

def findMatchingMutuos(selectedDataBase, desc, kp):
    for imgFeatures in selectedDataBase:
        imgFeatures.clearMatchingMutuos()
        for i in range(len(desc)):
            primerMatching = None
            canditatoDataBase = None
            matchingSegundo = None
            candidateWebCam = None
            for j in range(len(imgFeatures.desc)):
                valorMatching = np.linalg.norm(desc[i] - imgFeatures.desc[j])
                if (primerMatching is None or valorMatching < primerMatching):
                    primerMatching = valorMatching
                    canditatoDataBase = j
            for k in range(len(desc)):
                valorMatching = np.linalg.norm(imgFeatures.desc[canditatoDataBase] - desc[k])
                if (matchingSegundo is None or valorMatching < matchingSegundo):
                    matchingSegundo = valorMatching
                    candidateWebCam = k
            if not candidateWebCam is None and i == candidateWebCam:
                imgFeatures.matchingWebcam.append(kp[i].pt)
                imgFeatures.matchingDatabase.append(imgFeatures.kp[canditatoDataBase].pt)
    return selectedDataBase

#Función encargada de calcular los matching mutuos de una imagen de la webcam,
#con todas las imagenes de la base de datos. Recibe como parámetro de entrada
#la base de datos en función del método de calculo de features utilizado
#en la imagen entrada de la webcam.
def findMatchingMutuosOptimizado(selectedDataBase, desc, kp):
    #El algoritmo se repite para cada imagen de la base de datos.
    for img in selectedDataBase:
        img.clearMatchingMutuos()
        for i in range(len(desc)):
             #Se calcula la norma de la diferencia del descriptor actual, con todos
             #los descriptores de la imagen de la base de datos. Conseguimos
             #sin bucles y haciendo uso del broadcasting de Numpy, todas las distancias
             #entre el descriptor actual con todos los descriptores de la imagen actual
             distanceListFromWebCam = np.linalg.norm(desc[i] - img.desc, axis=-1)
             #Se obtiene el candidato que está a menor distancia del descriptor actual
             candidatoDataBase = distanceListFromWebCam.argmin() 
             #Se comprueba si el matching es mutuo, es decir, si se cumple 
             # en el otro sentido. Es decir, se comprueba que el candidatoDatabase 
             #tiene al descriptor actual como mejor matching
             distanceListFromDataBase = np.linalg.norm(img.desc[candidatoDataBase] - desc,
                                           axis=-1)
             candidatoWebCam = distanceListFromDataBase.argmin()
             #Si se cumple el matching mutuo, se almacena para despues tratarlo
             if (i == candidatoWebCam):
                img.matchingWebcam.append(kp[i].pt)
                img.matchingDatabase.append(img.kp[candidatoDataBase].pt)
        #Por comodidad se convierten en Numpy ND-Array
        img.matchingWebcam = np.array(img.matchingWebcam)
        img.matchingDatabase = np.array(img.matchingDatabase)
    return selectedDataBase

# Esta funcion calcula la mejor imagen en función del numero de inliers 
# que tiene cada imagen de la base de datos con la imagen obtenida de 
# la camara web.
def calculateBestImageByNumInliers(selectedDataBase, projer, minInliers):
    if minInliers < 15:
        minInliers = 15
    bestIndex = None
    bestMask = None
    numInliers = 0
    #Para cada una de las imagenes
    for index, imgWithMatching in enumerate(selectedDataBase):
        #Se computa el algoritmo RANSAC para calcular el numero de inliers
        _, mask = cv2.findHomography(imgWithMatching.matchingDatabase, 
                                     imgWithMatching.matchingWebcam, cv2.RANSAC, projer)
        if not mask is None:
            #Se comprueba, a partir de la mascara el número de inliers.
            #Si el número de inliers es superior al mínimo número de inliers,
            #y es un máximo (tiene más inliers que la imagen anterior), 
            #entonces se considera que es la imagen que cuadra con el objeto
            #almacenado en la base de datos.
            countNonZero = np.count_nonzero(mask)
            if (countNonZero >= minInliers and countNonZero > numInliers):
                numInliers = countNonZero
                bestIndex = index
                bestMask = (mask >= 1).reshape(-1)
    #Si se ha obtenido alguna imagen como la mejor imagen y, por tanto, 
    #debe tener un número mínimo de inlers, entonces se calculas finalmente
    #los keypoints que son inliers a partir de la mask obtenida en findHomography
    #y se devuelve como mejor imagen.
    if not bestIndex is None:
        bestImage = selectedDataBase[bestIndex]
        inliersWebCam = bestImage.matchingWebcam[bestMask]
        inliersDataBase = bestImage.matchingDatabase[bestMask]
        return bestImage, inliersWebCam, inliersDataBase
    return None, None, None
                
#Esta función calcula la matriz de afinidad A, pinta un rectángulo alrededor
#del objeto detectado y pinta en una nueva ventana la imagen de la base datos
#correspondiente al objeto reconocido.
def calculateAffinityMatrixAndDraw(bestImage, inliersDataBase, inliersWebCam, imgout):
    #Se calcula la matriz de afinidad A
    A = cv2.estimateRigidTransform(inliersDataBase, inliersWebCam, fullAffine=True)
    A = np.vstack((A, [0, 0, 1]))
    
    #Se calculan los puntos del rectangulo que ocupa el objeto reconocido
    a = np.array([0, 0, 1], np.float)
    b = np.array([bestImage.shape[1], 0, 1], np.float)
    c = np.array([bestImage.shape[1], bestImage.shape[0], 1], np.float)
    d = np.array([0, bestImage.shape[0], 1], np.float)
    centro = np.array([float(bestImage.shape[0])/2, 
       float(bestImage.shape[1])/2, 1], np.float)
       
    #Se multiplican los puntos del espacio virtual, para convertirlos en
    #puntos reales de la imagen
    a = np.dot(A, a)
    b = np.dot(A, b)
    c = np.dot(A, c)
    d = np.dot(A, d)
    centro = np.dot(A, centro)
    
    #Se deshomogeneizan los puntos
    areal = (int(a[0]/a[2]), int(a[1]/b[2]))
    breal = (int(b[0]/b[2]), int(b[1]/b[2]))
    creal = (int(c[0]/c[2]), int(c[1]/c[2]))
    dreal = (int(d[0]/d[2]), int(d[1]/d[2]))
    centroreal = (int(centro[0]/centro[2]), int(centro[1]/centro[2]))
    
    #Se pinta el polígono y el nombre del fichero de la imagen en el centro del polígono
    points = np.array([areal, breal, creal, dreal], np.int32)
    cv2.polylines(imgout, np.int32([points]),1, (255,255,255), thickness=2)
    utilscv.draw_str(imgout, centroreal, bestImage.nameFile.upper())
    #Se visualiza el objeto detectado en una ventana a parte
    cv2.imshow('ImageDetector', bestImage.imageBinary)

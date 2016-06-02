#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Paquetes estándar utilizados:
import sys
import cv2
import time

# Paquetes propios:
import videoinput
import utilscv
import objrecogn as orec

# Programa principal:
if __name__ == '__main__':

    # Creación de ventana y sliders asociados, y callback del ratón:
    def nothing(*arg):
        pass
    cv2.namedWindow('Features')
    cv2.namedWindow('ImageDetector')
    #Selección del method para computar las features
    cv2.createTrackbar('method', 'Features', 0, 4, nothing)
    #Error de reproyección para calcular los inliers con RANSAC
    cv2.createTrackbar('projer', 'Features', 5, 10, nothing)
    #Numero de inliers mínimo para indicar que se ha reconocido un objeto
    cv2.createTrackbar('inliers', 'Features', 20, 50, nothing)
    #Trackbar para indicar si se pintan las features o no
    cv2.createTrackbar('drawKP', 'Features', 0, 1, nothing)

    # Apertura de fuente de vídeo:
    if len(sys.argv) > 1:
        strsource = sys.argv[1]
    else:
        strsource = '0:rows=300:cols=400'  # Simple apertura de la cámara cero, sin escalado
    videoinput = videoinput.VideoInput(strsource)
    paused = False
    methodstr = 'None'

    #Cargamos la base de datos de los modelos
    dataBaseDictionary = orec.loadModelsFromDirectory()
    
    while True:
        # Lectura frame de entrada, y parámetros del interfaz:
        if not paused:
            frame = videoinput.read()
        if frame is None:
            print('End of video input')
            break

        # Creación del detector de features, según método (sólo al principio):
        method = cv2.getTrackbarPos('method', 'Features')
        if method == 0:
            if methodstr != 'SIFT':
                methodstr = 'SIFT'
                detector = cv2.xfeatures2d.SIFT_create(nfeatures=250)
        elif method == 1:
            if methodstr != 'AKAZE':
                methodstr = 'AKAZE'
                detector = cv2.AKAZE_create()
        elif method == 2:
            if methodstr != 'SURF':
                methodstr = 'SURF'
                detector = cv2.xfeatures2d.SURF_create(800)
        elif method == 3:
            if methodstr != 'ORB':
                methodstr = 'ORB'
                detector = cv2.ORB_create(400)
        elif method == 4:
            if methodstr != 'BRISK':
                detector = cv2.BRISK_create()
                methodstr = 'BRISK'
                
        # Pasamos imagen de entrada a grises:
        imgin = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Se calcula la imagen de salida
        imgout = frame.copy()
        # Detectamos features, y medimos tiempo:
        t1 = time.time()
        kp, desc = detector.detectAndCompute(imgin, None)
        selectedDataBase = dataBaseDictionary[methodstr]
        if len(selectedDataBase) > 0:
            #Realizamos el matching mutuo
            imgsMatchingMutuos = orec.findMatchingMutuosOptimizado(selectedDataBase, desc, kp)    
            minInliers = int(cv2.getTrackbarPos('inliers', 'Features'))
            projer = float(cv2.getTrackbarPos('projer', 'Features'))
            #Se calcula la mejor imagen en función del numero de inliers. 
            #La mejor imagen es aquella que tiene más número de inliers, pero siempre
            #superando el mínimo que se indica en el trackbar 'minInliers'
            bestImage, inliersWebCam, inliersDataBase =  orec.calculateBestImageByNumInliers(selectedDataBase, projer, minInliers)            
            if not bestImage is None:
                #Si encontramos una buena imagen, se calcula la matriz de afinidad y se pinta en pantalla el objeto reconocido.
               orec.calculateAffinityMatrixAndDraw(bestImage, inliersDataBase, inliersWebCam, imgout)
               
        t1 = 1000 * (time.time() - t1)  # Tiempo en milisegundos
        # Obtener dimensión de descriptores de cada feature:
        if desc is not None:
            if len(desc) > 0:
                dim = len(desc[0])
            else:
                dim = -1
        # Dibujamos features, y escribimos texto informativo sobre la imagen
        # Solo se dibuban las features si el slider lo indica
        if (int(cv2.getTrackbarPos('drawKP', 'Features')) > 0):
            cv2.drawKeypoints(imgout, kp, imgout,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        utilscv.draw_str(imgout, (20, 20),
                         "Method {0}, {1} features found, desc. dim. = {2} ".
                         format(methodstr, len(kp), dim))
        utilscv.draw_str(imgout, (20, 40), "Time (ms): {0}".format(str(t1)))
        # Mostrar resultados y comprobar teclas:
        cv2.imshow('Features', imgout)
        ch = cv2.waitKey(5) & 0xFF
        if ch == 27:  # Escape termina
            break
        elif ch == ord(' '):  # Barra espaciadora pausa
            paused = not paused
        elif ch == ord('.'):  # Punto avanza un sólo frame
            paused = True
            frame = videoinput.read()

    # Cerrar ventana(s) y fuente(s) de vídeo:
    videoinput.close()
    cv2.destroyAllWindows()

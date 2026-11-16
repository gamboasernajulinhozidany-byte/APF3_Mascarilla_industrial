import cv2

# ---------------------------------------------------------------
# DETECTOR DE MASCARILLA INDUSTRIAL - V1 (rectángulo exacto de la detección)
# Para un modelo entrenado con muestras de 40 x 32 (proporción 1.25 : 1)
# ---------------------------------------------------------------

IP_CELULAR = "192.168.1.37"
PUERTO = 4747
URLS_PRUEBA = [
    f"http://{IP_CELULAR}:{PUERTO}/video",
    f"http://{IP_CELULAR}:{PUERTO}/mjpegfeed",
    f"http://{IP_CELULAR}:{PUERTO}/videofeed",
]

ARCHIVO_MODELO = "cascade.xml"
ETIQUETA = "Mascarilla industrial"

# Parámetros de detección (ajústalos según tus pruebas):
SCALE_FACTOR = 1.1        # más cercano a 1 = más lento pero más sensible
MIN_NEIGHBORS = 8         # más alto = menos falsos positivos, pero puede perder detecciones
MIN_SIZE = (50, 40)       # mantiene la proporción 1.25 : 1 de la muestra

cap = None
for url in URLS_PRUEBA:
    print("Probando:", url)
    prueba = cv2.VideoCapture(url)
    if prueba.isOpened():
        ok, _ = prueba.read()
        if ok:
            print("CONECTADO con:", url)
            cap = prueba
            break
    prueba.release()

if cap is None:
    print("No se pudo conectar con DroidCam. Probando la webcam de la PC...")
    prueba = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if prueba.isOpened():
        cap = prueba

if cap is None:
    print("ERROR: no se pudo abrir ninguna cámara.")
    raise SystemExit

clasificador = cv2.CascadeClassifier(ARCHIVO_MODELO)
if clasificador.empty():
    print(f"ERROR: no se pudo cargar '{ARCHIVO_MODELO}' (debe estar junto a este script).")
    raise SystemExit

while True:

    ret, frame = cap.read()
    if not ret:
        print("No se recibe imagen de la cámara.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mascarillas = clasificador.detectMultiScale(gray,
                                                scaleFactor=SCALE_FACTOR,
                                                minNeighbors=MIN_NEIGHBORS,
                                                minSize=MIN_SIZE)

    for (x, y, w, h) in mascarillas:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, ETIQUETA, (x, max(y - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, f"Detecciones: {len(mascarillas)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

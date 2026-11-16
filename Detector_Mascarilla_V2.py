import os
import re
import threading
import time
import cv2
import numpy as np

# ---------------------------------------------------------------
# DETECTOR DE MASCARILLA INDUSTRIAL - V2 (corregido, con diagnóstico)
# Modelo entrenado con muestras de 40 x 32 (proporción 1.25 : 1).
#
# Base fluida: hilo de cámara + hilo de detección, SIN barras de ajuste.
#
# Qué se corrigió respecto a la versión anterior:
#   - Lee del cascade.xml el tamaño de ventana con el que se entrenó y el número de
#     etapas, y los muestra en pantalla (así se ve si se cargó el modelo equivocado).
#   - Si el hilo de detección fallaba, lo hacía en silencio (0 detecciones para siempre);
#     ahora el error se imprime y se muestra en pantalla.
#   - Se detecta a mayor resolución (antes se reducía a 480 px y la mascarilla podía
#     quedar más pequeña que la ventana del modelo) y con pasos de escala más finos.
#   - El tamaño mínimo ya no puede ser menor que la ventana del modelo.
#   - Modo DIAGNÓSTICO (tecla D): minNeighbors = 1 y muestra todos los candidatos.
#   - Opción de ecualizar el histograma (tecla E) para mejorar con poca luz.
#
# Teclas:  R = cambiar giro | F = pantalla completa / ventana
#          + / - = subir / bajar minNeighbors | D = diagnóstico | E = ecualizar | ESC = salir
# ---------------------------------------------------------------

# ---------------- PARÁMETROS ----------------
IP_CELULAR = "192.168.1.37"
PUERTO = 4747
URLS_PRUEBA = [
    f"http://{IP_CELULAR}:{PUERTO}/video",
    f"http://{IP_CELULAR}:{PUERTO}/mjpegfeed",
    f"http://{IP_CELULAR}:{PUERTO}/videofeed",
]

ARCHIVO_MODELO = "cascade.xml"
ETIQUETA = "Mascarilla industrial"
MUESTRA_ENTRENADA = (40, 32)   # tamaño de las positivas (debe coincidir con el entrenamiento)

SCALE_FACTOR = 1.05       # pasos de escala finos = más sensible (1.1 es más rápido)
MIN_NEIGHBORS = 3         # sube con '+' si salen falsos positivos, baja con '-' si no detecta
PESO_MIN = 0.0            # sin filtro por peso; súbelo si quieres descartar detecciones débiles
MAX_CUADROS = 2           # máximo de detecciones mostradas
MARGEN = 0.10             # el recuadro se agranda 10 % alrededor de la detección

ANCHO_PROCESO = 640       # ancho máximo (px) de la imagen usada SOLO para detectar
MIN_SIZE_REAL = None      # None = usa el tamaño de la ventana del modelo como mínimo;
                          # o un (ancho, alto) en px reales, p. ej. (60, 48), para ignorar objetos lejanos

ECUALIZAR_INICIAL = False # True = ecualiza el histograma de la imagen en gris (tecla E)

PANTALLA_COMPLETA = True
GIROS = [
    ("sin girar", None),
    ("90 grados derecha", cv2.ROTATE_90_CLOCKWISE),
    ("90 grados izquierda", cv2.ROTATE_90_COUNTERCLOCKWISE),
]
# --------------------------------------------


def tamano_pantalla():
    """Tamaño de la pantalla de Windows; si falla, 1920 x 1080."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


SW, SH = tamano_pantalla()

# ---------------------------------------------------------------
# MODELO Y CÁMARA
# ---------------------------------------------------------------
CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))


def buscar_modelo(nombre):
    """Busca el modelo junto al script, en 'classifier/', en la carpeta de trabajo
    actual y, por último, en cualquier subcarpeta del script."""
    candidatos = [
        os.path.join(CARPETA_SCRIPT, nombre),
        os.path.join(CARPETA_SCRIPT, "classifier", nombre),
        os.path.join(os.getcwd(), nombre),
    ]
    for ruta in candidatos:
        if os.path.isfile(ruta):
            return ruta
    for raiz, _, archivos in os.walk(CARPETA_SCRIPT):
        if nombre in archivos:
            return os.path.join(raiz, nombre)
    return None


def info_modelo(ruta):
    """Lee del cascade.xml: (ancho, alto) de la ventana, número de etapas y tipo de rasgos."""
    try:
        with open(ruta, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
    except Exception:
        return None, 0, "?"
    cabecera = txt[:4000]
    w = re.search(r"<width>\s*(\d+)\s*</width>", cabecera)
    h = re.search(r"<height>\s*(\d+)\s*</height>", cabecera)
    tipo = re.search(r"<featureType>\s*(\w+)\s*</featureType>", cabecera)
    etapas = len(re.findall(r"<maxWeakCount>", txt))
    ventana = (int(w.group(1)), int(h.group(1))) if (w and h) else None
    return ventana, etapas, (tipo.group(1) if tipo else "?")


ruta_modelo = buscar_modelo(ARCHIVO_MODELO)
if ruta_modelo is None:
    print(f"ERROR: no se encontró '{ARCHIVO_MODELO}'.")
    print("Se buscó en:", CARPETA_SCRIPT)
    raise SystemExit
print("Modelo cargado desde:", ruta_modelo)

clasificador = cv2.CascadeClassifier(ruta_modelo)
if clasificador.empty():
    print(f"ERROR: se encontró '{ruta_modelo}' pero OpenCV no pudo leerlo.")
    raise SystemExit

VENTANA, N_ETAPAS, TIPO_RASGOS = info_modelo(ruta_modelo)
if VENTANA is None:
    VENTANA = MUESTRA_ENTRENADA
    print("AVISO: no se pudo leer la ventana del modelo; se asume", VENTANA)
print(f"Modelo: ventana {VENTANA[0]}x{VENTANA[1]} | {N_ETAPAS} etapas | rasgos {TIPO_RASGOS}")

if abs(VENTANA[0] / VENTANA[1] - MUESTRA_ENTRENADA[0] / MUESTRA_ENTRENADA[1]) > 0.15:
    print("*" * 70)
    print(f"ATENCIÓN: la ventana del modelo es {VENTANA[0]}x{VENTANA[1]}, pero tus positivas")
    print(f"son de {MUESTRA_ENTRENADA[0]}x{MUESTRA_ENTRENADA[1]}. Puede ser un cascade.xml viejo (p. ej. el de lentes)")
    print("o se entrenó con otro 'Sample width/height'. Reentrena con 40 y 32.")
    print("*" * 70)
if 0 < N_ETAPAS < 8:
    print(f"AVISO: el modelo tiene solo {N_ETAPAS} etapas; es poco y puede dar muchos falsos positivos o no detectar.")

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

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


class LectorCamara:
    """Lee la cámara en un hilo aparte y guarda solo el cuadro más reciente."""

    def __init__(self, captura):
        self.captura = captura
        self.frame = None
        self.n = 0
        self.ok = True
        self.lock = threading.Lock()
        threading.Thread(target=self._leer, daemon=True).start()

    def _leer(self):
        while self.ok:
            ret, f = self.captura.read()
            if not ret:
                self.ok = False
                break
            with self.lock:
                self.frame = f
                self.n += 1

    def ultimo(self):
        with self.lock:
            return self.frame, self.n


lector = LectorCamara(cap)

# Estado compartido entre el hilo principal y el de detección
estado = {
    "salir": False,
    "giro": 0,
    "min_neighbors": MIN_NEIGHBORS,
    "diag": False,
    "ecualizar": ECUALIZAR_INICIAL,
    "resultado": (0, []),     # (giro con el que se calculó, [(x, y, w, h, peso)])
    "ms": 0.0,                # lo que tarda cada detección (prueba de que el hilo está vivo)
    "error": "",
}


def detectar(gris, min_neighbors, min_size):
    try:
        rects, _, pesos = clasificador.detectMultiScale3(
            gris,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=min_neighbors,
            minSize=min_size,
            outputRejectLevels=True,
        )
        pesos = np.array(pesos).flatten()
        if len(pesos) != len(rects):
            pesos = np.zeros(len(rects))
    except cv2.error:
        # Respaldo: si detectMultiScale3 no funciona con este modelo, se usa el método simple
        rects = clasificador.detectMultiScale(
            gris,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )
        pesos = np.zeros(len(rects))
    return [(int(x), int(y), int(w), int(h), float(p))
            for (x, y, w, h), p in zip(rects, pesos)]


def trabajador_deteccion():
    ultimo_proc = -1
    while not estado["salir"]:
        frame, n = lector.ultimo()
        if frame is None or n == ultimo_proc:
            time.sleep(0.005)
            continue
        ultimo_proc = n

        try:
            t0 = time.time()
            giro_id = estado["giro"]
            diag = estado["diag"]
            min_neighbors = 1 if diag else estado["min_neighbors"]
            maximo = 10 if diag else MAX_CUADROS

            codigo = GIROS[giro_id][1]
            rot = cv2.rotate(frame, codigo) if codigo is not None else frame
            h, w = rot.shape[:2]
            escala = min(1.0, ANCHO_PROCESO / w)   # solo se reduce, nunca se amplía
            if escala < 1.0:
                pequeno = cv2.resize(rot, (int(w * escala), int(h * escala)),
                                     interpolation=cv2.INTER_AREA)
            else:
                pequeno = rot
            gris = cv2.cvtColor(pequeno, cv2.COLOR_BGR2GRAY)
            if estado["ecualizar"]:
                gris = cv2.equalizeHist(gris)

            # El mínimo nunca puede ser menor que la ventana con que se entrenó el modelo
            if MIN_SIZE_REAL is None:
                min_size = (VENTANA[0], VENTANA[1])
            else:
                min_size = (max(int(MIN_SIZE_REAL[0] * escala), VENTANA[0]),
                            max(int(MIN_SIZE_REAL[1] * escala), VENTANA[1]))

            finales = []
            for (x, y, dw, dh, p) in detectar(gris, min_neighbors, min_size):
                if p >= PESO_MIN:
                    finales.append((int(x / escala), int(y / escala),
                                    int(dw / escala), int(dh / escala), p))
            finales.sort(key=lambda d: d[4], reverse=True)
            estado["resultado"] = (giro_id, finales[:maximo])
            estado["ms"] = (time.time() - t0) * 1000
            estado["error"] = ""
        except Exception as e:
            estado["error"] = f"{type(e).__name__}: {e}"
            print("ERROR en la detección:", estado["error"])
            time.sleep(0.5)


threading.Thread(target=trabajador_deteccion, daemon=True).start()

# ---------------------------------------------------------------
# VENTANA (sin barras)
# ---------------------------------------------------------------
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
if PANTALLA_COMPLETA:
    cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

giro_actual = 0
completa = PANTALLA_COMPLETA
ultimo_n = -1
lienzo = None
dims_previas = None
fps = 0.0
t_anterior = time.time()

while True:
    frame, n = lector.ultimo()

    if frame is not None and n != ultimo_n:
        ultimo_n = n
        estado["giro"] = giro_actual

        # Girar y reducir la imagen al tamaño de pantalla ANTES de dibujar
        nombre_giro, codigo = GIROS[giro_actual]
        rot = cv2.rotate(frame, codigo) if codigo is not None else frame
        h, w = rot.shape[:2]
        esc = min(SW / w, SH / h) if completa else min(1.0, (SW * 0.9) / w, (SH * 0.8) / h)
        vista = cv2.resize(rot, (max(int(w * esc), 1), max(int(h * esc), 1)))
        vh, vw = vista.shape[:2]

        # Dibujar las detecciones calculadas por el otro hilo
        diag = estado["diag"]
        color = (0, 255, 255) if diag else (0, 255, 0)
        giro_res, lista = estado["resultado"]
        if giro_res == giro_actual:
            for (x, y, dw, dh, p) in lista:
                x, y, dw, dh = int(x * esc), int(y * esc), int(dw * esc), int(dh * esc)
                mx, my = int(dw * MARGEN), int(dh * MARGEN)
                x1, y1 = max(x - mx, 0), max(y - my, 0)
                x2, y2 = min(x + dw + mx, vw - 1), min(y + dh + my, vh - 1)
                cv2.rectangle(vista, (x1, y1), (x2, y2), color, 3)
                cv2.putText(vista, f"{ETIQUETA} {p:.1f}", (x1, max(y1 - 12, 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            n_cuadros = len(lista)
        else:
            n_cuadros = 0

        ahora = time.time()
        fps = 0.9 * fps + 0.1 * (1 / max(ahora - t_anterior, 1e-6))
        t_anterior = ahora
        modo = "  [DIAGNOSTICO]" if diag else ""
        cv2.putText(vista, f"FPS: {fps:.0f}  Detecciones: {n_cuadros}  minNeighbors: "
                           f"{1 if diag else estado['min_neighbors']}{modo}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(vista, f"Modelo {VENTANA[0]}x{VENTANA[1]} | {N_ETAPAS} etapas | {TIPO_RASGOS} | "
                           f"deteccion {estado['ms']:.0f} ms | ecualizar: {'SI' if estado['ecualizar'] else 'NO'}",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if estado["error"]:
            cv2.putText(vista, "ERROR: " + estado["error"][:80], (10, 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(vista, "R: girar | F: pantalla | +/-: sensib. | D: diagnostico | E: ecualizar | ESC: salir",
                    (10, vh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if completa:
            # Lienzo reutilizado (no se crea uno nuevo en cada cuadro)
            if lienzo is None or lienzo.shape[:2] != (SH, SW):
                lienzo = np.zeros((SH, SW, 3), dtype=np.uint8)
            if dims_previas != (vh, vw):
                lienzo[:] = 0
                dims_previas = (vh, vw)
            y0, x0 = (SH - vh) // 2, (SW - vw) // 2
            lienzo[y0:y0 + vh, x0:x0 + vw] = vista
            cv2.imshow("frame", lienzo)
        else:
            cv2.imshow("frame", vista)
    else:
        if not lector.ok:
            print("Se perdió la conexión con la cámara.")
            break
        time.sleep(0.003)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k in (ord('r'), ord('R')):
        giro_actual = (giro_actual + 1) % len(GIROS)
    elif k in (ord('f'), ord('F')):
        completa = not completa
        dims_previas = None
        cv2.setWindowProperty(
            "frame", cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if completa else cv2.WINDOW_NORMAL)
    elif k in (ord('+'), ord('=')):
        estado["min_neighbors"] = min(estado["min_neighbors"] + 1, 100)
    elif k in (ord('-'), ord('_')):
        estado["min_neighbors"] = max(estado["min_neighbors"] - 1, 1)
    elif k in (ord('d'), ord('D')):
        estado["diag"] = not estado["diag"]
        print("Modo diagnóstico:", "ACTIVADO" if estado["diag"] else "desactivado")
    elif k in (ord('e'), ord('E')):
        estado["ecualizar"] = not estado["ecualizar"]
        print("Ecualizar histograma:", "SI" if estado["ecualizar"] else "NO")

estado["salir"] = True
lector.ok = False
time.sleep(0.1)
cap.release()
cv2.destroyAllWindows()

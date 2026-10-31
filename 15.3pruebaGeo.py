import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# --- CARPETAS Y CONFIGURACIÓN ---
CARPETA_PATRONES = "patrones_guardados"
if not os.path.exists(CARPETA_PATRONES):
    os.makedirs(CARPETA_PATRONES)

BG = "#0f172a"
PANEL = "#1e293b"
BORDER = "#334155"
ACCENT = "#3b82f6"
TEXT_MAIN = "#f8fafc"
TEXT_MUTED = "#94a3b8"

BTN_BLUE = "#2563eb"
BTN_GREEN = "#16a34a"
BTN_PURPLE = "#7c3aed"

FONT_TITLE = ("Segoe UI", 10, "bold")
FONT_LABEL = ("Segoe UI", 9)
FONT_CONSOLE = ("Consolas", 9)

STYLE_BTN_BASE = {
    "font": ("Segoe UI", 9, "bold"),
    "relief": "flat",
    "cursor": "hand2",
    "activebackground": "#475569",
    "activeforeground": "#ffffff",
    "bd": 0,
    "highlightthickness": 0,
    "padx": 8,
    "pady": 4
}

# --- PARÁMETROS DE LA IDENTIFICACIÓN (ZONA 4) ---
UMBRAL_ACEPTACION = 80.0     # por debajo de esto el resultado se marca como "baja confianza"
MARGEN_AMBIGUO = 5.0         # si 1° y 2° puestos difieren menos de esto, se marca como ambiguo
UMBRALES_BARRIDO = tuple(range(40, 241, 10))   # umbrales probados si el resultado no es confiable
CORTE_BARRIDO = 90.0         # el barrido se detiene si logra un puntaje >= a esto (y con ventaja)

USAR_MARCADORES = True       # buscar un marcador cuadrado (con dibujo interno) y compararlo
TAM_MARCADOR = 64            # lado (px) al que se rectifica el marcador para compararlo
LADO_MAX_MARCADOR = 1000     # imágenes más grandes se reducen a esto antes de buscar el marcador
MAX_CANDIDATOS = 8           # máximo de cuadrados candidatos por imagen
RAZON_LADOS_MIN = 0.70       # un candidato debe ser aproximadamente cuadrado
RELLENO_RECT_MIN = 0.80      # y llenar bien su rectángulo mínimo

# Comparación por forma (respaldo cuando no hay marcador cuadrado)
TAM_CANVAS = 128             # lado del lienzo normalizado donde se comparan las formas
MARGEN_CANVAS = 8
TAM_TRABAJO = 100
PASO_ROTACION = 5            # grados entre giros de la búsqueda gruesa
RADIO_REFINO = 4             # luego se refina a 1 grado alrededor del mejor ángulo
VENTANA_REFINO = 30.0
LADO_MAX_FORMA = 512         # imágenes más grandes se reducen a esto (acelera y quita ruido fino)
FRAC_CONTRASTE_MIN = 0.003   # si el color minoritario ocupa menos que esto, el recorte está vacío
QUANTIL_RECORTE = 0.005
SIGMA_SUAVIZADO = 1.0
FRAC_AREA_MIN = 0.002
FRAC_AREA_SIGNIF = 0.10
UMBRAL_REINTENTO = 80.0

# --- ESTADO GLOBAL ---
camara_activa = False
capture = None
imagen_base_gray = None
imagen_base_bgr = None

roi_coords_real = None
roi_start_ui = None
roi_end_ui = None
drawing_roi = False
ultima_binaria = None

# --- VENTANA PRINCIPAL ---
ventana = tk.Tk()
ventana.geometry("1480x790")
ventana.title("Identificador de Piezas por Umbralización Robusta")
ventana.configure(bg=BG)
ventana.resizable(False, False)


def mostrar(label, img_np, bgr=False, w=300, h=240):
    if img_np is None or img_np.size == 0:
        return
    if bgr:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_np = cv2.resize(img_np, (w, h), interpolation=cv2.INTER_AREA)
    pil_img = Image.fromarray(img_np)
    tk_img = ImageTk.PhotoImage(image=pil_img)
    label.configure(image=tk_img)
    label.image = tk_img


def escribir(caja, texto):
    caja.configure(state="normal")
    caja.delete("1.0", tk.END)
    caja.insert(tk.END, texto)
    caja.configure(state="disabled")


def archivo():
    global imagen_base_gray, imagen_base_bgr, roi_coords_real, roi_start_ui, roi_end_ui
    path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")])
    if not path:
        return
    img = cv2.imread(path)
    if img is None:
        messagebox.showerror("Error", "No se pudo leer la imagen")
        return

    roi_start_ui = None
    roi_end_ui = None
    roi_coords_real = None

    imagen_base_bgr = img
    imagen_base_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mostrar(LImagen, imagen_base_bgr, bgr=True)
    recortar_imagen()
    umbralizar()


def iniciar_camara():
    if not camara_activa or capture is None:
        return
    ret, frame = capture.read()
    if ret:
        mostrar(LImagen, frame, bgr=True)
        LImagen.after(30, iniciar_camara)


def alternar_camara(event=None):
    global capture, camara_activa
    if not camara_activa:
        capture = cv2.VideoCapture(1)
        if not capture.isOpened():
            messagebox.showerror("Error", "No se pudo acceder a la cámara")
            return
        camara_activa = True
        BCamara.configure(text="Detener (X)", bg="#dc2626")
        iniciar_camara()
    else:
        camara_activa = False
        if capture:
            capture.release()
            capture = None
        BCamara.configure(text="Cámara (X)", bg=BTN_BLUE)
        LImagen.configure(image="")


def tomar_foto(event=None):
    global imagen_base_gray, imagen_base_bgr, roi_coords_real, roi_start_ui, roi_end_ui, camara_activa
    if capture is None or not capture.isOpened():
        if event is None:
            messagebox.showwarning("Aviso", "Primero activa la cámara")
        return
    ret, frame = capture.read()
    if not ret:
        messagebox.showerror("Error", "No se pudo capturar el fotograma")
        return

    camara_activa = False
    BCamara.configure(text="Cámara (X)", bg=BTN_BLUE)

    roi_start_ui = None
    roi_end_ui = None
    roi_coords_real = None

    imagen_base_bgr = frame
    imagen_base_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mostrar(LImagen, imagen_base_bgr, bgr=True)
    recortar_imagen()
    umbralizar()
    # Al capturar una foto con la cámara se detecta la pieza y se muestra el patrón más parecido
    identificar_y_comparar(silencioso=True)


# --- SELECCIÓN MANUAL DE ROI (PANEL 1) ---
def empezar_seleccion(event):
    global roi_start_ui, roi_end_ui, drawing_roi
    if imagen_base_gray is None:
        return
    roi_start_ui = (event.x, event.y)
    roi_end_ui = (event.x, event.y)
    drawing_roi = True


def mover_seleccion(event):
    global roi_end_ui
    if not drawing_roi or imagen_base_gray is None:
        return
    w_label = max(1, LImagen.winfo_width())
    h_label = max(1, LImagen.winfo_height())
    x = max(0, min(event.x, w_label))
    y = max(0, min(event.y, h_label))
    roi_end_ui = (x, y)
    redibujar_seleccion()


def terminar_seleccion(event):
    global drawing_roi
    if not drawing_roi:
        return
    drawing_roi = False
    recortar_imagen()
    umbralizar()
    if imagen_base_bgr is not None:
        vis = imagen_base_bgr.copy()
        if roi_coords_real is not None:
            x1, y1, x2, y2 = roi_coords_real
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        mostrar(LImagen, vis, bgr=True)


def redibujar_seleccion():
    if imagen_base_bgr is None or roi_start_ui is None or roi_end_ui is None:
        return

    h_real, w_real = imagen_base_gray.shape[:2]
    vis = imagen_base_bgr.copy()

    scale_x = w_real / 300.0
    scale_y = h_real / 240.0

    x1 = int(roi_start_ui[0] * scale_x)
    y1 = int(roi_start_ui[1] * scale_y)
    x2 = int(roi_end_ui[0] * scale_x)
    y2 = int(roi_end_ui[1] * scale_y)

    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    mostrar(LImagen, vis, bgr=True, w=300, h=240)


def recortar_imagen(event=None):
    global roi_coords_real
    if imagen_base_gray is None:
        return None

    h_real, w_real = imagen_base_gray.shape[:2]

    if roi_start_ui is None or roi_end_ui is None:
        recortada = imagen_base_gray.copy()
        roi_coords_real = (0, 0, w_real, h_real)
    else:
        scale_x = w_real / 300.0
        scale_y = h_real / 240.0

        x1_ui, x2_ui = sorted([roi_start_ui[0], roi_end_ui[0]])
        y1_ui, y2_ui = sorted([roi_start_ui[1], roi_end_ui[1]])

        x1 = max(0, min(w_real - 1, int(x1_ui * scale_x)))
        x2 = max(0, min(w_real, int(x2_ui * scale_x)))
        y1 = max(0, min(h_real - 1, int(y1_ui * scale_y)))
        y2 = max(0, min(h_real, int(y2_ui * scale_y)))

        if (x2 - x1) < 15 or (y2 - y1) < 15:
            recortada = imagen_base_gray.copy()
            roi_coords_real = (0, 0, w_real, h_real)
        else:
            recortada = imagen_base_gray[y1:y2, x1:x2]
            roi_coords_real = (x1, y1, x2, y2)

    mostrar(LImagen2, recortada, w=300, h=240)
    return recortada


# --- PANEL 3: UMBRALIZACIÓN MANUAL DIRECTA ---
def umbralizar(val=None):
    global ultima_binaria
    if imagen_base_gray is None:
        return None

    recortada = recortar_imagen()
    if recortada is None:
        return None

    blurred = cv2.GaussianBlur(recortada, (3, 3), 0)
    umb_val = int(SGray.get())

    if umb_val == 0:
        _, binaria = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binaria = cv2.threshold(blurred, umb_val, 255, cv2.THRESH_BINARY)

    ultima_binaria = binaria
    mostrar(LImagenUmbral, binaria, w=300, h=240)
    return binaria


def guardar_patron():
    if ultima_binaria is None:
        messagebox.showwarning("Aviso", "No hay imagen umbralizada para guardar")
        return

    nombre = EntradaNombre.get().strip().replace(" ", "_") or "Pieza_1"
    ruta = os.path.join(CARPETA_PATRONES, f"{nombre}.png")

    # Se guarda a resolución NATIVA (sin forzar 120x120) para no deformar la pieza:
    # así se conservan sus proporciones reales (ancho/alto) al compararla después.
    _, patron_norm = cv2.threshold(ultima_binaria, 127, 255, cv2.THRESH_BINARY)

    if cv2.imwrite(ruta, patron_norm):
        messagebox.showinfo("Éxito", f"Patrón guardado correctamente:\n{nombre}.png")
    else:
        messagebox.showerror("Error", "No se pudo guardar el archivo")


# --- PANEL 4: IDENTIFICACIÓN Y MOSTRAR IMAGEN GANADORA UMBRALIZADA ---
# Cómo identifica (resumen):
#   1) Busca en la imagen umbralizada el MARCADOR: un cuadrado con dibujo interno. Funciona con el cuadrado
#      oscuro (umbrales bajos) o con el marco blanco de papel (umbrales altos), en cualquier giro,
#      tamaño o inclinación, y aunque la imagen tenga ruido o toda la escena (no hace falta recortar).
#   2) Endereza ese cuadrado (corrige la perspectiva) y lo compara con el marcador de cada patrón
#      guardado, probando sus 4 giros de 90°. Puntaje = correlación (0-100%).
#   3) Si no hay marcador cuadrado, compara la FORMA de la pieza (recorte justo, escala y giro libres).
#   4) Si el resultado no es confiable con el umbral del deslizador, prueba varios umbrales
#      automáticamente y se queda con el mejor (y deja el deslizador en ese valor).

# ---------------------------------------------------------------------------
# UTILIDADES COMUNES
# ---------------------------------------------------------------------------
def _hay_contraste(binaria):
    """False si el recorte es casi de un solo color (no hay nada que comparar)."""
    p = float(np.mean(binaria > 127))
    return FRAC_CONTRASTE_MIN < p < 1.0 - FRAC_CONTRASTE_MIN


def _a_binaria(img):
    return np.where(img > 127, 255, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 1) MARCADOR CUADRADO (ruta principal para tus piezas)
# ---------------------------------------------------------------------------
def _vector_normalizado(suave):
    v = suave.astype(np.float32).ravel()
    v = v - v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.zeros_like(v)


def _candidatos_marcador(binaria):
    """Busca cuadrados con dibujo interno (el marcador) sin importar umbral, giro ni polaridad.

    Se revisan las dos polaridades: a umbral bajo el marcador es un cuadrado oscuro sobre fondo
    claro; a umbral alto es el marco blanco de papel con el dibujo adentro. En ambos casos el
    candidato se rectifica (se endereza la perspectiva) a TAM_MARCADOR x TAM_MARCADOR.
    Devuelve una lista de diccionarios (los más grandes primero).
    """
    if not USAR_MARCADORES:
        return []
    b = _a_binaria(binaria)
    if not _hay_contraste(b):
        return []

    esc = min(1.0, LADO_MAX_MARCADOR / float(max(b.shape)))
    if esc < 1.0:
        b = _a_binaria(cv2.resize(b, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA))
    h, w = b.shape
    pad = max(8, int(0.05 * max(h, w)))
    min_area = max(100.0, 0.003 * h * w)
    S = TAM_MARCADOR
    esquinas_img = np.array([[pad, pad], [pad + w - 1, pad],
                             [pad + w - 1, pad + h - 1], [pad, pad + h - 1]], np.float32)
    tol = 3 + 0.01 * max(h, w)
    dst = np.array([[0, 0], [2 * S - 1, 0], [2 * S - 1, 2 * S - 1], [0, 2 * S - 1]], np.float32)

    cands = []
    for invertir in (False, True):
        img = cv2.bitwise_not(b) if invertir else b
        p = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        contornos, _ = cv2.findContours(p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contornos:
            # Se trabaja con la envolvente convexa: así un marco roto (papel levantado, esquina
            # faltante) o con una pestaña sigue contando como un cuadrado.
            hull = cv2.convexHull(c)
            area = cv2.contourArea(hull)
            if area < min_area or area > 0.9 * h * w:
                continue
            peri = cv2.arcLength(hull, True)
            approx = None
            for eps in (0.02, 0.035, 0.05, 0.07):
                a4 = cv2.approxPolyDP(hull, eps * peri, True)
                if len(a4) == 4:
                    approx = a4
                    break
            if approx is None or not cv2.isContourConvex(approx):
                continue
            (_, _), (rw, rh), _ = cv2.minAreaRect(hull)
            if min(rw, rh) < 4 or min(rw, rh) / max(rw, rh) < RAZON_LADOS_MIN:
                continue
            if area / (rw * rh) < RELLENO_RECT_MIN:
                continue
            quad = approx.reshape(4, 2).astype(np.float32)
            # descartar el "marco" de la imagen completa (esquinas pegadas a las esquinas de la imagen)
            cerca = sum(1 for q in quad if float(np.min(np.linalg.norm(esquinas_img - q, axis=1))) <= tol)
            if cerca >= 3:
                continue
            # orientación consistente (sin espejo): el orden del contorno puede ir en cualquier sentido
            x, y = quad[:, 0], quad[:, 1]
            if float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) < 0:
                quad = quad[::-1].copy()
            M = cv2.getPerspectiveTransform(quad, dst)
            warp = cv2.warpPerspective(p, M, (2 * S, 2 * S), flags=cv2.INTER_LINEAR)
            canon = cv2.resize(warp, (S, S), interpolation=cv2.INTER_AREA)
            # debe tener dibujo interno (ambos colores); un cuadrado liso no es un marcador
            a, z = int(0.15 * S), int(0.85 * S)
            frac = float(np.mean(canon[a:z, a:z] > 127))
            if not (0.04 <= frac <= 0.96):
                continue
            suave = cv2.GaussianBlur(canon.astype(np.float32) / 255.0, (0, 0), 1.0)
            dx, dy = float(quad[1, 0] - quad[0, 0]), float(quad[1, 1] - quad[0, 1])
            quad_orig = (quad - pad) / esc
            cands.append({
                "area": float(area) / (esc * esc),
                "alfa": float(np.degrees(np.arctan2(-dy, dx))),     # giro (antihorario) del lado 0->1
                "centro": (float(quad_orig[:, 0].mean()), float(quad_orig[:, 1].mean())),
                "lado": float(np.sqrt(area)) / esc,
                "vec": _vector_normalizado(suave),
                "vec_rots": np.stack([_vector_normalizado(np.rot90(suave, r)) for r in range(4)], axis=0),
            })
    cands.sort(key=lambda d: d["area"], reverse=True)
    return cands[:MAX_CANDIDATOS]


def _puntaje_marcadores(q_cands, p_cands):
    """Mejor correlación (0-100) entre algún candidato de la consulta y alguno del patrón (4 giros de 90°).

    Devuelve (score, giro_pieza_antihorario, centro_consulta, lado_consulta)."""
    Q = np.stack([c["vec"] for c in q_cands], axis=0)                       # (nq, S*S)
    P = np.concatenate([c["vec_rots"] for c in p_cands], axis=0)            # (np*4, S*S)
    sims = Q @ P.T
    k = int(np.argmax(sims))
    qi, j = divmod(k, P.shape[0])
    pj, r = divmod(j, 4)
    score = max(0.0, float(sims[qi, j])) * 100.0
    q, p = q_cands[qi], p_cands[pj]
    giro = (q["alfa"] - p["alfa"] + 90.0 * r) % 360.0     # giro antihorario de la pieza respecto al patrón
    return score, giro, q["centro"], q["lado"]


# ---------------------------------------------------------------------------
# 2) COMPARACIÓN POR FORMA (respaldo: piezas sin marcador cuadrado)
# ---------------------------------------------------------------------------
def _reducir(b):
    m = max(b.shape)
    if m <= LADO_MAX_FORMA:
        return b
    esc = LADO_MAX_FORMA / float(m)
    return _a_binaria(cv2.resize(b, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA))


def _forma_en_blanco(binaria, invertir=False):
    """Máscara 0/255 con la PIEZA en blanco; el fondo es el color dominante del borde del recorte.
    Si invertir=True se interpreta justo al revés (reintento de seguridad)."""
    b = _a_binaria(binaria)
    borde = np.concatenate([b[0, :], b[-1, :], b[:, 0], b[:, -1]])
    borde_es_blanco = np.mean(borde > 127) > 0.5
    if borde_es_blanco != invertir:
        b = cv2.bitwise_not(b)
    return b


def _quitar_ruido(mascara):
    """Elimina motas pequeñas (blancas) y rellena agujeros diminutos (negros)."""
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    if n <= 1:
        return None
    areas = stats[:, cv2.CC_STAT_AREA]
    mayor = int(areas[1:].max())
    min_area = max(4, int(mayor * FRAC_AREA_MIN))
    conservar = areas >= min_area
    conservar[0] = False
    limpia = conservar[etiquetas].astype(np.uint8) * 255

    inv = cv2.bitwise_not(limpia)
    n2, et2, st2, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    if n2 > 1:
        pequeno = st2[:, cv2.CC_STAT_AREA] < min_area
        pequeno[0] = False
        limpia[pequeno[et2]] = 255
    return limpia


def _recorte_pieza(mascara):
    """Recorta al rectángulo que contiene la pieza (ignora motas lejanas)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    mayor = int(areas.max())
    sig = np.where(areas >= mayor * FRAC_AREA_SIGNIF)[0] + 1
    x1 = int(stats[sig, cv2.CC_STAT_LEFT].min())
    y1 = int(stats[sig, cv2.CC_STAT_TOP].min())
    x2 = int((stats[sig, cv2.CC_STAT_LEFT] + stats[sig, cv2.CC_STAT_WIDTH]).max())
    y2 = int((stats[sig, cv2.CC_STAT_TOP] + stats[sig, cv2.CC_STAT_HEIGHT]).max())
    if x2 <= x1 or y2 <= y1:
        return None
    return mascara[y1:y2, x1:x2]


def _recortar_robusto(mascara):
    """Recorte al rectángulo de la pieza ignorando una pequeña fracción de masa en cada borde."""
    m = mascara > 0
    total = int(m.sum())
    if total == 0:
        return None
    cols = np.cumsum(m.sum(axis=0))
    filas = np.cumsum(m.sum(axis=1))
    lim = QUANTIL_RECORTE * total
    x1 = int(np.searchsorted(cols, lim))
    x2 = int(np.searchsorted(cols, total - lim)) + 1
    y1 = int(np.searchsorted(filas, lim))
    y2 = int(np.searchsorted(filas, total - lim)) + 1
    x2, y2 = min(x2, m.shape[1]), min(y2, m.shape[0])
    if x2 - x1 < 2 or y2 - y1 < 2:
        pts = cv2.findNonZero(mascara)
        x, y, w, h = cv2.boundingRect(pts)
        return mascara[y:y + h, x:x + w]
    return mascara[y1:y2, x1:x2]


def _al_lienzo(recorte):
    """Escala el recorte SIN deformarlo y lo centra en un lienzo fijo."""
    h, w = recorte.shape[:2]
    lado = TAM_CANVAS - 2 * MARGEN_CANVAS
    esc = lado / float(max(h, w))
    nw = max(1, int(round(w * esc)))
    nh = max(1, int(round(h * esc)))
    r = cv2.resize(recorte, (nw, nh), interpolation=cv2.INTER_AREA if esc < 1 else cv2.INTER_LINEAR)
    r = _a_binaria(r)
    lienzo = np.zeros((TAM_CANVAS, TAM_CANVAS), dtype=np.uint8)
    y0 = (TAM_CANVAS - nh) // 2
    x0 = (TAM_CANVAS - nw) // 2
    lienzo[y0:y0 + nh, x0:x0 + nw] = r
    return lienzo


def _suavizar(canvas):
    return cv2.GaussianBlur(canvas.astype(np.float32) / 255.0, (0, 0), SIGMA_SUAVIZADO)


def forma_canonica(binaria, invertir=False):
    """Imagen binaria -> lienzo normalizado (pieza blanca, centrada, sin deformar)."""
    if not _hay_contraste(binaria):
        return None
    mascara = _quitar_ruido(_forma_en_blanco(_reducir(_a_binaria(binaria)), invertir))
    if mascara is None:
        return None
    recorte = _recorte_pieza(mascara)
    if recorte is None:
        return None
    recorte = _recortar_robusto(recorte)
    if recorte is None:
        return None
    return _al_lienzo(recorte)


class _Consulta:
    """Forma de la consulta lista para generar su versión girada a cualquier ángulo (con caché)."""

    def __init__(self, binaria, invertir=False):
        self.valida = False
        self._cache = {}
        if not _hay_contraste(binaria):
            return
        mascara = _quitar_ruido(_forma_en_blanco(_reducir(_a_binaria(binaria)), invertir))
        if mascara is None:
            return
        recorte = _recorte_pieza(mascara)
        if recorte is None:
            return
        h, w = recorte.shape[:2]
        esc = TAM_TRABAJO / float(max(h, w))
        nw = max(1, int(round(w * esc)))
        nh = max(1, int(round(h * esc)))
        pequeno = cv2.resize(recorte, (nw, nh), interpolation=cv2.INTER_AREA if esc < 1 else cv2.INTER_LINEAR)
        self.D = int(np.ceil(TAM_TRABAJO * 1.4142)) + 6
        self.base = np.zeros((self.D, self.D), dtype=np.uint8)
        y0 = (self.D - nh) // 2
        x0 = (self.D - nw) // 2
        self.base[y0:y0 + nh, x0:x0 + nw] = pequeno
        self.valida = True

    def suave(self, ang):
        ang = ang % 360
        if ang in self._cache:
            return self._cache[ang]
        centro = (self.D / 2.0, self.D / 2.0)
        M = cv2.getRotationMatrix2D(centro, ang, 1.0)
        rot = cv2.warpAffine(self.base, M, (self.D, self.D), flags=cv2.INTER_LINEAR)
        rot = _a_binaria(rot)
        recorte = _recortar_robusto(rot)
        if recorte is None:
            self._cache[ang] = None
            return None
        res = _suavizar(_al_lienzo(recorte))
        self._cache[ang] = res
        return res


def _iou_suave(stack, patron_suave):
    num = np.minimum(stack, patron_suave).sum(axis=(1, 2))
    den = np.maximum(stack, patron_suave).sum(axis=(1, 2))
    return num / np.maximum(den, 1e-6)


def _mejor_angulo(consulta, patron_suave, angulos):
    imgs, angs = [], []
    for a in angulos:
        im = consulta.suave(a)
        if im is not None:
            imgs.append(im)
            angs.append(a % 360)
    if not imgs:
        return 0.0, 0
    iou = _iou_suave(np.stack(imgs, axis=0), patron_suave)
    i = int(np.argmax(iou))
    return float(iou[i]) * 100.0, angs[i]


def _puntajes_forma(binaria, patrones):
    """Puntaje por forma (0-100) y giro (antihorario) de la pieza para cada patrón. dict nombre->(score, giro)."""

    def evaluar(invertir):
        cons = _Consulta(binaria, invertir)
        if not cons.valida:
            return None
        gruesos = list(range(0, 360, PASO_ROTACION))
        res = {}
        for p in patrones:
            ps = p.forma_suave
            res[p.nombre] = (0.0, 0) if ps is None else _mejor_angulo(cons, ps, gruesos)
        mejor_grueso = max(v[0] for v in res.values())
        if PASO_ROTACION > 1 and RADIO_REFINO > 0:
            for p in patrones:
                sc, ang = res[p.nombre]
                if p.forma_suave is None or sc < mejor_grueso - VENTANA_REFINO:
                    continue
                finos = [ang + d for d in range(-RADIO_REFINO, RADIO_REFINO + 1)]
                sc2, ang2 = _mejor_angulo(cons, p.forma_suave, finos)
                if sc2 > sc:
                    res[p.nombre] = (sc2, ang2)
        return res

    res = evaluar(False)
    mejor = max(v[0] for v in res.values()) if res else -1.0
    if res is None or mejor < UMBRAL_REINTENTO:
        res2 = evaluar(True)
        if res2 is not None and (res is None or max(v[0] for v in res2.values()) > mejor):
            res = res2
    if res is None:
        return {}
    return {n: (s, (-a) % 360) for n, (s, a) in res.items()}


# ---------------------------------------------------------------------------
# 3) API: PATRONES PREPARADOS, COMPARACIÓN Y BARRIDO DE UMBRALES
# ---------------------------------------------------------------------------
class Patron:
    """Patrón guardado, con sus candidatos de marcador y su forma calculados una sola vez."""

    def __init__(self, nombre, img_gris):
        self.nombre = nombre
        self.img = img_gris
        self.marcadores = _candidatos_marcador(img_gris)
        self._forma = False          # se calcula solo si hace falta

    @property
    def forma_suave(self):
        if self._forma is False:
            can = forma_canonica(self.img)
            self._forma = None if can is None else _suavizar(can)
        return self._forma


def preparar_patrones(lista):
    """lista: [(nombre, imagen_gris)] -> [Patron]"""
    return [Patron(n, img) for n, img in lista]


def comparar_con_patrones(binaria, patrones):
    """Compara la binaria de la consulta con los patrones.

    Devuelve (resultados, info):
      resultados = [(nombre, puntaje_0_100, giro_antihorario_en_grados)] de mayor a menor similitud
      info       = {"metodo": "marcador"|"forma", "centro": (x, y) o None, "lado": px o None}
    """
    q_marc = _candidatos_marcador(binaria)
    datos = {}
    pendientes = []
    for p in patrones:
        if q_marc and p.marcadores:
            sc, giro, centro, lado = _puntaje_marcadores(q_marc, p.marcadores)
            datos[p.nombre] = (sc, giro, "marcador", centro, lado)
        else:
            pendientes.append(p)
    if pendientes:
        if _hay_contraste(binaria):
            for n, (sc, giro) in _puntajes_forma(binaria, pendientes).items():
                datos[n] = (sc, giro, "forma", None, None)
        for p in pendientes:
            datos.setdefault(p.nombre, (0.0, 0.0, "forma", None, None))
    if not datos or (not q_marc and not _hay_contraste(binaria)):
        return [], {"metodo": None, "centro": None, "lado": None}
    ordenados = sorted(datos.items(), key=lambda kv: kv[1][0], reverse=True)
    resultados = [(n, v[0], int(round(v[1])) % 360) for n, v in ordenados]
    top = ordenados[0][1]
    return resultados, {"metodo": top[2], "centro": top[3], "lado": top[4]}


def comparar_con_barrido(gris, patrones, umbrales=UMBRALES_BARRIDO):
    """Prueba varios umbrales sobre la imagen en gris y devuelve el mejor intento.

    Devuelve (umbral, resultados, info) o None si ningún umbral dio resultado.
    """
    borroso = cv2.GaussianBlur(gris, (3, 3), 0)
    mejor = None
    for t in umbrales:
        _, b = cv2.threshold(borroso, t, 255, cv2.THRESH_BINARY)
        resultados, info = comparar_con_patrones(b, patrones)
        if not resultados:
            continue
        sc = resultados[0][1]
        if mejor is None or sc > mejor[1][0][1]:
            mejor = (t, resultados, info)
        ventaja = sc - resultados[1][1] if len(resultados) > 1 else 100.0
        if sc >= CORTE_BARRIDO and ventaja >= 10.0:
            break
    return mejor


_CACHE_PATRONES = {}   # (ruta, fecha de modificación) -> Patron ya preparado (evita recalcular en cada clic)


def _cargar_patrones():
    """Lee la carpeta de patrones. Devuelve (lista_de_Patron, dict nombre->ruta)."""
    patrones, rutas, claves = [], {}, set()
    for archivo_patron in sorted(os.listdir(CARPETA_PATRONES)):
        if not archivo_patron.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        ruta = os.path.join(CARPETA_PATRONES, archivo_patron)
        clave = (ruta, os.path.getmtime(ruta))
        if clave not in _CACHE_PATRONES:
            img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            _CACHE_PATRONES[clave] = Patron(os.path.splitext(archivo_patron)[0], img)
        claves.add(clave)
        patrones.append(_CACHE_PATRONES[clave])
        rutas[patrones[-1].nombre] = ruta
    for k in list(_CACHE_PATRONES):
        if k not in claves:
            del _CACHE_PATRONES[k]
    return patrones, rutas


def identificar_y_comparar(silencioso=False):
    """silencioso=True: no muestra ventanas de aviso (se usa al capturar una foto con la cámara)."""
    if imagen_base_bgr is None:
        if not silencioso:
            messagebox.showwarning("Aviso", "Primero carga o captura una imagen")
        return

    umbralizar()
    if ultima_binaria is None:
        if not silencioso:
            messagebox.showwarning("Aviso", "No se ha generado el recorte umbralizado")
        return

    patrones, rutas = _cargar_patrones()
    if not patrones:
        if not silencioso:
            messagebox.showwarning("Aviso", "No hay patrones en 'patrones_guardados'. Guarda al menos un patrón primero.")
        return

    resultados, info = comparar_con_patrones(ultima_binaria, patrones)

    # Si con el umbral actual no hay una coincidencia confiable, se prueban varios umbrales
    umbral_usado = None
    barrido_intentado = False
    mejor_actual = resultados[0][1] if resultados else 0.0
    if mejor_actual < UMBRAL_ACEPTACION:
        gris_recorte = recortar_imagen()
        if gris_recorte is not None:
            barrido_intentado = True
            barrido = comparar_con_barrido(gris_recorte, patrones)
            if barrido is not None and barrido[1][0][1] > mejor_actual:
                umbral_usado, resultados, info = barrido
                if resultados[0][1] >= UMBRAL_ACEPTACION:
                    SGray.set(umbral_usado)      # la zona 3 pasa a mostrar el umbral que funcionó
                else:
                    umbral_usado = None          # no fue mejor en la práctica: se informa como fallido

    x1, y1, x2, y2 = roi_coords_real if roi_coords_real else (0, 0, 0, 0)
    txt_pos = "INFORMACIÓN DE DETECCIÓN:\n"

    # Recorte sin pieza (todo de un solo color): no hay nada que comparar
    if not resultados:
        mostrar(LImagenResultado, np.zeros((240, 300), dtype=np.uint8))
        escribir(CajaTemplate, "No se detectó ninguna pieza en el recorte (imagen sin contraste).\n"
                               "Ajusta el umbral o la zona seleccionada e inténtalo de nuevo.\n")
        txt_pos += "Pieza Identificada: ninguna\n"
        txt_pos += f"Zona de Recorte: ({x1}, {y1}) a ({x2}, {y2})\n"
        txt_pos += f"Dimensiones ROI: {x2 - x1} x {y2 - y1} px\n"
        escribir(CajaShapes, txt_pos)
        return

    pieza_detectada, mejor_score, mejor_giro = resultados[0]
    segundo_score = resultados[1][1] if len(resultados) > 1 else None

    # Nivel de confianza del resultado
    if mejor_score < UMBRAL_ACEPTACION:
        confianza = f"BAJA (menor a {UMBRAL_ACEPTACION:.0f}%): ningún patrón coincide bien"
        marca_ganador = "  <-- MÁS PARECIDA (baja confianza)"
    elif segundo_score is not None and (mejor_score - segundo_score) < MARGEN_AMBIGUO:
        confianza = f"AMBIGUA (diferencia con la 2da: {mejor_score - segundo_score:.1f} pts)"
        marca_ganador = "  <-- DETECTADA (ambigua)"
    else:
        confianza = "ALTA"
        marca_ganador = "  <-- DETECTADA"

    # --- PANEL 4: MOSTRAR LA IMAGEN UMBRALIZADA DEL PATRÓN GANADOR ---
    ruta_ganador = rutas.get(pieza_detectada, "")
    if ruta_ganador and os.path.exists(ruta_ganador):
        img_ganadora = cv2.imread(ruta_ganador, cv2.IMREAD_GRAYSCALE)
        mostrar(LImagenResultado, img_ganadora, w=300, h=240)
    else:
        mostrar(LImagenResultado, np.zeros((240, 300), dtype=np.uint8))

    # Consolas inferiores
    txt_sim = f"{'Pieza Guardada':<20} {'Similitud':>10} {'Giro':>7}\n" + "─" * 40 + "\n"
    for nom, sim, giro in resultados:
        marca = marca_ganador if nom == pieza_detectada else ""
        txt_sim += f"{nom:<20} {sim:>8.1f}% {giro:>5}°{marca}\n"
    escribir(CajaTemplate, txt_sim)

    txt_pos += f"Pieza Identificada: {pieza_detectada}\n"
    txt_pos += f"Similitud Máxima: {mejor_score:.2f}%\n"
    txt_pos += f"Confianza: {confianza}\n"
    txt_pos += f"Giro respecto al patrón: {mejor_giro}° (antihorario)\n"
    if info.get("metodo") == "marcador" and info.get("centro"):
        cx, cy = info["centro"]
        txt_pos += f"Marcador detectado en: ({x1 + cx:.0f}, {y1 + cy:.0f}) px  (lado ≈ {info['lado']:.0f} px)\n"
        txt_pos += "Método: marcador cuadrado\n"
    elif info.get("metodo") == "forma":
        txt_pos += "Método: comparación por forma (no se halló un marcador cuadrado)\n"
    if umbral_usado is not None:
        txt_pos += f"Umbral usado: {umbral_usado} (barrido automático)\n"
    elif barrido_intentado:
        txt_pos += "Se probaron varios umbrales automáticamente; ninguno dio una coincidencia confiable.\n"
    txt_pos += f"Zona de Recorte: ({x1}, {y1}) a ({x2}, {y2})\n"
    txt_pos += f"Dimensiones ROI: {x2 - x1} x {y2 - y1} px\n"
    escribir(CajaShapes, txt_pos)


def crear_scale(parent, label_text, from_val, to_val, command=None, orient="horizontal"):
    return tk.Scale(
        parent, from_=from_val, to=to_val, orient=orient, label=label_text,
        bg=PANEL, fg=TEXT_MAIN, activebackground=ACCENT, troughcolor="#0f172a",
        highlightthickness=0, bd=0, font=FONT_LABEL, command=command,
    )

# --- ESTRUCTURA DE LA INTERFAZ ---
top_frame = tk.Frame(ventana, bg=BG)
top_frame.pack(fill="x", padx=12, pady=10)

# PANEL 1: Captura
F0 = tk.LabelFrame(top_frame, text=" 1. Captura y Selección ", bg=PANEL, fg=TEXT_MUTED, font=FONT_TITLE, bd=1, relief="solid")
F0.pack(side="left", expand=True, fill="both", padx=4)

lbl_info = tk.Label(F0, text="Arrastra con el ratón sobre el marcador o pieza", bg=PANEL, fg=TEXT_MUTED, font=FONT_LABEL)
lbl_info.pack(pady=(6, 2))

FContenedor = tk.Frame(F0, width=300, height=240, bg="#0f172a", bd=1, relief="solid")
FContenedor.pack_propagate(False)
FContenedor.pack(padx=6, pady=4)

LImagen = tk.Label(FContenedor, bg="#0f172a", cursor="crosshair")
LImagen.pack(expand=True, fill="both")

LImagen.bind("<ButtonPress-1>", empezar_seleccion)
LImagen.bind("<B1-Motion>", mover_seleccion)
LImagen.bind("<ButtonRelease-1>", terminar_seleccion)

FBotones = tk.Frame(F0, bg=PANEL)
FBotones.pack(pady=10)

BArchivo = tk.Button(FBotones, text="Archivo", command=archivo, bg="#475569", fg=TEXT_MAIN, **STYLE_BTN_BASE)
BArchivo.pack(side="left", padx=3)

BCamara = tk.Button(FBotones, text="Cámara (X)", command=alternar_camara, bg=BTN_BLUE, fg=TEXT_MAIN, **STYLE_BTN_BASE)
BCamara.pack(side="left", padx=3)

BCapturar = tk.Button(FBotones, text="Foto (Z)", command=tomar_foto, bg="#059669", fg=TEXT_MAIN, **STYLE_BTN_BASE)
BCapturar.pack(side="left", padx=3)

# PANEL 2: Área Recortada
F1 = tk.LabelFrame(top_frame, text=" 2. Área Seleccionada ", bg=PANEL, fg=TEXT_MUTED, font=FONT_TITLE, bd=1, relief="solid")
F1.pack(side="left", expand=True, fill="both", padx=4)

FContenedor2 = tk.Frame(F1, width=300, height=240, bg="#0f172a", bd=1, relief="solid")
FContenedor2.pack_propagate(False)
FContenedor2.pack(padx=8, pady=6)

LImagen2 = tk.Label(FContenedor2, bg="#0f172a")
LImagen2.pack(expand=True, fill="both")

SGray = crear_scale(F1, "Umbralización Manual (0 = Otsu Auto)", 0, 255, command=umbralizar)
SGray.pack(fill="x", padx=8, pady=12)

# PANEL 3: Imagen Recortada Umbralizada
F2 = tk.LabelFrame(top_frame, text=" 3. Recorte Umbralizado ", bg=PANEL, fg=TEXT_MUTED, font=FONT_TITLE, bd=1, relief="solid")
F2.pack(side="left", expand=True, fill="both", padx=4)

FContenedor3 = tk.Frame(F2, width=300, height=240, bg="#0f172a", bd=1, relief="solid")
FContenedor3.pack_propagate(False)
FContenedor3.pack(padx=8, pady=6)

LImagenUmbral = tk.Label(FContenedor3, bg="#0f172a")
LImagenUmbral.pack(expand=True, fill="both")

lbl_nombre = tk.Label(F2, text="Nombre del Patrón (ej: Pieza_1):", bg=PANEL, fg=TEXT_MUTED, font=FONT_LABEL)
lbl_nombre.pack(anchor="w", padx=8, pady=(4, 0))

EntradaNombre = tk.Entry(F2, bg="#0f172a", fg=TEXT_MAIN, font=FONT_CONSOLE, insertbackground=TEXT_MAIN, bd=1, relief="solid", highlightthickness=0)
EntradaNombre.insert(0, "Pieza_1")
EntradaNombre.pack(fill="x", padx=8, pady=4)

BGuardar = tk.Button(F2, text="Guardar Patrón Umbralizado", command=guardar_patron, bg=BTN_GREEN, fg=TEXT_MAIN, **STYLE_BTN_BASE)
BGuardar.pack(fill="x", padx=8, pady=8)

# PANEL 4: Patrón Ganador Umbralizado
F3 = tk.LabelFrame(top_frame, text=" 4. Patrón Más Similar Umbralizado ", bg=PANEL, fg=TEXT_MUTED, font=FONT_TITLE, bd=1, relief="solid")
F3.pack(side="left", expand=True, fill="both", padx=4)

FContenedor4 = tk.Frame(F3, width=300, height=240, bg="#0f172a", bd=1, relief="solid")
FContenedor4.pack_propagate(False)
FContenedor4.pack(padx=8, pady=6)

LImagenResultado = tk.Label(FContenedor4, bg="#0f172a")
LImagenResultado.pack(expand=True, fill="both")

BIdentificar = tk.Button(F3, text="Identificar Pieza", command=identificar_y_comparar, bg=BTN_PURPLE, fg=TEXT_MAIN, **STYLE_BTN_BASE)
BIdentificar.pack(fill="x", padx=8, pady=14)

# CONSOLAS DE TEXTO INFERIORES
bottom_frame = tk.Frame(ventana, bg=BG)
bottom_frame.pack(fill="both", expand=True, padx=12, pady=(0, 10))

bottom_frame.columnconfigure(0, weight=2)
bottom_frame.columnconfigure(1, weight=1)
bottom_frame.rowconfigure(0, weight=1)


def crear_caja_resultado(parent, col, titulo, color_header):
    frame_box = tk.Frame(parent, bg=PANEL, bd=1, relief="solid")
    frame_box.grid(row=0, column=col, sticky="nsew", padx=4, pady=4)

    lbl = tk.Label(frame_box, text=titulo, bg=color_header, fg=TEXT_MAIN, font=FONT_TITLE, pady=3)
    lbl.pack(fill="x")

    text_area = tk.Text(frame_box, bg="#0f172a", fg="#38bdf8", font=FONT_CONSOLE, state="disabled", relief="flat", bd=0, highlightthickness=0, padx=8, pady=8)
    text_area.pack(fill="both", expand=True)
    return text_area


CajaTemplate = crear_caja_resultado(bottom_frame, 0, "Similitud vs Patrones Guardados (%)", "#b45309")
CajaShapes = crear_caja_resultado(bottom_frame, 1, "Datos de la Zona Detectada", "#047857")

# ATAJOS DE TECLADO
ventana.bind("<x>", alternar_camara)
ventana.bind("<X>", alternar_camara)
ventana.bind("<z>", tomar_foto)
ventana.bind("<Z>", tomar_foto)
ventana.bind("<r>", recortar_imagen)
ventana.bind("<R>", recortar_imagen)


def al_cerrar():
    global camara_activa, capture
    camara_activa = False
    if capture:
        capture.release()
    ventana.destroy()


ventana.protocol("WM_DELETE_WINDOW", al_cerrar)
ventana.mainloop()

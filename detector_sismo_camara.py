import csv
import math
import os
import time
from collections import deque
from functools import lru_cache

import cv2
import numpy as np

VERDE = (153, 211, 52)
AMARILLO = (36, 191, 251)
NARANJA = (60, 146, 251)
ROJO = (68, 68, 239)
GRIS = (175, 163, 156)
BLANCO = (245, 245, 245)
CIAN = (240, 210, 60)
PANEL = (30, 20, 14)

FUENTE = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA
COLORES_SEGMENTOS = (VERDE, AMARILLO, NARANJA, ROJO)

CARPETA_GRABACIONES = "grabaciones_sismos"
ARCHIVO_REGISTRO_CSV = os.path.join(CARPETA_GRABACIONES, "registro_sismos.csv")

pos_total_x = 0.0
pos_total_y = 0.0


def mezclar(a, b, k):
    return tuple(int(a[i] * (1 - k) + b[i] * k) for i in range(3))


def oscurecer(c, k):
    return tuple(int(v * k) for v in c)


@lru_cache(maxsize=32)
def _mascara(h, w, r):
    r = max(1, min(r, h // 2, w // 2))
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(m, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
        cv2.circle(m, (cx, cy), r, 255, -1, AA)
    return (m.astype(np.float32) / 255.0)[:, :, None]


@lru_cache(maxsize=32)
def _borde(h, w, r):
    externo = _mascara(h, w, r)
    interno = np.zeros_like(externo)
    if h > 4 and w > 4:
        interno[1:-1, 1:-1] = _mascara(h - 2, w - 2, max(1, r - 1))
    return np.clip(externo - interno, 0, 1)


def panel_vidrio(img, x, y, w, h, r, tinte=PANEL, alpha=0.60, borde=0.16):
    H, W = img.shape[:2]
    x, y = max(0, x), max(0, y)
    w, h = min(w, W - x), min(h, H - y)
    if w < 8 or h < 8:
        return
    roi = img[y:y + h, x:x + w]
    base = cv2.GaussianBlur(roi, (0, 0), 9).astype(np.float32)
    base = base * (1 - alpha) + np.array(tinte, np.float32) * alpha
    m = _mascara(h, w, r)
    out = roi.astype(np.float32) * (1 - m) + base * m
    if borde:
        out += _borde(h, w, r) * 255.0 * borde
    roi[:] = np.clip(out, 0, 255).astype(np.uint8)


def ancho_texto(s, escala, grosor=1, espaciado=0):
    if espaciado == 0:
        return cv2.getTextSize(s, FUENTE, escala, grosor)[0][0]
    return sum(cv2.getTextSize(c, FUENTE, escala, grosor)[0][0] for c in s) + espaciado * (len(s) - 1)


def texto(img, s, org, escala, color, grosor=1, espaciado=0, ancla="izq", sombra=False):
    x, y = org
    tw = ancho_texto(s, escala, grosor, espaciado)
    if ancla == "der":
        x -= tw
    elif ancla == "centro":
        x -= tw // 2

    def dibujar(dx_t, dy_t, col, gr):
        if espaciado == 0:
            cv2.putText(img, s, (x + dx_t, y + dy_t), FUENTE, escala, col, gr, AA)
        else:
            cx = x + dx_t
            for c in s:
                cv2.putText(img, c, (cx, y + dy_t), FUENTE, escala, col, gr, AA)
                cx += cv2.getTextSize(c, FUENTE, escala, grosor)[0][0] + espaciado

    if sombra:
        dibujar(1, 2, (10, 10, 10), grosor + 1)
    dibujar(0, 0, color, grosor)
    return tw


def barra(img, x, y, w, h, color):
    r = h // 2
    if w <= h:
        cv2.circle(img, (x + r, y + r), r, color, -1, AA)
        return
    cv2.rectangle(img, (x + r, y), (x + w - r - 1, y + h - 1), color, -1)
    cv2.circle(img, (x + r, y + r), r, color, -1, AA)
    cv2.circle(img, (x + w - r - 1, y + r), r, color, -1, AA)


def pos_gauge(v):
    if v <= 0.0:
        return 0.0
    if v < 2.8:
        return v / 2.8
    if v < 8.0:
        return 1.0 + (v - 2.8) / 5.2
    if v < 25.0:
        return 2.0 + (v - 8.0) / 17.0
    return min(4.0, 3.0 + (v - 25.0) / 15.0)


@lru_cache(maxsize=4)
def _vineta(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    a = np.clip((d - 0.50) / 0.90, 0, 1) ** 1.5
    return (a * 255).astype(np.uint8)


def vineta_alerta(frame, color, fuerza):
    h, w = frame.shape[:2]
    a = cv2.convertScaleAbs(_vineta(h, w), alpha=fuerza)
    inv = cv2.cvtColor(255 - a, cv2.COLOR_GRAY2BGR)
    base = cv2.multiply(frame, inv, scale=1 / 255.0)
    tinte = cv2.merge([cv2.convertScaleAbs(a, alpha=c / 255.0) for c in color])
    frame[:] = cv2.add(base, tinte)


def dibujar_hud(
    frame, pos_x, pos_y, estado, nivel, color, gauge_t, t,
    modo_manual=False, cant_puntos=0, grabando=False, pico_evento=0.0, mostrar_ayuda=True
):
    H, W = frame.shape[:2]
    s = max(0.6, W / 960.0)

    def S(v):
        return int(round(v * s))

    pulso = 0.5 + 0.5 * math.sin(t * 2 * math.pi * (1.6 if nivel == 3 else 0.8))
    m = S(24)

    if nivel == 3:
        vineta_alerta(frame, ROJO, 0.55 + 0.45 * pulso)
        borde_col = mezclar(oscurecer(ROJO, 0.55), ROJO, pulso)
        cv2.rectangle(frame, (0, 0), (W - 1, H - 1), borde_col, S(6))

    pw, ph = S(420), S(195)
    panel_vidrio(frame, m, m, pw, ph, S(18))
    x0 = m + S(24)

    texto(frame, "MONITOREO SISMICO CARTESIANO", (x0, m + S(34)), 0.46 * s, GRIS, 1, espaciado=S(2), sombra=False)

    cx, cy = m + pw - S(28), m + S(29)
    radio_halo = S(5) + int(S(6) * pulso)
    cv2.circle(frame, (cx, cy), radio_halo, oscurecer(color, 0.25 + 0.5 * (1 - pulso)), 1, AA)
    cv2.circle(frame, (cx, cy), S(5), color, -1, AA)

    esc_v, gr_v = 1.05 * s, max(2, S(2))
    base_y = m + S(82)

    txt_x = f"X: {pos_x:+.2f}"
    dir_x = "DER" if pos_x > 0.5 else ("IZQ" if pos_x < -0.5 else "0")
    tw_x = texto(frame, txt_x, (x0, base_y), esc_v, color, gr_v, sombra=True)
    texto(frame, f"px ({dir_x})", (x0 + tw_x + S(8), base_y), 0.48 * s, GRIS, 1, sombra=False)

    x_col2 = x0 + S(195)
    txt_y = f"Y: {pos_y:+.2f}"
    dir_y = "ARR" if pos_y > 0.5 else ("ABJ" if pos_y < -0.5 else "0")
    tw_y = texto(frame, txt_y, (x_col2, base_y), esc_v, color, gr_v, sombra=True)
    texto(frame, f"px ({dir_y})", (x_col2 + tw_y + S(8), base_y), 0.48 * s, GRIS, 1, sombra=False)

    texto(frame, "POSICION DESDE ORIGEN (0,0)", (x0, m + S(122)), 0.42 * s, GRIS, 1, espaciado=S(2), sombra=False)

    gx0, gy0 = x0, m + S(138)
    ancho_total = pw - 2 * S(24)
    hueco = S(4)
    seg_w = (ancho_total - 3 * hueco) // 4
    alto_barra = max(6, S(10))
    for i, col in enumerate(COLORES_SEGMENTOS):
        sx = gx0 + i * (seg_w + hueco)
        barra(frame, sx, gy0, seg_w, alto_barra, mezclar(PANEL, col, 0.28))
        if gauge_t > i:
            barra(frame, sx, gy0, max(alto_barra, int(seg_w * min(1.0, gauge_t - i))), alto_barra, col)
    i = min(int(gauge_t), 3)
    frac = 1.0 if gauge_t >= 4 else gauge_t - i
    xm = gx0 + i * (seg_w + hueco) + int(frac * seg_w)
    ym = gy0 + alto_barra // 2
    cv2.circle(frame, (xm, ym), S(8), (20, 14, 10), -1, AA)
    cv2.circle(frame, (xm, ym), S(6), BLANCO, -1, AA)

    marcas = [("0", gx0, "izq"), ("2.8", gx0 + seg_w + hueco // 2, "centro"),
              ("8", gx0 + 2 * seg_w + hueco + hueco // 2, "centro"),
              ("25", gx0 + 3 * seg_w + 2 * hueco + hueco // 2, "centro")]
    for etiqueta, mx, ancla in marcas:
        texto(frame, etiqueta, (mx, gy0 + alto_barra + S(18)), 0.36 * s, oscurecer(GRIS, 0.8), 1,
              ancla=ancla, sombra=False)

    nombre, _, sub = estado.partition(" - ")
    nombre = nombre.upper()
    esc_n, esc_sub = 0.62 * s, 0.4 * s
    w_n = ancho_texto(nombre, esc_n, 1)
    w_s = ancho_texto(sub, esc_sub, 1, S(2)) if sub else 0
    bw = S(18) + S(14) + S(12) + w_n + ((S(16) + w_s) if sub else 0) + S(20)
    bh = S(40)
    bx, by = m, m + ph + S(12)
    k = 0.30 + (0.30 * pulso if nivel == 3 else 0.0)
    panel_vidrio(frame, bx, by, bw, bh, bh // 2, tinte=mezclar(PANEL, color, k), alpha=0.72)
    cv2.circle(frame, (bx + S(26), by + bh // 2), S(6), color, -1, AA)
    px = bx + S(26) + S(6) + S(12)
    texto(frame, nombre, (px, by + bh // 2 + S(7)), esc_n, BLANCO, 1)
    if sub:
        texto(frame, sub, (px + w_n + S(16), by + bh // 2 + S(5)), esc_sub, mezclar(BLANCO, color, 0.35), 1,
              espaciado=S(2), sombra=False)

    cw, ch = S(150), S(58)
    panel_vidrio(frame, W - m - cw, m, cw, ch, S(14))
    texto(frame, time.strftime("%H:%M:%S"), (W - m - cw + S(18), m + S(30)), 0.8 * s, BLANCO, 1)
    texto(frame, time.strftime("%d/%m/%Y"), (W - m - cw + S(18), m + S(48)), 0.4 * s, GRIS, 1,
          espaciado=S(1), sombra=False)

    color_etiqueta = ROJO if cant_puntos == 0 else (CIAN if modo_manual else GRIS)
    etiqueta_pts = f"PUNTOS {'MANUAL' if modo_manual else 'AUTO'}: {cant_puntos}"
    texto(frame, etiqueta_pts, (W - m, m + ch + S(22)), 0.42 * s, color_etiqueta, 1,
          espaciado=S(1), ancla="der", sombra=True)

    if grabando:
        rw, rh = S(185), S(34)
        rx, ry = W - m - rw, m + ch + S(34)
        panel_vidrio(frame, rx, ry, rw, rh, rh // 2, tinte=mezclar(PANEL, ROJO, 0.45), alpha=0.80)
        if int(t * 2) % 2 == 0:
            cv2.circle(frame, (rx + S(18), ry + rh // 2), S(5), ROJO, -1, AA)
        texto(frame, f"REC  MAX {pico_evento:.1f}px", (rx + S(30), ry + rh // 2 + S(5)), 0.40 * s, BLANCO, 1,
              espaciado=S(1), sombra=False)

    if nivel == 3:
        bw2, bh2 = S(440), S(64)
        panel_vidrio(frame, (W - bw2) // 2, H - m - bh2, bw2, bh2, S(16),
                     tinte=mezclar(PANEL, ROJO, 0.55 + 0.35 * pulso), alpha=0.82)
        texto(frame, "TERREMOTO", ((W) // 2, H - m - bh2 // 2 + S(12)), 1.25 * s, BLANCO, max(2, S(2)),
              ancla="centro", sombra=True)
    elif mostrar_ayuda:
        texto(frame, "CLIC IZQ: PONER PUNTO   CLIC DER: QUITAR   C: AUTO   P: MOSTRAR/OCULTAR PUNTOS   R: RESET   Q: SALIR",
              (m, H - S(18)), 0.38 * s, oscurecer(GRIS, 0.9), 1, espaciado=S(1), sombra=True)


def detectar_puntos_distribuidos(gray, puntos_existentes=None, max_puntos=10):
    H, W = gray.shape
    filas, cols = 2, 5
    celda_h, celda_w = H // filas, W // cols
    margen_borde = 20

    existentes_2d = (
        puntos_existentes.reshape(-1, 2)
        if puntos_existentes is not None and len(puntos_existentes) > 0
        else np.empty((0, 2), dtype=np.float32)
    )

    nuevos_pts = []
    for r in range(filas):
        for c in range(cols):
            if len(existentes_2d) + len(nuevos_pts) >= max_puntos:
                break

            y1 = max(margen_borde, r * celda_h)
            y2 = min(H - margen_borde, (r + 1) * celda_h)
            x1 = max(margen_borde, c * celda_w)
            x2 = min(W - margen_borde, (c + 1) * celda_w)

            if len(existentes_2d) > 0:
                en_celda = (
                    (existentes_2d[:, 0] >= x1) & (existentes_2d[:, 0] < x2) &
                    (existentes_2d[:, 1] >= y1) & (existentes_2d[:, 1] < y2)
                )
                if np.any(en_celda):
                    continue

            mascara = np.zeros_like(gray)
            mascara[y1:y2, x1:x2] = 255
            pt = cv2.goodFeaturesToTrack(
                gray, mask=mascara, maxCorners=1, qualityLevel=0.01, minDistance=15, blockSize=7
            )
            if pt is not None:
                nuevos_pts.append(pt[0])

    faltantes = max_puntos - (len(existentes_2d) + len(nuevos_pts))
    if faltantes > 0:
        todos_actuales = (
            np.vstack([existentes_2d, np.array(nuevos_pts).reshape(-1, 2)])
            if len(nuevos_pts) > 0
            else existentes_2d
        )
        extras = cv2.goodFeaturesToTrack(
            gray, maxCorners=max_puntos * 2, qualityLevel=0.01, minDistance=20, blockSize=7
        )
        if extras is not None:
            for e in extras:
                if len(existentes_2d) + len(nuevos_pts) >= max_puntos:
                    break
                if len(todos_actuales) == 0 or np.min(np.linalg.norm(todos_actuales - e.ravel(), axis=1)) > 20:
                    nuevos_pts.append(e)
                    todos_actuales = np.vstack([todos_actuales, e.reshape(1, 2)])

    if len(nuevos_pts) > 0:
        arr_nuevos = np.array(nuevos_pts, dtype=np.float32).reshape(-1, 1, 2)
        if puntos_existentes is not None and len(puntos_existentes) > 0:
            return np.vstack((puntos_existentes, arr_nuevos))[:max_puntos]
        return arr_nuevos[:max_puntos]

    return puntos_existentes


def evaluar_nivel(mag):
    if mag > 25.0:
        return "TERREMOTO - ALERTA ROJA", ROJO, 3
    if mag > 8.0:
        return "Sismo Fuerte - ALERTA NARANJA", NARANJA, 2
    if mag > 0.0:
        return "Sismo Leve - ALERTA AMARILLA", AMARILLO, 1
    return "Normal", VERDE, 0


def nombre_nivel_corto(nivel):
    return {1: "SISMO_LEVE", 2: "SISMO_FUERTE", 3: "TERREMOTO"}.get(nivel, "NORMAL")


def clasificacion_csv(nivel):
    return {
        1: "Leve",
        2: "Fuerte",
        3: "Terremoto",
    }.get(nivel, "Normal")


ENCABEZADOS_CSV = [
    "Evento",
    "Fecha",
    "Inicio",
    "Fin",
    "Duración",
    "Tipo",
    "Max Izq (-X)",
    "Max Der (+X)",
    "Max Abj (-Y)",
    "Max Arr (+Y)",
    "Archivo de Video",
]


def centrar_celdas(valores):
    anchos_excel = [10, 12, 10, 10, 10, 11, 14, 14, 14, 14, 45]
    fila_centrada = []
    for i, val in enumerate(valores):
        ancho = anchos_excel[i] if i < len(anchos_excel) else len(str(val)) + 4
        fila_centrada.append(str(val).center(ancho))
    return fila_centrada


def inicializar_carpeta_registro():
    os.makedirs(CARPETA_GRABACIONES, exist_ok=True)
    if not os.path.exists(ARCHIVO_REGISTRO_CSV) or os.path.getsize(ARCHIVO_REGISTRO_CSV) == 0:
        with open(ARCHIVO_REGISTRO_CSV, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["REGISTRO DE MONITOREO SÍSMICO CARTESIANO"])
            writer.writerow([])
            writer.writerow(centrar_celdas(ENCABEZADOS_CSV))


def obtener_siguiente_num_evento():
    if not os.path.exists(ARCHIVO_REGISTRO_CSV):
        return 1
    with open(ARCHIVO_REGISTRO_CSV, mode="r", encoding="utf-8-sig") as f:
        filas = list(csv.reader(f, delimiter=";"))
    eventos = [r for r in filas[3:] if any(c.strip() for c in r)]
    return len(eventos) + 1


def guardar_evento_csv(fecha_ini, hora_ini, hora_fin, duracion, nivel_max, max_izq, max_der, max_abj, max_arr, nombre_archivo):
    inicializar_carpeta_registro()
    num_evento = obtener_siguiente_num_evento()
    fecha_linda = time.strftime("%d/%m/%Y", time.strptime(fecha_ini, "%Y-%m-%d"))

    with open(ARCHIVO_REGISTRO_CSV, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(centrar_celdas([
            f"#{num_evento:03d}",
            fecha_linda,
            hora_ini,
            hora_fin,
            f"{duracion:.2f} s",
            clasificacion_csv(nivel_max),
            f"-{max_izq:.2f} px",
            f"+{max_der:.2f} px",
            f"-{max_abj:.2f} px",
            f"+{max_arr:.2f} px",
            nombre_archivo,
        ]))


inicializar_carpeta_registro()

cap = cv2.VideoCapture(0)

ret, prev_frame = cap.read()
if not ret:
    cap.release()
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
H_img, W_img = prev_gray.shape

fps_cam = cap.get(cv2.CAP_PROP_FPS)
if not fps_cam or fps_cam <= 1 or fps_cam > 120:
    fps_cam = 30.0

MAX_PUNTOS_AUTO = 10
MAX_PUNTOS_MANUAL = 10
lk_params = dict(
    winSize=(41, 41),
    maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# Inicia en modo manual para que el usuario coloque los puntos; presionar 'C' activa autopuntos
p0 = None
modo_manual = True

UMBRAL_MOVIMIENTO_EJE = 0.25
UMBRAL_VIENTO = 2.80
CUADROS_MIN_SISMO = 2
ERROR_MAX_FLUJO = 25.0

rachas_movimiento = 0
TIEMPO_RETENCION = 1.5
nivel_retenido = 0
estado_retenido = "Normal"
color_retenido = VERDE
ultimo_pico_t = 0.0
intensidad_retenida = 0.0

buffer_previo = deque(maxlen=int(fps_cam * 2))
grabando = False
writer_video = None
ruta_temp_video = ""
fecha_inicio_ev = ""
hora_inicio_ev = ""
t_inicio_ev = 0.0
t_ultimo_mov_ev = 0.0
nivel_max_ev = 0

max_izq_ev = 0.0
max_der_ev = 0.0
max_abj_ev = 0.0
max_arr_ev = 0.0

MARGEN_POST_SISMO = 2.0
gauge_t = 0.0
mostrar_puntos = True
NOMBRE_VENTANA = "Monitor Sismico Estable"


def finalizar_grabacion():
    global grabando, writer_video
    if not grabando or writer_video is None:
        return

    writer_video.release()
    writer_video = None
    grabando = False

    hora_fin_ev = time.strftime("%H:%M:%S")
    duracion = max(0.1, time.time() - t_inicio_ev)
    etiqueta_niv = nombre_nivel_corto(nivel_max_ev)
    pico_total_ref = max(max_der_ev, max_izq_ev, max_arr_ev, max_abj_ev)

    nombre_final = (
        f"sismo_{fecha_inicio_ev}_{hora_inicio_ev.replace(':', '-')}_"
        f"{etiqueta_niv}_max{pico_total_ref:.1f}px.mp4"
    )
    ruta_final = os.path.join(CARPETA_GRABACIONES, nombre_final)

    if os.path.exists(ruta_temp_video):
        os.replace(ruta_temp_video, ruta_final)

    guardar_evento_csv(
        fecha_inicio_ev,
        hora_inicio_ev,
        hora_fin_ev,
        duracion,
        nivel_max_ev,
        max_izq_ev,
        max_der_ev,
        max_abj_ev,
        max_arr_ev,
        nombre_final,
    )


def manejar_clic(event, x, y, flags, param):
    global p0, modo_manual, mostrar_puntos

    if event == cv2.EVENT_LBUTTONDOWN:
        mostrar_puntos = True
        nuevo_pt = np.array([[[float(x), float(y)]]], dtype=np.float32)

        if not modo_manual:
            modo_manual = True
            p0 = nuevo_pt
        else:
            if p0 is None or len(p0) == 0:
                p0 = nuevo_pt
            elif len(p0) < MAX_PUNTOS_MANUAL:
                p0 = np.vstack((p0, nuevo_pt))
            else:
                p0 = np.vstack((p0[1:], nuevo_pt))

    elif event == cv2.EVENT_RBUTTONDOWN:
        if p0 is not None and len(p0) > 0:
            pts_2d = p0.reshape(-1, 2)
            distancias = np.linalg.norm(pts_2d - np.array([x, y]), axis=1)
            idx_cercano = int(np.argmin(distancias))
            if distancias[idx_cercano] <= 25.0:
                p0 = np.delete(p0, idx_cercano, axis=0)
                if len(p0) == 0:
                    p0 = None


cv2.namedWindow(NOMBRE_VENTANA)
cv2.setMouseCallback(NOMBRE_VENTANA, manejar_clic)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    ahora = time.time()
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dx_frame, dy_frame = 0.0, 0.0
    movimiento_bruto = 0.0

    if p0 is not None and len(p0) > 0:
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, p0, None, **lk_params
        )
        if p1 is not None and st is not None:
            validos = st.ravel() == 1
            if err is not None:
                validos = validos & (err.ravel() < ERROR_MAX_FLUJO)

            buenos_antiguos = p0[validos]
            buenos_nuevos = p1[validos]

            if len(buenos_nuevos) > 0:
                en_rango = (
                    (buenos_nuevos[:, 0, 0] >= 10) & (buenos_nuevos[:, 0, 0] < W_img - 10) &
                    (buenos_nuevos[:, 0, 1] >= 10) & (buenos_nuevos[:, 0, 1] < H_img - 10)
                )
                buenos_antiguos = buenos_antiguos[en_rango]
                buenos_nuevos = buenos_nuevos[en_rango]

            if len(buenos_nuevos) > 0:
                desplazamientos = buenos_nuevos.reshape(-1, 2) - buenos_antiguos.reshape(-1, 2)
                vector_mediano = np.median(desplazamientos, axis=0)

                dist_mediana = np.linalg.norm(desplazamientos - vector_mediano, axis=1)
                mad = float(np.median(dist_mediana))
                limite_outlier = max(3.5, 1.8 * mad)
                inliers = dist_mediana <= limite_outlier

                if np.any(inliers):
                    vector_consenso = np.median(desplazamientos[inliers], axis=0)
                    buenos_nuevos = buenos_nuevos[inliers]
                else:
                    vector_consenso = vector_mediano

                movimiento_bruto = float(np.linalg.norm(vector_consenso))

                raw_dx = -float(vector_consenso[0])
                raw_dy = float(vector_consenso[1])
                if abs(raw_dx) >= UMBRAL_MOVIMIENTO_EJE:
                    dx_frame = raw_dx
                if abs(raw_dy) >= UMBRAL_MOVIMIENTO_EJE:
                    dy_frame = raw_dy

                p0 = buenos_nuevos.reshape(-1, 1, 2)
            else:
                p0 = None
        else:
            p0 = None

    if not modo_manual and movimiento_bruto < 15.0 and (p0 is None or len(p0) < MAX_PUNTOS_AUTO):
        p0 = detectar_puntos_distribuidos(curr_gray, p0, max_puntos=MAX_PUNTOS_AUTO)

    if mostrar_puntos and p0 is not None and len(p0) > 0:
        for pt in p0:
            nx, ny = map(int, pt.ravel())
            if modo_manual:
                cv2.circle(frame, (nx, ny), 6, CIAN, 1, cv2.LINE_8)
                cv2.circle(frame, (nx, ny), 2, BLANCO, -1, cv2.LINE_8)
            else:
                cv2.circle(frame, (nx, ny), 3, (230, 230, 230), -1, cv2.LINE_8)

    cant_actual = len(p0) if p0 is not None else 0
    magnitud_mov = math.hypot(dx_frame, dy_frame)

    if magnitud_mov >= UMBRAL_VIENTO:
        rachas_movimiento += 1
    else:
        rachas_movimiento = 0

    intensidad_actual = magnitud_mov if rachas_movimiento >= CUADROS_MIN_SISMO else 0.0

    if cant_actual == 0 and modo_manual:
        nivel_retenido = 0
        estado_retenido = "Sin Puntos - EN ESPERA"
        color_retenido = GRIS
        intensidad_retenida = 0.0
        rachas_movimiento = 0
    else:
        # Acumula X e Y también en movimientos lentos/leves (>= UMBRAL_MOVIMIENTO_EJE) sin disparar alerta de sismo
        if magnitud_mov >= UMBRAL_MOVIMIENTO_EJE:
            pos_total_x += dx_frame
            pos_total_y += dy_frame

        estado_actual, color_actual, nivel_actual = evaluar_nivel(intensidad_actual)

        if nivel_actual >= nivel_retenido and nivel_actual > 0:
            nivel_retenido = nivel_actual
            estado_retenido = estado_actual
            color_retenido = color_actual
            intensidad_retenida = max(intensidad_retenida, intensidad_actual)
            ultimo_pico_t = ahora
        elif nivel_retenido > 0:
            if intensidad_actual >= intensidad_retenida * 0.6:
                intensidad_retenida = intensidad_actual
            if (ahora - ultimo_pico_t) > TIEMPO_RETENCION:
                nivel_retenido = nivel_actual
                estado_retenido = estado_actual
                color_retenido = color_actual
                intensidad_retenida = intensidad_actual
                if nivel_actual > 0:
                    ultimo_pico_t = ahora
        elif nivel_retenido == 0:
            estado_retenido = estado_actual
            color_retenido = color_actual
            intensidad_retenida = intensidad_actual

    if nivel_retenido > 0:
        intensidad_barra = max(intensidad_actual, intensidad_retenida * 0.75)
    else:
        intensidad_barra = intensidad_actual

    objetivo_gauge = pos_gauge(intensidad_barra)
    velocidad_gauge = 0.55 if objetivo_gauge > gauge_t else 0.18
    gauge_t += (objetivo_gauge - gauge_t) * velocidad_gauge

    pico_actual_ref = max(max_izq_ev, max_der_ev, max_arr_ev, max_abj_ev)
    dibujar_hud(
        frame, pos_total_x, pos_total_y, estado_retenido, nivel_retenido, color_retenido,
        gauge_t, ahora, modo_manual=modo_manual, cant_puntos=cant_actual,
        grabando=grabando, pico_evento=pico_actual_ref
    )

    if nivel_retenido > 0:
        t_ultimo_mov_ev = ahora
        if not grabando:
            grabando = True
            fecha_inicio_ev = time.strftime("%Y-%m-%d")
            hora_inicio_ev = time.strftime("%H:%M:%S")
            t_inicio_ev = ahora
            nivel_max_ev = nivel_retenido

            max_izq_ev = abs(min(0.0, pos_total_x))
            max_der_ev = max(0.0, pos_total_x)
            max_abj_ev = abs(min(0.0, pos_total_y))
            max_arr_ev = max(0.0, pos_total_y)

            ruta_temp_video = os.path.join(CARPETA_GRABACIONES, f"grabando_{int(ahora)}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer_video = cv2.VideoWriter(ruta_temp_video, fourcc, fps_cam, (W_img, H_img))

            for f_prev in buffer_previo:
                writer_video.write(f_prev)
            buffer_previo.clear()
        else:
            if nivel_retenido > nivel_max_ev:
                nivel_max_ev = nivel_retenido

            if pos_total_x < 0 and abs(pos_total_x) > max_izq_ev:
                max_izq_ev = abs(pos_total_x)
            if pos_total_x > 0 and pos_total_x > max_der_ev:
                max_der_ev = pos_total_x
            if pos_total_y < 0 and abs(pos_total_y) > max_abj_ev:
                max_abj_ev = abs(pos_total_y)
            if pos_total_y > 0 and pos_total_y > max_arr_ev:
                max_arr_ev = pos_total_y

        if writer_video is not None:
            writer_video.write(frame)
    else:
        if grabando:
            if writer_video is not None:
                writer_video.write(frame)
            if (ahora - t_ultimo_mov_ev) >= MARGEN_POST_SISMO:
                finalizar_grabacion()
        else:
            buffer_previo.append(frame.copy())

    cv2.imshow(NOMBRE_VENTANA, frame)

    prev_gray = curr_gray.copy()

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord("q"):
        break
    if tecla == ord("p"):
        mostrar_puntos = not mostrar_puntos
    if tecla == ord("c"):
        modo_manual = False
        p0 = detectar_puntos_distribuidos(curr_gray, max_puntos=MAX_PUNTOS_AUTO)
    if tecla == ord("r"):
        pos_total_x = 0.0
        pos_total_y = 0.0

if grabando:
    finalizar_grabacion()

cap.release()
cv2.destroyAllWindows()

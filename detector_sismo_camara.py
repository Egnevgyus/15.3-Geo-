import math
import time
from functools import lru_cache

import cv2
import numpy as np

VERDE = (153, 211, 52)
AMARILLO = (36, 191, 251)
NARANJA = (60, 146, 251)
ROJO = (68, 68, 239)
GRIS = (175, 163, 156)
BLANCO = (245, 245, 245)
PANEL = (30, 20, 14)

FUENTE = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA
COLORES_SEGMENTOS = (VERDE, AMARILLO, NARANJA, ROJO)


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

    def dibujar(dx, dy, col, gr):
        if espaciado == 0:
            cv2.putText(img, s, (x + dx, y + dy), FUENTE, escala, col, gr, AA)
        else:
            cx = x + dx
            for c in s:
                cv2.putText(img, c, (cx, y + dy), FUENTE, escala, col, gr, AA)
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
    if v < 1.5:
        return v / 1.5
    if v < 5.0:
        return 1 + (v - 1.5) / 3.5
    if v < 10.0:
        return 2 + (v - 5.0) / 5.0
    return min(4.0, 3 + (v - 10.0) / 5.0)


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


def dibujar_hud(frame, mag, estado, nivel, color, gauge_t, t, mostrar_ayuda=True):
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

    pw, ph = S(400), S(188)
    panel_vidrio(frame, m, m, pw, ph, S(18))
    x0 = m + S(24)

    texto(frame, "MONITOR SISMICO", (x0, m + S(34)), 0.5 * s, GRIS, 1, espaciado=S(2), sombra=False)

    cx, cy = m + pw - S(28), m + S(29)
    radio_halo = S(5) + int(S(6) * pulso)
    cv2.circle(frame, (cx, cy), radio_halo, oscurecer(color, 0.25 + 0.5 * (1 - pulso)), 1, AA)
    cv2.circle(frame, (cx, cy), S(5), color, -1, AA)

    valor = f"{mag:.2f}"
    esc_v, gr_v = 2.4 * s, max(2, S(3))
    base_y = m + S(104)
    gx, gy = x0 - S(10), m + S(40)
    gw, gh = pw - S(30), S(80)
    roi = frame[gy:gy + gh, gx:gx + gw]
    capa = np.zeros_like(roi)
    texto(capa, valor, (x0 - gx, base_y - gy), esc_v, color, gr_v, sombra=False)
    brillo = cv2.GaussianBlur(capa, (0, 0), S(7))
    roi[:] = cv2.add(roi, (brillo * (0.55 + 0.25 * nivel)).astype(np.uint8))
    tw = texto(frame, valor, (x0, base_y), esc_v, color, gr_v, sombra=True)
    texto(frame, "px", (x0 + tw + S(10), base_y), 0.8 * s, GRIS, 1, sombra=False)

    texto(frame, "VELOCIDAD ESTRUCTURAL", (x0, m + S(130)), 0.42 * s, GRIS, 1, espaciado=S(2), sombra=False)

    gx0, gy0 = x0, m + S(144)
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

    marcas = [("0", gx0, "izq"), ("1.5", gx0 + seg_w + hueco // 2, "centro"),
              ("5", gx0 + 2 * seg_w + hueco + hueco // 2, "centro"),
              ("10", gx0 + 3 * seg_w + 2 * hueco + hueco // 2, "centro")]
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

    if nivel == 3:
        bw2, bh2 = S(440), S(64)
        panel_vidrio(frame, (W - bw2) // 2, H - m - bh2, bw2, bh2, S(16),
                     tinte=mezclar(PANEL, ROJO, 0.55 + 0.35 * pulso), alpha=0.82)
        texto(frame, "TERREMOTO", ((W) // 2, H - m - bh2 // 2 + S(12)), 1.25 * s, BLANCO, max(2, S(2)),
              ancla="centro", sombra=True)
    elif mostrar_ayuda:
        texto(frame, "Q  SALIR    P  PUNTOS", (m, H - S(18)), 0.4 * s, oscurecer(GRIS, 0.9), 1, espaciado=S(1), sombra=True)


cap = cv2.VideoCapture(0)

ret, prev_frame = cap.read()
if not ret:
  cap.release()
  exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
h_img, w_img = prev_gray.shape

feature_params = dict(
    maxCorners=200, qualityLevel=0.1, minDistance=10, blockSize=7
)
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
smoothed_mag = 0.0

gauge_t = 0.0
mostrar_puntos = True

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  magnitude = 0.0

  if p0 is not None and len(p0) > 10:
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, p0, None, **lk_params
    )

    if p1 is not None:
      good_old = p0[st == 1]
      good_new = p1[st == 1]

      if len(good_old) > 10:
        displacements = good_new - good_old
        magnitudes = np.linalg.norm(displacements, axis=1)
        magnitude = np.median(magnitudes)

        if mostrar_puntos:
          for pt in good_new:
            nx, ny = map(int, pt.ravel())
            cv2.circle(frame, (nx, ny), 2, (230, 230, 230), -1, AA)

  smoothed_mag = (0.6 * smoothed_mag) + (0.4 * magnitude)

  if smoothed_mag < 1.5:
    smoothed_mag = 0.0

  estado = "Normal"
  color = VERDE
  nivel = 0

  if smoothed_mag > 10.0:
    estado = "TERREMOTO - ALERTA ROJA"
    color = ROJO
    nivel = 3
  elif smoothed_mag > 5.0:
    estado = "Sismo Fuerte - ALERTA NARANJA"
    color = NARANJA
    nivel = 2
  elif smoothed_mag > 0.0:
    estado = "Sismo Leve - ALERTA AMARILLA"
    color = AMARILLO
    nivel = 1

  gauge_t += (pos_gauge(smoothed_mag) - gauge_t) * 0.25
  dibujar_hud(frame, smoothed_mag, estado, nivel, color, gauge_t, time.time())

  cv2.imshow("Monitor Sismico Estable", frame)

  prev_gray = curr_gray.copy()
  p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
  if p0 is None or len(p0) < 10:
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

  tecla = cv2.waitKey(1) & 0xFF
  if tecla == ord("q"):
    break
  if tecla == ord("p"):
    mostrar_puntos = not mostrar_puntos

cap.release()
cv2.destroyAllWindows()

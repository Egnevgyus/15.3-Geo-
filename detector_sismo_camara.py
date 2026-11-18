import csv
import html
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
ARCHIVO_REPORTE_HTML = os.path.join(CARPETA_GRABACIONES, "reporte_sismos.html")


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
    if v < 1.2:
        return v / 1.2
    if v < 5.0:
        return 1 + (v - 1.2) / 3.8
    if v < 20.0:
        return 2 + (v - 5.0) / 15.0
    return min(4.0, 3 + (v - 20.0) / 10.0)


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
    frame, mag, estado, nivel, color, gauge_t, t,
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

    pw, ph = S(400), S(188)
    panel_vidrio(frame, m, m, pw, ph, S(18))
    x0 = m + S(24)

    texto(frame, "MONITOREO SISMICO", (x0, m + S(34)), 0.5 * s, GRIS, 1, espaciado=S(2), sombra=False)

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

    marcas = [("0", gx0, "izq"), ("1.2", gx0 + seg_w + hueco // 2, "centro"),
              ("5", gx0 + 2 * seg_w + hueco + hueco // 2, "centro"),
              ("20", gx0 + 3 * seg_w + 2 * hueco + hueco // 2, "centro")]
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
    etiqueta_pts = f"PUNTOS {'MANUAL' if modo_manual else 'AUTO'}: {cant_puntos}/10"
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
        texto(frame, "CLIC IZQ: PONER PUNTO   CLIC DER: QUITAR   C: AUTO   P: VER/OCULTAR   Q: SALIR",
              (m, H - S(18)), 0.38 * s, oscurecer(GRIS, 0.9), 1, espaciado=S(1), sombra=True)


def detectar_10_puntos_distribuidos(gray, puntos_existentes=None, max_puntos=10):
    """Mantiene exactamente 10 puntos repartidos en una cuadricula 5x2."""
    H, W = gray.shape
    filas, cols = 2, 5
    celda_h, celda_w = H // filas, W // cols
    margen_borde = 25

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
                gray, mask=mascara, maxCorners=1, qualityLevel=0.08, minDistance=20, blockSize=7
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
            gray, maxCorners=max_puntos * 2, qualityLevel=0.08, minDistance=35, blockSize=7
        )
        if extras is not None:
            for e in extras:
                if len(existentes_2d) + len(nuevos_pts) >= max_puntos:
                    break
                if len(todos_actuales) == 0 or np.min(np.linalg.norm(todos_actuales - e.ravel(), axis=1)) > 35:
                    nuevos_pts.append(e)
                    todos_actuales = np.vstack([todos_actuales, e.reshape(1, 2)])

    if len(nuevos_pts) > 0:
        arr_nuevos = np.array(nuevos_pts, dtype=np.float32).reshape(-1, 1, 2)
        if puntos_existentes is not None and len(puntos_existentes) > 0:
            return np.vstack((puntos_existentes, arr_nuevos))[:max_puntos]
        return arr_nuevos[:max_puntos]

    return puntos_existentes


def evaluar_nivel(mag):
    if mag > 20.0:
        return "TERREMOTO - ALERTA ROJA", ROJO, 3
    if mag > 5.0:
        return "Sismo Fuerte - ALERTA NARANJA", NARANJA, 2
    if mag > 0.0:
        return "Sismo Leve - ALERTA AMARILLA", AMARILLO, 1
    return "Normal", VERDE, 0


def nombre_nivel_corto(nivel):
    return {1: "SISMO_LEVE", 2: "SISMO_FUERTE", 3: "TERREMOTO"}.get(nivel, "NORMAL")


def descripcion_nivel_csv(nivel):
    return {
        1: "Sismo Leve (Alerta Amarilla)",
        2: "Sismo Fuerte (Alerta Naranja)",
        3: "Terremoto (Alerta Roja)",
    }.get(nivel, "Normal")


ENCABEZADOS_CSV = [
    "N° Evento",
    "Fecha",
    "Hora Inicio",
    "Hora Fin",
    "Duración (s)",
    "Clasificación",
    "Pico Máximo (px)",
    "Promedio (px)",
    "Archivo de Video",
]


def inicializar_carpeta_registro():
    os.makedirs(CARPETA_GRABACIONES, exist_ok=True)
    if not os.path.exists(ARCHIVO_REGISTRO_CSV) or os.path.getsize(ARCHIVO_REGISTRO_CSV) == 0:
        with open(ARCHIVO_REGISTRO_CSV, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(ENCABEZADOS_CSV)


def leer_eventos_csv():
    if not os.path.exists(ARCHIVO_REGISTRO_CSV):
        return []
    eventos = []
    with open(ARCHIVO_REGISTRO_CSV, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for fila in reader:
            if fila and fila.get("N° Evento"):
                eventos.append(fila)
    return eventos


def regenerar_reporte_html():
    """Genera un panel visual moderno en HTML dentro de la carpeta de grabaciones."""
    eventos = leer_eventos_csv()
    total_eventos = len(eventos)

    max_global = 0.0
    total_terremotos = 0
    for ev in eventos:
        try:
            val_max = float(ev["Pico Máximo (px)"].replace(",", "."))
            if val_max > max_global:
                max_global = val_max
        except (ValueError, KeyError):
            pass
        if "Terremoto" in ev.get("Clasificación", ""):
            total_terremotos += 1

    filas_html = []
    for ev in reversed(eventos):
        clasif = ev.get("Clasificación", "")
        if "Terremoto" in clasif:
            badge_cls = "badge-rojo"
            bar_color = "#ef4444"
        elif "Fuerte" in clasif:
            badge_cls = "badge-naranja"
            bar_color = "#fb923c"
        else:
            badge_cls = "badge-amarillo"
            bar_color = "#facc15"

        try:
            pico_num = float(ev.get("Pico Máximo (px)", "0").replace(",", "."))
        except ValueError:
            pico_num = 0.0
        ancho_barra = min(100, int((pico_num / 30.0) * 100))

        video_file = html.escape(ev.get("Archivo de Video", ""))
        filas_html.append(f"""
            <tr>
                <td class="col-id">{html.escape(ev.get('N° Evento', ''))}</td>
                <td>{html.escape(ev.get('Fecha', ''))}</td>
                <td class="mono">{html.escape(ev.get('Hora Inicio', ''))} &rarr; {html.escape(ev.get('Hora Fin', ''))}</td>
                <td class="mono">{html.escape(ev.get('Duración (s)', ''))} s</td>
                <td><span class="badge {badge_cls}">{html.escape(clasif)}</span></td>
                <td>
                    <div class="metric-wrap">
                        <span class="mono strong">{pico_num:.2f} px</span>
                        <div class="bar-bg"><div class="bar-fill" style="width:{ancho_barra}%;background:{bar_color};"></div></div>
                    </div>
                </td>
                <td class="mono">{html.escape(ev.get('Promedio (px)', ''))} px</td>
                <td><a class="btn-video" href="{video_file}" target="_blank">&#9654; Ver video</a></td>
            </tr>
        """)

    cuerpo_tabla = (
        "\n".join(filas_html)
        if filas_html
        else '<tr><td colspan="8" class="empty">Aún no se han registrado movimientos sísmicos.</td></tr>'
    )

    pagina = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Reporte de Monitoreo Sísmico</title>
<style>
  :root {{
    --bg: #0b0f17;
    --card: #131a26;
    --border: #222d40;
    --text: #e2e8f0;
    --muted: #8b9bb4;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 36px;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  header {{
    display: flex; justify-content: space-between; align-items: flex-end;
    margin-bottom: 28px; border-bottom: 1px solid var(--border); padding-bottom: 18px;
  }}
  h1 {{ font-size: 24px; font-weight: 700; letter-spacing: 0.5px; }}
  .subtitle {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
  .cards {{
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; margin-bottom: 28px;
  }}
  .card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 14px;
    padding: 20px 24px;
  }}
  .card-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); }}
  .card-value {{ font-size: 30px; font-weight: 700; margin-top: 8px; }}
  .table-box {{
    background: var(--card); border: 1px solid var(--border); border-radius: 14px;
    overflow: hidden; box-shadow: 0 12px 30px rgba(0,0,0,0.35);
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th {{
    background: #182131; color: var(--muted); font-size: 11px; text-transform: uppercase;
    letter-spacing: 1px; text-align: left; padding: 14px 16px; border-bottom: 1px solid var(--border);
  }}
  td {{ padding: 14px 16px; border-bottom: 1px solid var(--border); vertical-align: middle; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(255,255,255,0.02); }}
  .col-id {{ font-weight: 700; color: #38bdf8; }}
  .mono {{ font-variant-numeric: tabular-nums; }}
  .strong {{ font-weight: 700; }}
  .badge {{
    display: inline-block; padding: 5px 11px; border-radius: 999px; font-size: 12px; font-weight: 600;
  }}
  .badge-amarillo {{ background: rgba(250, 204, 21, 0.14); color: #fde047; border: 1px solid rgba(250, 204, 21, 0.35); }}
  .badge-naranja  {{ background: rgba(251, 146, 60, 0.14); color: #fdba74; border: 1px solid rgba(251, 146, 60, 0.35); }}
  .badge-rojo     {{ background: rgba(239, 68, 68, 0.16); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.45); }}
  .metric-wrap {{ display: flex; flex-direction: column; gap: 6px; min-width: 110px; }}
  .bar-bg {{ width: 100%; height: 6px; background: #1e293b; border-radius: 4px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 4px; }}
  .btn-video {{
    display: inline-block; padding: 6px 12px; border-radius: 8px; background: #1e293b;
    color: #e2e8f0; text-decoration: none; font-size: 12px; font-weight: 600;
    border: 1px solid #334155; transition: 0.15s;
  }}
  .btn-video:hover {{ background: #38bdf8; color: #0b0f17; border-color: #38bdf8; }}
  .empty {{ text-align: center; color: var(--muted); padding: 36px; }}
</style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <h1>Historial de Monitoreo Sísmico</h1>
      <div class="subtitle">Registro automático de eventos y grabaciones de video</div>
    </div>
    <div class="subtitle">Actualizado: {time.strftime("%d/%m/%Y %H:%M:%S")}</div>
  </header>

  <div class="cards">
    <div class="card">
      <div class="card-label">Eventos Registrados</div>
      <div class="card-value">{total_eventos}</div>
    </div>
    <div class="card">
      <div class="card-label">Pico Máximo Histórico</div>
      <div class="card-value" style="color:#38bdf8;">{max_global:.2f} px</div>
    </div>
    <div class="card">
      <div class="card-label">Alertas Rojas (Terremoto)</div>
      <div class="card-value" style="color:#f87171;">{total_terremotos}</div>
    </div>
  </div>

  <div class="table-box">
    <table>
      <thead>
        <tr>
          <th>Evento</th>
          <th>Fecha</th>
          <th>Horario</th>
          <th>Duración</th>
          <th>Clasificación</th>
          <th>Pico Máximo</th>
          <th>Promedio</th>
          <th>Grabación</th>
        </tr>
      </thead>
      <tbody>
        {cuerpo_tabla}
      </tbody>
    </table>
  </div>
</div>
</body>
</html>"""

    with open(ARCHIVO_REPORTE_HTML, mode="w", encoding="utf-8") as f:
        f.write(pagina)


def guardar_evento_csv(fecha_ini, hora_ini, hora_fin, duracion, nivel_max, mag_max, mag_prom, nombre_archivo):
    inicializar_carpeta_registro()
    num_evento = len(leer_eventos_csv()) + 1
    fecha_linda = time.strftime("%d/%m/%Y", time.strptime(fecha_ini, "%Y-%m-%d"))

    with open(ARCHIVO_REGISTRO_CSV, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            f"#{num_evento:03d}",
            fecha_linda,
            hora_ini,
            hora_fin,
            f"{duracion:.2f}",
            descripcion_nivel_csv(nivel_max),
            f"{mag_max:.2f}",
            f"{mag_prom:.2f}",
            nombre_archivo,
        ])

    regenerar_reporte_html()


inicializar_carpeta_registro()
regenerar_reporte_html()

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

hann = cv2.createHanningWindow((W_img, H_img), cv2.CV_32F)
prev_float = prev_gray.astype(np.float32)

MAX_PUNTOS = 10
lk_params = dict(
    winSize=(25, 25),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

p0 = detectar_10_puntos_distribuidos(prev_gray, max_puntos=MAX_PUNTOS)
modo_manual = False

UMBRAL_RUIDO_SENSOR = 0.25
smoothed_mag = 0.0

TIEMPO_RETENCION = 1.5
nivel_retenido = 0
estado_retenido = "Normal"
color_retenido = VERDE
ultimo_pico_t = 0.0

buffer_previo = deque(maxlen=int(fps_cam * 2))
grabando = False
writer_video = None
ruta_temp_video = ""
fecha_inicio_ev = ""
hora_inicio_ev = ""
t_inicio_ev = 0.0
t_ultimo_mov_ev = 0.0
pico_mag_ev = 0.0
nivel_max_ev = 0
lecturas_mag_ev = []
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
    mag_prom = float(np.mean(lecturas_mag_ev)) if lecturas_mag_ev else pico_mag_ev
    etiqueta_niv = nombre_nivel_corto(nivel_max_ev)

    nombre_final = (
        f"sismo_{fecha_inicio_ev}_{hora_inicio_ev.replace(':', '-')}_"
        f"{etiqueta_niv}_max{pico_mag_ev:.1f}px.mp4"
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
        pico_mag_ev,
        mag_prom,
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
            elif len(p0) < MAX_PUNTOS:
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
    curr_float = curr_gray.astype(np.float32)
    magnitude = 0.0

    if p0 is not None and len(p0) > 0:
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, p0, None, **lk_params
        )

        if p1 is not None and st is not None:
            validos = st.ravel() == 1
            buenos_antiguos = p0[validos]
            buenos_nuevos = p1[validos]

            if len(buenos_nuevos) > 0:
                if modo_manual:
                    desplazamientos = buenos_nuevos.reshape(-1, 2) - buenos_antiguos.reshape(-1, 2)
                    vector_mediano = np.median(desplazamientos, axis=0)
                    desp_manual = float(np.linalg.norm(vector_mediano))
                    if desp_manual > UMBRAL_RUIDO_SENSOR:
                        magnitude = desp_manual
                else:
                    (dx, dy), respuesta = cv2.phaseCorrelate(prev_float, curr_float, hann)
                    desplazamiento_camara = math.hypot(dx, dy)
                    if respuesta > 0.15 and desplazamiento_camara > UMBRAL_RUIDO_SENSOR:
                        magnitude = desplazamiento_camara

                if mostrar_puntos:
                    for pt in buenos_nuevos:
                        nx, ny = map(int, pt.ravel())
                        if modo_manual:
                            cv2.circle(frame, (nx, ny), 6, CIAN, 1, cv2.LINE_8)
                            cv2.circle(frame, (nx, ny), 2, BLANCO, -1, cv2.LINE_8)
                        else:
                            cv2.circle(frame, (nx, ny), 3, (230, 230, 230), -1, cv2.LINE_8)

                p0 = buenos_nuevos.reshape(-1, 1, 2)
            else:
                p0 = None
        else:
            p0 = None

    if not modo_manual and (p0 is None or len(p0) < MAX_PUNTOS):
        p0 = detectar_10_puntos_distribuidos(curr_gray, p0, max_puntos=MAX_PUNTOS)

    cant_actual = len(p0) if p0 is not None else 0

    if cant_actual == 0:
        smoothed_mag = 0.0
        nivel_retenido = 0
        estado_retenido = "Sin Puntos - EN ESPERA"
        color_retenido = GRIS
    else:
        if magnitude > smoothed_mag:
            smoothed_mag = (0.25 * smoothed_mag) + (0.75 * magnitude)
        else:
            smoothed_mag = (0.92 * smoothed_mag) + (0.08 * magnitude)

        if smoothed_mag < 1.2:
            smoothed_mag = 0.0

        estado_actual, color_actual, nivel_actual = evaluar_nivel(smoothed_mag)

        if nivel_actual >= nivel_retenido and nivel_actual > 0:
            nivel_retenido = nivel_actual
            estado_retenido = estado_actual
            color_retenido = color_actual
            ultimo_pico_t = ahora
        elif nivel_retenido > 0:
            if (ahora - ultimo_pico_t) > TIEMPO_RETENCION:
                nivel_retenido = nivel_actual
                estado_retenido = estado_actual
                color_retenido = color_actual
                if nivel_actual > 0:
                    ultimo_pico_t = ahora
        elif nivel_retenido == 0:
            estado_retenido = estado_actual
            color_retenido = color_actual

    gauge_t += (pos_gauge(smoothed_mag) - gauge_t) * 0.25
    dibujar_hud(
        frame, smoothed_mag, estado_retenido, nivel_retenido, color_retenido,
        gauge_t, ahora, modo_manual=modo_manual, cant_puntos=cant_actual,
        grabando=grabando, pico_evento=pico_mag_ev
    )

    if nivel_retenido > 0 and cant_actual > 0:
        t_ultimo_mov_ev = ahora
        if not grabando:
            grabando = True
            fecha_inicio_ev = time.strftime("%Y-%m-%d")
            hora_inicio_ev = time.strftime("%H:%M:%S")
            t_inicio_ev = ahora
            pico_mag_ev = smoothed_mag
            nivel_max_ev = nivel_retenido
            lecturas_mag_ev = [smoothed_mag]

            ruta_temp_video = os.path.join(CARPETA_GRABACIONES, f"grabando_{int(ahora)}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer_video = cv2.VideoWriter(ruta_temp_video, fourcc, fps_cam, (W_img, H_img))

            for f_prev in buffer_previo:
                writer_video.write(f_prev)
            buffer_previo.clear()
        else:
            if smoothed_mag > pico_mag_ev:
                pico_mag_ev = smoothed_mag
            if nivel_retenido > nivel_max_ev:
                nivel_max_ev = nivel_retenido
            if smoothed_mag > 0:
                lecturas_mag_ev.append(smoothed_mag)

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
    prev_float = curr_float

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord("q"):
        break
    if tecla == ord("p"):
        mostrar_puntos = not mostrar_puntos
    if tecla == ord("c"):
        modo_manual = False
        p0 = detectar_10_puntos_distribuidos(curr_gray, max_puntos=MAX_PUNTOS)

if grabando:
    finalizar_grabacion()

cap.release()
cv2.destroyAllWindows()

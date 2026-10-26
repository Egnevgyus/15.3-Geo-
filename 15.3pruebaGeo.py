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
        capture = cv2.VideoCapture(0)
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

    # Normalizar el patrón a 120x120 para asegurar consistencia absoluta en el tamaño
    patron_norm = cv2.resize(ultima_binaria, (120, 120), interpolation=cv2.INTER_NEAREST)
    _, patron_norm = cv2.threshold(patron_norm, 127, 255, cv2.THRESH_BINARY)

    if cv2.imwrite(ruta, patron_norm):
        messagebox.showinfo("Éxito", f"Patrón guardado correctamente:\n{nombre}.png")
    else:
        messagebox.showerror("Error", "No se pudo guardar el archivo")


# --- PANEL 4: IDENTIFICACIÓN Y SIMILITUD ROBUSTA ---
def calcular_similitud_robusta(img_actual, img_patron):
    """
    Normaliza ambas imágenes a 120x120 y evalúa coincidencia exacta de píxeles
    y correlación cruzada en 4 orientaciones (0°, 90°, 180°, 270°).
    """
    if img_actual is None or img_patron is None:
        return 0.0

    p1 = cv2.resize(img_actual, (120, 120), interpolation=cv2.INTER_NEAREST)
    _, p1 = cv2.threshold(p1, 127, 255, cv2.THRESH_BINARY)

    p2 = cv2.resize(img_patron, (120, 120), interpolation=cv2.INTER_NEAREST)
    _, p2 = cv2.threshold(p2, 127, 255, cv2.THRESH_BINARY)

    mejor_puntaje = 0.0
    versiones_patron = [p2, cv2.bitwise_not(p2)]

    for version in versiones_patron:
        for rot in [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            cand = version if rot is None else cv2.rotate(version, rot)

            # 1. Coincidencia de píxeles binarios (%)
            coincidencias = np.sum(p1 == cand)
            score_pixel = (coincidencias / (120.0 * 120.0)) * 100.0

            # 2. Correlación cruzada normalizada
            res = cv2.matchTemplate(p1, cand, cv2.TM_CCOEFF_NORMED)
            score_tm = max(0.0, float(res[0][0])) * 100.0

            # Ponderación balanceada
            score_final = (score_pixel * 0.5) + (score_tm * 0.5)

            if score_final > mejor_puntaje:
                mejor_puntaje = score_final

    return mejor_puntaje


def identificar_y_comparar():
    global ultima_binaria
    if imagen_base_bgr is None:
        messagebox.showwarning("Aviso", "Primero carga o captura una imagen")
        return

    umbralizar()
    if ultima_binaria is None:
        messagebox.showwarning("Aviso", "No se ha generado el recorte umbralizado")
        return

    archivos = [f for f in os.listdir(CARPETA_PATRONES) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not archivos:
        messagebox.showwarning("Aviso", "No hay patrones en 'patrones_guardados'. Guarda al menos un patrón primero.")
        return

    resultados = []
    mejor_score = -1.0
    pieza_detectada = "Desconocido"

    for archivo_patron in archivos:
        ruta = os.path.join(CARPETA_PATRONES, archivo_patron)
        patron_img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if patron_img is None:
            continue

        nombre_pieza = os.path.splitext(archivo_patron)[0]
        score = calcular_similitud_robusta(ultima_binaria, patron_img)
        resultados.append((nombre_pieza, score))

        if score > mejor_score:
            mejor_score = score
            pieza_detectada = nombre_pieza

    resultados.sort(key=lambda x: x[1], reverse=True)

    # --- PANEL 4: DIBUJAR SOBRE LA IMAGEN ORIGINAL ENTERA ---
    img_resultado = imagen_base_bgr.copy()
    h_img, w_img = img_resultado.shape[:2]
    grosor = max(2, int(max(h_img, w_img) / 320))
    escala_txt = max(0.65, w_img / 800.0)

    if roi_coords_real is not None:
        x1, y1, x2, y2 = roi_coords_real
        w_roi = x2 - x1
        h_roi = y2 - y1

        # Cuadro cian delimitando la zona seleccionada
        cv2.rectangle(img_resultado, (x1, y1), (x2, y2), (255, 255, 0), grosor)

        # Etiqueta con la pieza identificada y su porcentaje
        texto_etiqueta = f"{pieza_detectada} ({mejor_score:.1f}%)"
        pos_y = max(32, y1 - 12)
        cv2.putText(
            img_resultado,
            texto_etiqueta,
            (x1, pos_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            escala_txt,
            (0, 255, 0),
            grosor + 1,
            cv2.LINE_AA
        )

        cx = x1 + w_roi // 2
        cy = y1 + h_roi // 2
        cv2.drawMarker(img_resultado, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, int(22 * escala_txt), grosor)

    mostrar(LImagenResultado, img_resultado, bgr=True, w=300, h=240)

    # Consolas inferiores
    txt_sim = f"{'Pieza Guardada':<20} {'Similitud':>10}\n" + "─" * 34 + "\n"
    for nom, sim in resultados:
        marca = "  <-- DETECTADA" if nom == pieza_detectada else ""
        txt_sim += f"{nom:<20} {sim:>8.1f}%{marca}\n"
    escribir(CajaTemplate, txt_sim)

    x1, y1, x2, y2 = roi_coords_real if roi_coords_real else (0, 0, 0, 0)
    txt_pos = "INFORMACIÓN DE DETECCIÓN:\n"
    txt_pos += f"Pieza Identificada: {pieza_detectada}\n"
    txt_pos += f"Similitud Máxima: {mejor_score:.2f}%\n"
    txt_pos += f"Zona de Recorte: ({x1}, {y1}) a ({x2}, {y2})\n"
    txt_pos += f"Dimensiones ROI: {x2 - x1} x {y2 - y1} px\n"
    txt_pos += f"Centroide Global: ({x1 + (x2 - x1)//2}, {y1 + (y2 - y1)//2})\n"
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

# PANEL 4: Detección en Imagen Completa
F3 = tk.LabelFrame(top_frame, text=" 4. Detección en Imagen Entera ", bg=PANEL, fg=TEXT_MUTED, font=FONT_TITLE, bd=1, relief="solid")
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
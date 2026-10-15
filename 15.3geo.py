import os
from math import copysign, log10
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# --- CONSTANTES Y CONFIGURACIÓN DE TEMA ---
CARPETA_PATRONES = "patrones_guardados"
if not os.path.exists(CARPETA_PATRONES):
    os.makedirs(CARPETA_PATRONES)

BG = "#0f172a"          # Fondo principal profundo
PANEL = "#1e293b"       # Tarjetas / Paneles
BORDER = "#334155"      # Bordes
ACCENT = "#3b82f6"      # Color primario
TEXT_MAIN = "#f8fafc"   # Texto principal
TEXT_MUTED = "#94a3b8"  # Texto secundario

BTN_BLUE = "#2563eb"
BTN_CYAN = "#0284c7"
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

roi_start = None
roi_end = None
drawing_roi = False

# --- VENTANA PRINCIPAL ---
ventana = tk.Tk()
ventana.geometry("1480x780")
ventana.title("Detector de Objetos - Control Visión Optimizado")
ventana.configure(bg=BG)
ventana.resizable(False, False)


def mostrar(label, img_np, bgr=False, w=300, h=240):
    if img_np is None or img_np.size == 0:
        return
    if bgr:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_np = cv2.resize(img_np, (w, h), interpolation=cv2.INTER_NEAREST)
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
    global imagen_base_gray, roi_start, roi_end
    path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png")])
    if not path:
        return
    bgr = cv2.imread(path)
    if bgr is None:
        messagebox.showerror("Error", "No se pudo leer la imagen")
        return

    roi_start = None
    roi_end = None

    mostrar(LImagen, bgr, bgr=True)
    imagen_base_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    recortar_imagen()


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
    global imagen_base_gray, roi_start, roi_end, camara_activa
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

    roi_start = None
    roi_end = None

    imagen_base_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mostrar(LImagen, imagen_base_gray)
    recortar_imagen()


# --- FUNCIONES DE RECORTE INTERACTIVO POR RATÓN ---
def empezar_seleccion(event):
    global roi_start, roi_end, drawing_roi
    if imagen_base_gray is None:
        return
    roi_start = (event.x, event.y)
    roi_end = (event.x, event.y)
    drawing_roi = True


def mover_seleccion(event):
    global roi_end
    if not drawing_roi or imagen_base_gray is None:
        return
    w_label = max(1, LImagen.winfo_width())
    h_label = max(1, LImagen.winfo_height())

    x = max(0, min(event.x, w_label))
    y = max(0, min(event.y, h_label))
    roi_end = (x, y)
    redibujar_seleccion()


def terminar_seleccion(event):
    global drawing_roi
    if not drawing_roi:
        return
    drawing_roi = False
    recortar_imagen()
    if imagen_base_gray is not None:
        mostrar(LImagen, imagen_base_gray)


def redibujar_seleccion():
    if imagen_base_gray is None or roi_start is None or roi_end is None:
        return

    h_real, w_real = imagen_base_gray.shape[:2]
    vis = cv2.cvtColor(imagen_base_gray, cv2.COLOR_GRAY2BGR)

    scale_x = w_real / 300.0
    scale_y = h_real / 240.0

    x1 = int(roi_start[0] * scale_x)
    y1 = int(roi_start[1] * scale_y)
    x2 = int(roi_end[0] * scale_x)
    y2 = int(roi_end[1] * scale_y)

    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    mostrar(LImagen, vis, bgr=True, w=300, h=240)


def recortar_imagen(event=None):
    if imagen_base_gray is None:
        if event is None:
            messagebox.showwarning("Aviso", "Primero carga o captura una imagen válida.")
        return None

    if roi_start is None or roi_end is None:
        recortada = imagen_base_gray.copy()
    else:
        h_real, w_real = imagen_base_gray.shape[:2]

        scale_x = w_real / 300.0
        scale_y = h_real / 240.0

        x1_ui, x2_ui = sorted([roi_start[0], roi_end[0]])
        y1_ui, y2_ui = sorted([roi_start[1], roi_end[1]])

        x1 = int(x1_ui * scale_x)
        x2 = int(x2_ui * scale_x)
        y1 = int(y1_ui * scale_y)
        y2 = int(y2_ui * scale_y)

        x1 = max(0, min(w_real - 1, x1))
        x2 = max(0, min(w_real, x2))
        y1 = max(0, min(h_real - 1, y1))
        y2 = max(0, min(h_real, y2))

        if (x2 - x1) < 5 or (y2 - y1) < 5:
            recortada = imagen_base_gray.copy()
        else:
            recortada = imagen_base_gray[y1:y2, x1:x2]

    mostrar(LImagen2, recortada, w=300, h=240)
    return recortada


def umbralizar(event=None):
    recortada = recortar_imagen()
    if recortada is None:
        if event is None:
            messagebox.showwarning("Aviso", "Primero carga o captura una imagen válida.")
        return None
    umb = int(SGray.get())
    _, binaria = cv2.threshold(recortada, umb, 255, cv2.THRESH_BINARY)
    mostrar(LImagenUmbral, binaria)
    return binaria


def guardar_patron():
    binaria = umbralizar()
    if binaria is None:
        messagebox.showwarning("Aviso", "Carga o captura una imagen válida")
        return
    nombre = EntradaNombre.get().strip().replace(" ", "") or "patron"
    ruta = os.path.join(CARPETA_PATRONES, f"{nombre}.png")
    if cv2.imwrite(ruta, binaria):
        messagebox.showinfo(
            "Éxito",
            f"Patrón guardado correctamente en:\n{ruta}\n\nTamaño: {binaria.shape[1]}x{binaria.shape[0]} px",
        )
    else:
        messagebox.showerror("Error", "No se pudo guardar la imagen")


def obtener_hu(imagen_bin):
    cnts, _ = cv2.findContours(imagen_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    moments = cv2.moments(cnt)
    if moments["m00"] == 0:
        return None
    hu = cv2.HuMoments(moments).flatten()
    return [-copysign(1.0, v) * log10(abs(v)) if v != 0 else 0.0 for v in hu]


def contorno_mayor(imagen_bin):
    cnts, _ = cv2.findContours(imagen_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea) if cnts else None


def rotar_template(template, angulo):
    if angulo == 0:
        return template
    h, w = template.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angulo, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy
    return cv2.warpAffine(
        template, M, (new_w, new_h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )


def comparar_todo():
    if imagen_base_gray is None:
        messagebox.showwarning("Aviso", "Primero carga o captura una imagen")
        return

    archivos = [f for f in os.listdir(CARPETA_PATRONES) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not archivos:
        messagebox.showwarning("Aviso", "No hay patrones guardados en la carpeta")
        return

    img_evaluar = recortar_imagen()
    if img_evaluar is None:
        return

    umb = int(SGray.get())
    paso = max(1, int(SAngulo.get()))
    
    # 1. Binarizar la imagen recortada para la evaluación actual
    _, img_bin = cv2.threshold(img_evaluar, umb, 255, cv2.THRESH_BINARY)
    ih, iw = img_bin.shape[:2]

    cnt_actual = contorno_mayor(img_bin)
    hu_actual = obtener_hu(img_bin)

    res_tm, res_shapes, res_hu = [], [], []
    mejor_tm_val = -1.0
    mejor_patron_img = None
    nombre_mas_parecido = ""

    # 2. Iterar sobre cada patrón umbralizado guardado
    for archivo_patron in archivos:
        path = os.path.join(CARPETA_PATRONES, archivo_patron)
        patron = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if patron is None:
            continue

        # Asegurar binarización limpia del patrón cargado
        _, patron_bin = cv2.threshold(patron, 127, 255, cv2.THRESH_BINARY)
        nombre = os.path.splitext(archivo_patron)[0]

        best_v = -1.0

        # Comparar normal y versión invertida para tolerar fondos blanco/negro
        for p_eval in (patron_bin, cv2.bitwise_not(patron_bin)):
            for angulo in range(0, 360, paso):
                rot = rotar_template(p_eval, angulo)
                th, tw = rot.shape[:2]

                # Búsqueda multiescala
                for scale in (0.7, 0.85, 1.0, 1.15):
                    nw, nh = int(tw * scale), int(th * scale)

                    if nh >= ih or nw >= iw or nh < 10 or nw < 10:
                        continue

                    rot_resized = cv2.resize(rot, (nw, nh), interpolation=cv2.INTER_NEAREST)
                    r = cv2.matchTemplate(img_bin, rot_resized, cv2.TM_CCOEFF_NORMED)
                    _, v, _, _ = cv2.minMaxLoc(r)

                    if v > best_v:
                        best_v = v

        sim = best_v * 100 if best_v >= 0 else 0.0
        res_tm.append((nombre, sim))

        if best_v > mejor_tm_val:
            mejor_tm_val = best_v
            mejor_patron_img = patron_bin.copy()
            nombre_mas_parecido = nombre

        # Comparación suplementaria de formas (matchShapes y Hu)
        cnt_p = contorno_mayor(patron_bin)
        score = (
            cv2.matchShapes(cnt_actual, cnt_p, cv2.CONTOURS_MATCH_I2, 0)
            if cnt_actual is not None and cnt_p is not None
            else float("inf")
        )
        res_shapes.append((nombre, score))

        hu_p = obtener_hu(patron_bin)
        dist = (
            float(np.linalg.norm(np.array(hu_actual) - np.array(hu_p)))
            if hu_actual and hu_p
            else float("inf")
        )
        res_hu.append((nombre, dist))

    # Ordenar resultados de mayor a menor coincidencia
    res_tm.sort(key=lambda x: x[1], reverse=True)
    res_shapes.sort(key=lambda x: x[1])
    res_hu.sort(key=lambda x: x[1])

    # 3. Mostrar la imagen umbralizada que MÁS SE PARECE
    if mejor_patron_img is not None:
        mostrar(LImagenResultado, mejor_patron_img, w=300, h=220)

    # 4. Imprimir resultados en las consolas inferiores indicando la más parecida
    txt = f"{'Patrón':<18} {'Similitud':>10}\n" + "─" * 32 + "\n"
    for nom, sim in res_tm:
        marca = " ◀ (MÁS PARECIDO)" if nom == nombre_mas_parecido else ""
        txt += f"{nom:<18} {sim:>8.2f}%{marca}\n"
    escribir(CajaTemplate, txt)

    txt = f"{'Patrón':<18} {'Diferencia':>10}\n" + "─" * 32 + "\n"
    for nom, sc in res_shapes:
        marca = " ◀" if res_shapes and nom == res_shapes[0][0] else ""
        txt += f"{nom:<18} {sc:>10.5f}{marca}\n"
    escribir(CajaShapes, txt)

    txt = f"{'Patrón':<16} {'Dist':>6}  Calidad\n" + "─" * 38 + "\n"
    for nom, d in res_hu:
        marca = " ◀" if res_hu and nom == res_hu[0][0] else ""
        cal = "sin forma" if d == float("inf") else ("muy similar" if d < 0.5 else ("parecido" if d < 1.5 else "diferente"))
        txt += f"{nom:<16} {d:>6.3f}  {cal}{marca}\n"
    escribir(CajaHu, txt)


def crear_scale(parent, label_text, from_val, to_val, command=None, orient="horizontal"):
    return tk.Scale(
        parent, from_=from_val, to=to_val, orient=orient, label=label_text,
        bg=PANEL, fg=TEXT_MAIN, activebackground=ACCENT, troughcolor="#0f172a",
        highlightthickness=0, bd=0, font=FONT_LABEL, command=command,
    )

top_frame = tk.Frame(ventana, bg=BG)
top_frame.pack(fill="x", padx=12, pady=10)

F0 = tk.LabelFrame(top_frame, text=" 1. Captura y Selección ", bg=PANEL, fg=TEXT_MUTED, font=FONT_TITLE, bd=1, relief="solid")
F0.pack(side="left", expand=True, fill="both", padx=4)

lbl_info = tk.Label(F0, text="Arrastra con el ratón sobre la imagen para recortar", bg=PANEL, fg=TEXT_MUTED, font=FONT_LABEL)
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

# PANEL F1
F1 = tk.LabelFrame(top_frame, text=" 2. Área Seleccionada ", bg=PANEL, fg=TEXT_MUTED, font=FONT_TITLE, bd=1, relief="solid")
F1.pack(side="left", expand=True, fill="both", padx=4)

FContenedor2 = tk.Frame(F1, width=300, height=240, bg="#0f172a", bd=1, relief="solid")
FContenedor2.pack_propagate(False)
FContenedor2.pack(padx=8, pady=6)

LImagen2 = tk.Label(FContenedor2, bg="#0f172a")
LImagen2.pack(expand=True, fill="both")

SGray = crear_scale(F1, "Umbral Binarización", 0, 255)
SGray.pack(fill="x", padx=8)

SAngulo = crear_scale(F1, "Paso Angular Busqueda (°)", 1, 90)
SAngulo.set(5)
SAngulo.pack(fill="x", padx=8)

# PANEL F2
F2 = tk.LabelFrame(top_frame, text=" 3. Umbralización y Guardado ", bg=PANEL, fg=TEXT_MUTED, font=FONT_TITLE, bd=1, relief="solid")
F2.pack(side="left", expand=True, fill="both", padx=4)

FContenedor3 = tk.Frame(F2, width=300, height=240, bg="#0f172a", bd=1, relief="solid")
FContenedor3.pack_propagate(False)
FContenedor3.pack(padx=8, pady=6)

LImagenUmbral = tk.Label(FContenedor3, bg="#0f172a")
LImagenUmbral.pack(expand=True, fill="both")

BUmbral = tk.Button(F2, text="Umbralizar (C)", command=umbralizar, bg=BTN_CYAN, fg=TEXT_MAIN, **STYLE_BTN_BASE)
BUmbral.pack(fill="x", padx=8, pady=4)

lbl_nombre = tk.Label(F2, text="Nombre del Patrón:", bg=PANEL, fg=TEXT_MUTED, font=FONT_LABEL)
lbl_nombre.pack(anchor="w", padx=8, pady=(4, 0))

EntradaNombre = tk.Entry(F2, bg="#0f172a", fg=TEXT_MAIN, font=FONT_CONSOLE, insertbackground=TEXT_MAIN, bd=1, relief="solid", highlightthickness=0)
EntradaNombre.insert(0, "patron")
EntradaNombre.pack(fill="x", padx=8, pady=4)

BGuardar = tk.Button(F2, text="Guardar Patrón", command=guardar_patron, bg=BTN_GREEN, fg=TEXT_MAIN, **STYLE_BTN_BASE)
BGuardar.pack(fill="x", padx=8, pady=6)

# PANEL F3
F3 = tk.LabelFrame(top_frame, text=" 4. Resultado Detección ", bg=PANEL, fg=TEXT_MUTED, font=FONT_TITLE, bd=1, relief="solid")
F3.pack(side="left", expand=True, fill="both", padx=4)

FContenedor4 = tk.Frame(F3, width=300, height=240, bg="#0f172a", bd=1, relief="solid")
FContenedor4.pack_propagate(False)
FContenedor4.pack(padx=8, pady=6)

LImagenResultado = tk.Label(FContenedor4, bg="#0f172a")
LImagenResultado.pack(expand=True, fill="both")

BCompararCamara = tk.Button(F3, text="Comparar Todo", command=comparar_todo, bg=BTN_PURPLE, fg=TEXT_MAIN, **STYLE_BTN_BASE)
BCompararCamara.pack(fill="x", padx=8, pady=12)

# CAJAS RESULTADOS
bottom_frame = tk.Frame(ventana, bg=BG)
bottom_frame.pack(fill="both", expand=True, padx=12, pady=(0, 10))

for i in range(3):
    bottom_frame.columnconfigure(i, weight=1, uniform="cajas")
bottom_frame.rowconfigure(0, weight=1)


def crear_caja_resultado(parent, col, titulo, color_header):
    frame_box = tk.Frame(parent, bg=PANEL, bd=1, relief="solid")
    frame_box.grid(row=0, column=col, sticky="nsew", padx=4, pady=4)

    lbl = tk.Label(frame_box, text=titulo, bg=color_header, fg=TEXT_MAIN, font=FONT_TITLE, pady=3)
    lbl.pack(fill="x")

    text_area = tk.Text(frame_box, bg="#0f172a", fg="#38bdf8", font=FONT_CONSOLE, state="disabled", relief="flat", bd=0, highlightthickness=0, padx=8, pady=8)
    text_area.pack(fill="both", expand=True)
    return text_area


CajaTemplate = crear_caja_resultado(bottom_frame, 0, "Template Matching", "#b45309")
CajaShapes = crear_caja_resultado(bottom_frame, 1, "matchShapes", "#047857")
CajaHu = crear_caja_resultado(bottom_frame, 2, "Momentos Hu", "#be123c")

# ATAJOS TECLADO
ventana.bind("<x>", alternar_camara)
ventana.bind("<X>", alternar_camara)
ventana.bind("<z>", tomar_foto)
ventana.bind("<Z>", tomar_foto)
ventana.bind("<r>", recortar_imagen)
ventana.bind("<R>", recortar_imagen)
ventana.bind("<c>", umbralizar)
ventana.bind("<C>", umbralizar)


def al_cerrar():
    global camara_activa, capture
    camara_activa = False
    if capture:
        capture.release()
    ventana.destroy()


ventana.protocol("WM_DELETE_WINDOW", al_cerrar)
ventana.mainloop()

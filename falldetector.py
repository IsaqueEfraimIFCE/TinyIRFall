import serial  
import struct
import numpy as np
import matplotlib.pyplot as plt
import joblib
import cv2
import pygame
from tensorflow.keras.models import load_model
from scipy.ndimage import uniform_filter1d

# ================= CONFIGURAÇÕES =================
PORT = "COM4"
BAUD = 115200
FRAME_SIZE = 1544  # bytes por frame (MLX90640 24x32)

altura, largura = 24, 32
delta_body = 3.0

ambient_ref, body_ref = 17.54, 22.93

# ================= FUNÇÕES DE NORMALIZAÇÃO =================
def estimate_ambient(frame):
    return np.percentile(frame, 10)

def estimate_body_temp(frame, ambient):
    body_expected = ambient + delta_body
    body_measured = np.max(frame)
    return body_expected, body_measured

def normalize_frame(frame, ambient_ref, body_ref, ambient_new, body_new):
    frame_rel = frame - ambient_new
    delta_ref = body_ref - ambient_ref
    delta_new = body_new - ambient_new
    scale = delta_ref / delta_new if delta_new != 0 else 1.0
    frame_scaled = frame_rel * scale
    frame_final = frame_scaled + ambient_ref
    return frame_final

# ================= CONFIG MODELO =================
model_path = "./2.keras"
scaler_path = "./2.joblib"

fps_value = 4
frame_interval = 1.0 / fps_value
target_frames = 16

queda_limiar = 1.0         
prob_limiar = 0.5          
suavizacao_tamanho = 8     

# ================= FUNÇÃO DE LEITURA DO SENSOR =================
def read_sensor_generator(port=PORT, baud=BAUD):
    ser = serial.Serial(port, baud, timeout=1)
    while True:
        header = ser.read(2)
        if header != b'\x5A\x5A':
            continue
        frame_data = ser.read(FRAME_SIZE - 2)
        if len(frame_data) != FRAME_SIZE - 2:
            continue
        try:
            temps = struct.unpack('<768h', frame_data[:1536])
            temps = np.array(temps, dtype=np.float32) * 0.01
            frame = temps.reshape((altura, largura))
            yield frame
        except struct.error:
            continue

# ================= FUNÇÕES AUXILIARES =================
def calc_barycenter_y(frame, threshold=29):
    mask = frame > threshold
    y_coords, _ = np.where(mask)
    return np.median(y_coords) if len(y_coords) > 0 else np.nan

# ================= CARREGAR MODELO E SCALER =================
print("Carregando modelo e scaler...")
model = load_model(model_path)
scaler = joblib.load(scaler_path)

if hasattr(model, "input_shape"):
    in_shape = model.input_shape
    if len(in_shape) == 3:
        model_frames = in_shape[1]
        model_features = in_shape[2]
        expected_features = altura * largura
        if model_features != expected_features:
            raise ValueError(
                f"Model expects {model_features} features per frame but frame size is {expected_features}" )
        print(f"Model expects {model_frames} frames. Setting target_frames = {model_frames}")
        target_frames = model_frames
    else:
        raise ValueError(f"Unexpected model.input_shape: {in_shape}")

if hasattr(scaler, 'n_features_in_'):
    if scaler.n_features_in_ != altura * largura:
        raise ValueError(
            f"Scaler expects {scaler.n_features_in_} features but frame has {altura*largura} features.")

# ================= INTERFACE =================
scale = 30
colormap_min, colormap_max = 17.0, 33.0
temp_threshold = 29.0

colormaps = [
    ("JET", cv2.COLORMAP_JET),
    ("TURBO", cv2.COLORMAP_TURBO),
    ("HOT", cv2.COLORMAP_HOT),
    ("COOL", cv2.COLORMAP_COOL),
    ("BONE", cv2.COLORMAP_BONE),
]
cmap_index = 0

upsamples = [
    ("NEAREST", cv2.INTER_NEAREST),
    ("LINEAR", cv2.INTER_LINEAR),
    ("CUBIC", cv2.INTER_CUBIC),
    ("LANCZOS", cv2.INTER_LANCZOS4),
]
upsample_index = 2

pygame.init()
window_size = (largura * scale, altura * scale)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Sensor Térmico + Detecção de Quedas (Sensor Real)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 22, bold=True)

# Retângulos interativos
min_rect = pygame.Rect(10, 10, 90, 30)
max_rect = pygame.Rect(110, 10, 90, 30)
temp_rect = pygame.Rect(210, 10, 90, 30)
dy_rect = pygame.Rect(310, 10, 90, 30)
cmap_rect = pygame.Rect(410, 10, 140, 30)
upsample_rect = pygame.Rect(560, 10, 160, 30)
# novo botão Filtro (não alterar mais nada além do necessário)
filter_rect = pygame.Rect(730, 10, 100, 30)

# Estado de edição
min_text, min_active = "", False
max_text, max_active = "", False
temp_text, temp_active = "", False
dy_text, dy_active = "", False

color_inactive = (200, 200, 200)
color_active = (255, 255, 255)

# ================= VARIÁVEIS DE DETECÇÃO =================
frame_buffer = []
fall_counter = 0
fall_alert = False
prob_display = 0.0
bary_y_history = []
dy_dt = 0.0

# tempo em ms do último evento de queda detectada (para manter alerta por 5s)
last_fall_detected_time = None
ALERT_DURATION_MS = 5000  # 5 segundos

# Helper
def apply_text_to_value(text, current_value, value_type=float, min_val=None, max_val=None):
    if text is None or text == "":
        return current_value
    try:
        v = value_type(text)
        if min_val is not None and v < min_val:
            return current_value
        if max_val is not None and v > max_val:
            return current_value
        return v
    except Exception:
        return current_value

def confirm_inputs():
    global colormap_min, colormap_max, temp_threshold, queda_limiar
    colormap_min = apply_text_to_value(min_text, colormap_min, float, -50.0, 150.0)
    colormap_max = apply_text_to_value(max_text, colormap_max, float, -50.0, 300.0)
    if colormap_max <= colormap_min:
        colormap_max = colormap_min + 0.1
    temp_threshold = apply_text_to_value(temp_text, temp_threshold, float, -50.0, 300.0)
    queda_limiar = apply_text_to_value(dy_text, queda_limiar, float, 0.0, 100.0)

# ================= FILTRO TEMPORAL (LEVE) =================
# alpha é o peso da frame anterior; "leve" -> alpha pequeno (mais responsivo)
filter_on = False
filter_alpha = 0.7  # 0..1, quanto maior, mais suavização (aqui escolhemos 0.2 para leve suavização)
prev_filtered_frame = None

# ================= LOOP PRINCIPAL =================
running = True
sensor_gen = read_sensor_generator(PORT, BAUD)

while running:
    try:
        frame_temp = next(sensor_gen)
    except Exception as e:
        print("Erro leitura serial:", e)
        pygame.time.delay(100)
        continue

    # ---------- EVENTOS ----------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            if min_rect.collidepoint(mx, my):
                min_active = True
                max_active = temp_active = dy_active = False
                min_text = ""
            elif max_rect.collidepoint(mx, my):
                max_active = True
                min_active = temp_active = dy_active = False
                max_text = ""
            elif temp_rect.collidepoint(mx, my):
                temp_active = True
                min_active = max_active = dy_active = False
                temp_text = ""
            elif dy_rect.collidepoint(mx, my):
                dy_active = True
                min_active = max_active = temp_active = False
                dy_text = ""
            elif cmap_rect.collidepoint(mx, my):
                cmap_index = (cmap_index + 1) % len(colormaps)
            elif upsample_rect.collidepoint(mx, my):
                upsample_index = (upsample_index + 1) % len(upsamples)
            elif filter_rect.collidepoint(mx, my):
                # alterna estado do filtro
                filter_on = not filter_on
                # quando desliga, zera o histórico para evitar "fantasmas"
                if not filter_on:
                    prev_filtered_frame = None
            else:
                was_any_active = min_active or max_active or temp_active or dy_active
                min_active = max_active = temp_active = dy_active = False
                if was_any_active:
                    confirm_inputs()

        elif event.type == pygame.KEYDOWN:
            active_field = None
            if min_active: active_field = "min"
            elif max_active: active_field = "max"
            elif temp_active: active_field = "temp"
            elif dy_active: active_field = "dy"

            if active_field:
                if event.key == pygame.K_BACKSPACE:
                    if active_field == "min": min_text = min_text[:-1]
                    elif active_field == "max": max_text = max_text[:-1]
                    elif active_field == "temp": temp_text = temp_text[:-1]
                    elif active_field == "dy": dy_text = dy_text[:-1]
                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    confirm_inputs()
                    min_active = max_active = temp_active = dy_active = False
                else:
                    ch = event.unicode
                    if ch.isdigit() or ch in ".-":
                        if active_field == "min": min_text += ch
                        elif active_field == "max": max_text += ch
                        elif active_field == "temp": temp_text += ch
                        elif active_field == "dy": dy_text += ch

    # ---------- APLICAR FILTRO TEMPORAL LEVE (se ativado) ----------
    if filter_on:
        if prev_filtered_frame is None:
            prev_filtered_frame = frame_temp.copy()
        else:
            # EMA: prev = alpha * prev + (1-alpha) * current
            prev_filtered_frame = filter_alpha * prev_filtered_frame + (1.0 - filter_alpha) * frame_temp
        proc_frame = prev_filtered_frame
    else:
        proc_frame = frame_temp

    # ---------- BARICENTRO ----------
    bary_y = calc_barycenter_y(proc_frame, threshold=temp_threshold)
    bary_y_history.append(bary_y)
    if len(bary_y_history) > suavizacao_tamanho:
        valid_vals = np.array([y for y in bary_y_history if not np.isnan(y)])
        if len(valid_vals) > 1:
            bary_y_smoothed = uniform_filter1d(valid_vals, size=suavizacao_tamanho, mode="nearest")
            dy_dt = np.gradient(bary_y_smoothed, frame_interval)[-1]
        else:
            dy_dt = 0.0
    else:
        dy_dt = 0.0

    # ---------- DETECÇÃO ----------
    trigger_inference = dy_dt > queda_limiar
    frame_buffer.append(proc_frame)
    if len(frame_buffer) > target_frames:
        frame_buffer.pop(0)

    if trigger_inference and len(frame_buffer) == target_frames:
        normalized_frames = []
        for f in frame_buffer:
            ambient_new = estimate_ambient(f)
            _, body_measured_new = estimate_body_temp(f, ambient_new)
            f_norm = normalize_frame(f, ambient_ref, body_ref, ambient_new, body_measured_new)
            normalized_frames.append(f_norm)

        window = np.array(normalized_frames)
        window_flat = window.reshape(target_frames, altura * largura)
        window_scaled_flat = scaler.transform(window_flat)
        window_scaled = window_scaled_flat.reshape(1, target_frames, altura * largura)
        window_scaled = window_scaled.astype(np.float32)

        prob_display = float(model.predict(window_scaled, verbose=0)[0][0])
        fall_counter = fall_counter + 1 if prob_display >= prob_limiar else 0
        fall_alert = fall_counter >= target_frames
        # se detectou queda, atualiza timestamp do último evento detectado
        if fall_alert:
            last_fall_detected_time = pygame.time.get_ticks()
    else:
        prob_display = 0.0
        fall_alert = False

    # ---------- VISUALIZAÇÃO ----------
    frame_clipped = np.clip(proc_frame, colormap_min, colormap_max)
    frame_norm = ((frame_clipped - colormap_min) / (colormap_max - colormap_min) * 255).astype(np.uint8)
    frame_color = cv2.applyColorMap(frame_norm, colormaps[cmap_index][1])

    mask_hot = (proc_frame > temp_threshold).astype(np.uint8) * 255
    frame_up = cv2.resize(frame_color, window_size, interpolation=upsamples[upsample_index][1])
    mask_up = cv2.resize(mask_hot, window_size, interpolation=cv2.INTER_NEAREST)

    frame_up = cv2.GaussianBlur(frame_up, (5, 5), 0)
    contours, _ = cv2.findContours(mask_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame_up, contours, -1, (0, 0, 0), 2)

    frame_rgb = cv2.cvtColor(frame_up, cv2.COLOR_BGR2RGB)
    surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
    screen.blit(surface, (0, 0))

    # calcular temp_max normalmente; para temp_min, desconsidera os 5 menores valores
    temp_max = float(np.max(proc_frame))
    flat = proc_frame.flatten()
    if flat.size > 5:
        sorted_vals = np.sort(flat)
        # pega o mínimo ignorando os 5 menores valores
        temp_min = float(np.min(sorted_vals[5:])) if sorted_vals[5:].size > 0 else float(np.min(sorted_vals))
    else:
        temp_min = float(np.min(flat))

    screen.blit(font.render(f"Mín: {temp_min:.1f}°C | Máx: {temp_max:.1f}°C", True, (255, 255, 255)), (10, 60))
    screen.blit(font.render(f"Pred: {prob_display:.3f}", True, (255, 255, 0)), (10, 90))

    # exibe alerta se último evento de queda foi dentro dos últimos ALERT_DURATION_MS
    show_alert = False
    if last_fall_detected_time is not None:
        elapsed = pygame.time.get_ticks() - last_fall_detected_time
        if elapsed <= ALERT_DURATION_MS:
            show_alert = True
        else:
            # expira o tempo do alerta
            last_fall_detected_time = None

    if show_alert:
        screen.blit(font.render("QUEDA DETECTADA!", True, (255, 0, 0)), (10, 120))

    screen.blit(font.render(f"dy/dt: {dy_dt:.2f}", True, (0, 255, 255)), (10, 150))
    screen.blit(font.render(f"FPS: {fps_value}", True, (0, 200, 0)), (10, 180))

    # ---------- INPUT BOXES ----------
    pygame.draw.rect(screen, color_active if min_active else color_inactive, min_rect, 2)
    min_display = min_text if min_text != "" else f"{int(colormap_min)}"
    txt_surface = font.render(f"Min {min_display}", True, (255, 255, 255))
    screen.blit(txt_surface, (min_rect.x + 5, min_rect.y + 5))

    pygame.draw.rect(screen, color_active if max_active else color_inactive, max_rect, 2)
    max_display = max_text if max_text != "" else f"{int(colormap_max)}"
    txt_surface = font.render(f"Max {max_display}", True, (255, 255, 255))
    screen.blit(txt_surface, (max_rect.x + 5, max_rect.y + 5))

    pygame.draw.rect(screen, color_active if temp_active else color_inactive, temp_rect, 2)
    temp_display = temp_text if temp_text != "" else f"{int(temp_threshold)}"
    txt_surface = font.render(f"Temp {temp_display}", True, (255, 255, 255))
    screen.blit(txt_surface, (temp_rect.x + 5, temp_rect.y + 5))

    pygame.draw.rect(screen, color_active if dy_active else color_inactive, dy_rect, 2)
    dy_display = dy_text if dy_text != "" else f"{queda_limiar:.1f}"
    txt_surface = font.render(f"dY {dy_display}", True, (255, 255, 255))
    screen.blit(txt_surface, (dy_rect.x + 5, dy_rect.y + 5))

    # Colormap button
    pygame.draw.rect(screen, (100, 100, 200), cmap_rect)
    txt_surface = font.render(colormaps[cmap_index][0], True, (255, 255, 255))
    screen.blit(txt_surface, (cmap_rect.x + 5, cmap_rect.y + 5))

    # Upsample button
    pygame.draw.rect(screen, (100, 200, 100), upsample_rect)
    txt_surface = font.render(upsamples[upsample_index][0], True, (0, 0, 0))
    screen.blit(txt_surface, (upsample_rect.x + 5, upsample_rect.y + 5))

    # Botão Filtro: verde quando ON, vermelho quando OFF
    filter_color = (0, 200, 0) if filter_on else (200, 0, 0)
    pygame.draw.rect(screen, filter_color, filter_rect)
    txt_surface = font.render("Filtro", True, (255, 255, 255))
    screen.blit(txt_surface, (filter_rect.x + 12, filter_rect.y + 5))

    pygame.display.flip()
    clock.tick(fps_value)

pygame.quit()
  adicione uma segunda ligação serial com a com7, e ao detectar uma queda, envie o comando "on" e depois de 5 segundos, o comando "off"
Dito pelo ChatGPT:

Boa! 👌
Você já tem toda a lógica de detecção implementada. Para adicionar uma segunda porta serial (COM7) e enviar o comando "on" quando detectar queda e "off" após 5 s, basta:

Abrir uma segunda conexão serial.

No ponto onde você detecta queda, mandar "on\n".

Após 5 s, mandar "off\n" (isso pode ser feito aproveitando a variável last_fall_detected_time que você já usa para o alerta).

Segue o código adaptado (só os trechos que mudam, sem mexer no resto):

# ================= FUNÇÃO DE LEITURA DO SENSOR =================
def read_sensor_generator(port=PORT, baud=BAUD):
    ser = serial.Serial(port, baud, timeout=1)
    while True:
        header = ser.read(2)
        if header != b'\x5A\x5A':
            continue
        frame_data = ser.read(FRAME_SIZE - 2)
        if len(frame_data) != FRAME_SIZE - 2:
            continue
        try:
            temps = struct.unpack('<768h', frame_data[:1536])
            temps = np.array(temps, dtype=np.float32) * 0.01
            frame = temps.reshape((altura, largura))
            yield frame
        except struct.error:
            continue

# ================= SERIAL EXTRA (COM7) =================
try:
    ser_out = serial.Serial("COM7", BAUD, timeout=1)
    print("Ligação serial COM7 aberta para comandos de saída.")
except Exception as e:
    ser_out = None
    print("Erro ao abrir COM7:", e)


Agora no loop principal, dentro da parte de detecção de queda:

        if fall_alert:
            last_fall_detected_time = pygame.time.get_ticks()
            # Envia comando "on" ao detectar queda
            if ser_out is not None:
                ser_out.write(b"on\n")


E logo depois, na parte onde você já verifica o tempo do alerta (show_alert):

    if show_alert:
        screen.blit(font.render("QUEDA DETECTADA!", True, (255, 0, 0)), (10, 120))

        # Se já passaram 5s do alerta, envia "off"
        if elapsed >= ALERT_DURATION_MS and ser_out is not None:
            ser_out.write(b"off\n")


🔧 Resumindo:

COM4 continua para o sensor (entrada).

COM7 agora serve para saída de comandos.

Ao detectar queda → envia "on".

Depois de 5 s → envia "off".

Quer que eu te entregue o código inteiro já com essas modificações embutidas, em vez de só os trechos?

Dito por si:
o código inteiro exatamente como antes, apenas com as modificações necessárias para o comando "on" e "off"
Dito pelo ChatGPT:

Perfeito, vou te devolver o código completo, idêntico ao que você me passou, mas já com a segunda serial (COM7) e o envio dos comandos "on" e "off" após detectar uma queda.

Aqui está:

import serial  
import struct
import numpy as np
import matplotlib.pyplot as plt
import joblib
import cv2
import pygame
from tensorflow.keras.models import load_model
from scipy.ndimage import uniform_filter1d

# ================= CONFIGURAÇÕES =================
PORT = "COM4"
BAUD = 115200
FRAME_SIZE = 1544  # bytes por frame (MLX90640 24x32)

altura, largura = 24, 32
delta_body = 3.0

ambient_ref, body_ref = 17.54, 22.93

# ================= FUNÇÕES DE NORMALIZAÇÃO =================
def estimate_ambient(frame):
    return np.percentile(frame, 10)

def estimate_body_temp(frame, ambient):
    body_expected = ambient + delta_body
    body_measured = np.max(frame)
    return body_expected, body_measured

def normalize_frame(frame, ambient_ref, body_ref, ambient_new, body_new):
    frame_rel = frame - ambient_new
    delta_ref = body_ref - ambient_ref
    delta_new = body_new - ambient_new
    scale = delta_ref / delta_new if delta_new != 0 else 1.0
    frame_scaled = frame_rel * scale
    frame_final = frame_scaled + ambient_ref
    return frame_final

# ================= CONFIG MODELO =================
model_path = "./2.keras"
scaler_path = "./2.joblib"

fps_value = 4
frame_interval = 1.0 / fps_value
target_frames = 16

queda_limiar = 1.0         
prob_limiar = 0.5          
suavizacao_tamanho = 8     

# ================= FUNÇÃO DE LEITURA DO SENSOR =================
def read_sensor_generator(port=PORT, baud=BAUD):
    ser = serial.Serial(port, baud, timeout=1)
    while True:
        header = ser.read(2)
        if header != b'\x5A\x5A':
            continue
        frame_data = ser.read(FRAME_SIZE - 2)
        if len(frame_data) != FRAME_SIZE - 2:
            continue
        try:
            temps = struct.unpack('<768h', frame_data[:1536])
            temps = np.array(temps, dtype=np.float32) * 0.01
            frame = temps.reshape((altura, largura))
            yield frame
        except struct.error:
            continue

# ================= SERIAL EXTRA (COM7) =================
try:
    ser_out = serial.Serial("COM7", BAUD, timeout=1)
    print("Ligação serial COM7 aberta para comandos de saída.")
except Exception as e:
    ser_out = None
    print("Erro ao abrir COM7:", e)

# ================= FUNÇÕES AUXILIARES =================
def calc_barycenter_y(frame, threshold=29):
    mask = frame > threshold
    y_coords, _ = np.where(mask)
    return np.median(y_coords) if len(y_coords) > 0 else np.nan

# ================= CARREGAR MODELO E SCALER =================
print("Carregando modelo e scaler...")
model = load_model(model_path)
scaler = joblib.load(scaler_path)

if hasattr(model, "input_shape"):
    in_shape = model.input_shape
    if len(in_shape) == 3:
        model_frames = in_shape[1]
        model_features = in_shape[2]
        expected_features = altura * largura
        if model_features != expected_features:
            raise ValueError(
                f"Model expects {model_features} features per frame but frame size is {expected_features}" )
        print(f"Model expects {model_frames} frames. Setting target_frames = {model_frames}")
        target_frames = model_frames
    else:
        raise ValueError(f"Unexpected model.input_shape: {in_shape}")

if hasattr(scaler, 'n_features_in_'):
    if scaler.n_features_in_ != altura * largura:
        raise ValueError(
            f"Scaler expects {scaler.n_features_in_} features but frame has {altura*largura} features.")

# ================= INTERFACE =================
scale = 30
colormap_min, colormap_max = 17.0, 33.0
temp_threshold = 29.0

colormaps = [
    ("JET", cv2.COLORMAP_JET),
    ("TURBO", cv2.COLORMAP_TURBO),
    ("HOT", cv2.COLORMAP_HOT),
    ("COOL", cv2.COLORMAP_COOL),
    ("BONE", cv2.COLORMAP_BONE),
]
cmap_index = 0

upsamples = [
    ("NEAREST", cv2.INTER_NEAREST),
    ("LINEAR", cv2.INTER_LINEAR),
    ("CUBIC", cv2.INTER_CUBIC),
    ("LANCZOS", cv2.INTER_LANCZOS4),
]
upsample_index = 2

pygame.init()
window_size = (largura * scale, altura * scale)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Sensor Térmico + Detecção de Quedas (Sensor Real)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 22, bold=True)

# Retângulos interativos
min_rect = pygame.Rect(10, 10, 90, 30)
max_rect = pygame.Rect(110, 10, 90, 30)
temp_rect = pygame.Rect(210, 10, 90, 30)
dy_rect = pygame.Rect(310, 10, 90, 30)
cmap_rect = pygame.Rect(410, 10, 140, 30)
upsample_rect = pygame.Rect(560, 10, 160, 30)
filter_rect = pygame.Rect(730, 10, 100, 30)

# Estado de edição
min_text, min_active = "", False
max_text, max_active = "", False
temp_text, temp_active = "", False
dy_text, dy_active = "", False

color_inactive = (200, 200, 200)
color_active = (255, 255, 255)

# ================= VARIÁVEIS DE DETECÇÃO =================
frame_buffer = []
fall_counter = 0
fall_alert = False
prob_display = 0.0
bary_y_history = []
dy_dt = 0.0

last_fall_detected_time = None
ALERT_DURATION_MS = 5000  # 5 segundos

# Helper
def apply_text_to_value(text, current_value, value_type=float, min_val=None, max_val=None):
    if text is None or text == "":
        return current_value
    try:
        v = value_type(text)
        if min_val is not None and v < min_val:
            return current_value
        if max_val is not None and v > max_val:
            return current_value
        return v
    except Exception:
        return current_value

def confirm_inputs():
    global colormap_min, colormap_max, temp_threshold, queda_limiar
    colormap_min = apply_text_to_value(min_text, colormap_min, float, -50.0, 150.0)
    colormap_max = apply_text_to_value(max_text, colormap_max, float, -50.0, 300.0)
    if colormap_max <= colormap_min:
        colormap_max = colormap_min + 0.1
    temp_threshold = apply_text_to_value(temp_text, temp_threshold, float, -50.0, 300.0)
    queda_limiar = apply_text_to_value(dy_text, queda_limiar, float, 0.0, 100.0)

# ================= FILTRO TEMPORAL (LEVE) =================
filter_on = False
filter_alpha = 0.7
prev_filtered_frame = None

# ================= LOOP PRINCIPAL =================
running = True
sensor_gen = read_sensor_generator(PORT, BAUD)

while running:
    try:
        frame_temp = next(sensor_gen)
    except Exception as e:
        print("Erro leitura serial:", e)
        pygame.time.delay(100)
        continue

    # ---------- EVENTOS ----------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            if min_rect.collidepoint(mx, my):
                min_active = True
                max_active = temp_active = dy_active = False
                min_text = ""
            elif max_rect.collidepoint(mx, my):
                max_active = True
                min_active = temp_active = dy_active = False
                max_text = ""
            elif temp_rect.collidepoint(mx, my):
                temp_active = True
                min_active = max_active = dy_active = False
                temp_text = ""
            elif dy_rect.collidepoint(mx, my):
                dy_active = True
                min_active = max_active = temp_active = False
                dy_text = ""
            elif cmap_rect.collidepoint(mx, my):
                cmap_index = (cmap_index + 1) % len(colormaps)
            elif upsample_rect.collidepoint(mx, my):
                upsample_index = (upsample_index + 1) % len(upsamples)
            elif filter_rect.collidepoint(mx, my):
                filter_on = not filter_on
                if not filter_on:
                    prev_filtered_frame = None
            else:
                was_any_active = min_active or max_active or temp_active or dy_active
                min_active = max_active = temp_active = dy_active = False
                if was_any_active:
                    confirm_inputs()

        elif event.type == pygame.KEYDOWN:
            active_field = None
            if min_active: active_field = "min"
            elif max_active: active_field = "max"
            elif temp_active: active_field = "temp"
            elif dy_active: active_field = "dy"

            if active_field:
                if event.key == pygame.K_BACKSPACE:
                    if active_field == "min": min_text = min_text[:-1]
                    elif active_field == "max": max_text = max_text[:-1]
                    elif active_field == "temp": temp_text = temp_text[:-1]
                    elif active_field == "dy": dy_text = dy_text[:-1]
                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    confirm_inputs()
                    min_active = max_active = temp_active = dy_active = False
                else:
                    ch = event.unicode
                    if ch.isdigit() or ch in ".-":
                        if active_field == "min": min_text += ch
                        elif active_field == "max": max_text += ch
                        elif active_field == "temp": temp_text += ch
                        elif active_field == "dy": dy_text += ch

    # ---------- APLICAR FILTRO ----------
    if filter_on:
        if prev_filtered_frame is None:
            prev_filtered_frame = frame_temp.copy()
        else:
            prev_filtered_frame = filter_alpha * prev_filtered_frame + (1.0 - filter_alpha) * frame_temp
        proc_frame = prev_filtered_frame
    else:
        proc_frame = frame_temp

    # ---------- BARICENTRO ----------
    bary_y = calc_barycenter_y(proc_frame, threshold=temp_threshold)
    bary_y_history.append(bary_y)
    if len(bary_y_history) > suavizacao_tamanho:
        valid_vals = np.array([y for y in bary_y_history if not np.isnan(y)])
        if len(valid_vals) > 1:
            bary_y_smoothed = uniform_filter1d(valid_vals, size=suavizacao_tamanho, mode="nearest")
            dy_dt = np.gradient(bary_y_smoothed, frame_interval)[-1]
        else:
            dy_dt = 0.0
    else:
        dy_dt = 0.0

    # ---------- DETECÇÃO ----------
    trigger_inference = dy_dt > queda_limiar
    frame_buffer.append(proc_frame)
    if len(frame_buffer) > target_frames:
        frame_buffer.pop(0)

    if trigger_inference and len(frame_buffer) == target_frames:
        normalized_frames = []
        for f in frame_buffer:
            ambient_new = estimate_ambient(f)
            _, body_measured_new = estimate_body_temp(f, ambient_new)
            f_norm = normalize_frame(f, ambient_ref, body_ref, ambient_new, body_measured_new)
            normalized_frames.append(f_norm)

        window = np.array(normalized_frames)
        window_flat = window.reshape(target_frames, altura * largura)
        window_scaled_flat = scaler.transform(window_flat)
        window_scaled = window_scaled_flat.reshape(1, target_frames, altura * largura)
        window_scaled = window_scaled.astype(np.float32)

        prob_display = float(model.predict(window_scaled, verbose=0)[0][0])
        fall_counter = fall_counter + 1 if prob_display >= prob_limiar else 0
        fall_alert = fall_counter >= target_frames
        if fall_alert:
            last_fall_detected_time = pygame.time.get_ticks()
            # envia "on" quando detecta queda
            if ser_out is not None:
                ser_out.write(b"on\n")
    else:
        prob_display = 0.0
        fall_alert = False

    # ---------- VISUALIZAÇÃO ----------
    frame_clipped = np.clip(proc_frame, colormap_min, colormap_max)
    frame_norm = ((frame_clipped - colormap_min) / (colormap_max - colormap_min) * 255).astype(np.uint8)
    frame_color = cv2.applyColorMap(frame_norm, colormaps[cmap_index][1])

    mask_hot = (proc_frame > temp_threshold).astype(np.uint8) * 255
    frame_up = cv2.resize(frame_color, window_size, interpolation=upsamples[upsample_index][1])
    mask_up = cv2.resize(mask_hot, window_size, interpolation=cv2.INTER_NEAREST)

    frame_up = cv2.GaussianBlur(frame_up, (5, 5), 0)
    contours, _ = cv2.findContours(mask_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame_up, contours, -1, (0, 0, 0), 2)

    frame_rgb = cv2.cvtColor(frame_up, cv2.COLOR_BGR2RGB)
    surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
    screen.blit(surface, (0, 0))

    temp_max = float(np.max(proc_frame))
    flat = proc_frame.flatten()
    if flat.size > 5:
        sorted_vals = np.sort(flat)
        temp_min = float(np.min(sorted_vals[5:])) if sorted_vals[5:].size > 0 else float(np.min(sorted_vals))
    else:
        temp_min = float(np.min(flat))

    screen.blit(font.render(f"Mín: {temp_min:.1f}°C | Máx: {temp_max:.1f}°C", True, (255, 255, 255)), (10, 60))
    screen.blit(font.render(f"Pred: {prob_display:.3f}", True, (255, 255, 0)), (10, 90))

    show_alert = False
    if last_fall_detected_time is not None:
        elapsed = pygame.time.get_ticks() - last_fall_detected_time
        if elapsed <= ALERT_DURATION_MS:
            show_alert = True
        else:
            # expira alerta e envia "off"
            if ser_out is not None:
                ser_out.write(b"off\n")
            last_fall_detected_time = None

    if show_alert:
        screen.blit(font.render("QUEDA DETECTADA!", True, (255, 0, 0)), (10, 120))

    screen.blit(font.render(f"dy/dt: {dy_dt:.2f}", True, (0, 255, 255)), (10, 150))
    screen.blit(font.render(f"FPS: {fps_value}", True, (0, 200, 0)), (10, 180))

    # ---------- INPUT BOXES ----------
    pygame.draw.rect(screen, color_active if min_active else color_inactive, min_rect, 2)
    min_display = min_text if min_text != "" else f"{int(colormap_min)}"
    txt_surface = font.render(f"Min {min_display}", True, (255, 255, 255))
    screen.blit(txt_surface, (min_rect.x + 5, min_rect.y + 5))

    pygame.draw.rect(screen, color_active if max_active else color_inactive, max_rect, 2)
    max_display = max_text if max_text != "" else f"{int(colormap_max)}"
    txt_surface = font.render(f"Max {max_display}", True, (255, 255, 255))
    screen.blit(txt_surface, (max_rect.x + 5, max_rect.y + 5))

    pygame.draw.rect(screen, color_active if temp_active else color_inactive, temp_rect, 2)
    temp_display = temp_text if temp_text != "" else f"{int(temp_threshold)}"
    txt_surface = font.render(f"Temp {temp_display}", True, (255, 255, 255))
    screen.blit(txt_surface, (temp_rect.x + 5, temp_rect.y + 5))

    pygame.draw.rect(screen, color_active if dy_active else color_inactive, dy_rect, 2)
    dy_display = dy_text if dy_text != "" else f"{queda_limiar:.1f}"
    txt_surface = font.render(f"dY {dy_display}", True, (255, 255, 255))
    screen.blit(txt_surface, (dy_rect.x + 5, dy_rect.y + 5))

    pygame.draw.rect(screen, (100, 100, 200), cmap_rect)
    txt_surface = font.render(colormaps[cmap_index][0], True, (255, 255, 255))
    screen.blit(txt_surface, (cmap_rect.x + 5, cmap_rect.y + 5))

    pygame.draw.rect(screen, (100, 200, 100), upsample_rect)
    txt_surface = font.render(upsamples[upsample_index][0], True, (0, 0, 0))
    screen.blit(txt_surface, (upsample_rect.x + 5, upsample_rect.y + 5))

    filter_color = (0, 200, 0) if filter_on else (200, 0, 0)
    pygame.draw.rect(screen, filter_color, filter_rect)
    txt_surface = font.render("Filtro", True, (255, 255, 255))
    screen.blit(txt_surface, (filter_rect.x + 12, filter_rect.y + 5))

    pygame.display.flip()
    clock.tick(fps_value)

pygame.quit()

import os
import math
import datetime
import urllib.request
import gzip
import shutil
import ssl
import json
import threading
import re
import functools
from flask import Flask, request, send_file, Response
import tempfile

app = Flask(__name__)

# =====================================================================
# CONFIGURACIÓN DEL ENTORNO Y GESTIÓN DE ESTADO (AISLAMIENTO EN RAM)
# =====================================================================
BASE_DIR = tempfile.gettempdir()
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'temp_rinex')
REPORT_FOLDER = os.path.join(BASE_DIR, 'informes')
STATE_FILE = os.path.join(UPLOAD_FOLDER, 'estado_proyecto.json')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

STATE_LOCK = threading.Lock()
SP3_LOCK = threading.Lock() 

# --- CONSTANTES GEODÉSICAS INMUTABLES ---
C_LIGHT = 299792458.0
OMEGA_E = 7.2921151467e-5
MU = 3.986005e14
FREQ_L1 = 1575.42e6
FREQ_L5 = 1176.45e6
WAVE_L1 = C_LIGHT / FREQ_L1
WAVE_L5 = C_LIGHT / FREQ_L5

def f_14(val):
    if val is None: return "0.0"
    s = f"{val:.14f}"
    if '.' in s:
        s = s.rstrip('0')
        if s.endswith('.'): s += '0'
    return s

def safe_f(val, default=0.0):
    try: return float(val) if val and str(val).strip() != '' else default
    except: return default

def safe_i(val, default=19):
    try: return int(val) if val and str(val).strip() != '' else default
    except: return default

def guardar_estado(clave, valor):
    with STATE_LOCK:
        estado = {}
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r', encoding='utf-8') as f: estado = json.load(f)
            except: pass
        estado[clave] = valor
        try:
            with open(STATE_FILE, 'w', encoding='utf-8') as f: json.dump(estado, f)
        except: pass

def leer_estado(clave):
    with STATE_LOCK:
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r', encoding='utf-8') as f: return json.load(f).get(clave)
            except: pass
        return None

def gps_time_to_tow(year, month, day, hour, minute, second):
    sec_int, sec_frac = int(second), second - int(second)
    total = (datetime.datetime(year, month, day, hour, minute, sec_int) - datetime.datetime(1980, 1, 6)).total_seconds() + sec_frac
    return total - (int(total // 604800) * 604800)

# =====================================================================
# INTEGRACIÓN E/S GOOGLE DRIVE (TOLERANCIA A FALLOS SSL)
# =====================================================================
def descargar_desde_gdrive(url, filepath):
    match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    if not match:
        match = re.search(r'id=([a-zA-Z0-9_-]+)', url)
    if not match:
        raise ValueError("URL de Google Drive no reconocida o malformada.")
    
    file_id = match.group(1)
    direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    req = urllib.request.Request(direct_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, context=ctx, timeout=120) as response, open(filepath, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    
    return True

# =====================================================================
# ÁLGEBRA LINEAL DE ESTADO SÓLIDO (INDEPENDIENTE DE MÓDULOS)
# =====================================================================
def transpose_matrix(M):
    if not M or not M[0]: return []
    try: return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]
    except IndexError: return []

def matmul(A, B):
    if not A or not B or not A[0] or not B[0]: return []
    try:
        result = [[0.0 for _ in range(len(B[0]))] for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    result[i][j] += A[i][k] * B[k][j]
        return result
    except IndexError: return []

def matadd(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def matsub(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def matid(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def cholesky_decompose(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            sum1 = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - sum1
                if val <= 0: raise ValueError("Matriz no definida positiva")
                L[i][j] = math.sqrt(val)
            else:
                L[i][j] = (A[i][j] - sum1) / L[j][j]
    return L

def invert_lower_triangular(L):
    n = len(L)
    inv = [[0.0] * n for _ in range(n)]
    for i in range(n):
        inv[i][i] = 1.0 / L[i][i]
        for j in range(i):
            sum1 = sum(L[i][k] * inv[k][j] for k in range(j, i))
            inv[i][j] = -sum1 / L[i][i]
    return inv

def gauss_jordan_inverse(M):
    n = len(M)
    A = [[float(M[i][j]) for j in range(n)] for i in range(n)]
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        max_k = i
        for k in range(i + 1, n):
            if abs(A[k][i]) > abs(A[max_k][i]): max_k = k
        if max_k != i:
            A[i], A[max_k] = A[max_k], A[i]
            I[i], I[max_k] = I[max_k], I[i]
        pivot = A[i][i]
        if abs(pivot) < 1e-15: return None 
        for j in range(n):
            A[i][j] /= pivot
            I[i][j] /= pivot
        for k in range(n):
            if k == i: continue
            factor = A[k][i]
            for j in range(n):
                A[k][j] -= factor * A[i][j]
                I[k][j] -= factor * I[i][j]
    return I

def invert_matrix_nxn(M):
    if not M or not M[0]: return None
    try:
        L = cholesky_decompose(M)
        L_inv = invert_lower_triangular(L)
        return matmul(transpose_matrix(L_inv), L_inv)
    except:
        return gauss_jordan_inverse(M)

# =====================================================================
# PARSERS Y GESTIÓN DE ARCHIVOS
# =====================================================================
def parse_rinex_obs_completo(path):
    obs = {}
    sys_idx = {}
    sys_tokens = {}
    last_sys_char = None
    
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        in_h = True
        tow = None
        for line in f:
            if in_h:
                if "SYS / # / OBS TYPES" in line:
                    sys_char = line[0].strip()
                    if sys_char: last_sys_char = sys_char
                    if last_sys_char:
                        tokens = [x.strip() for x in line[6:60].split() if x.strip()]
                        if last_sys_char not in sys_tokens: sys_tokens[last_sys_char] = []
                        sys_tokens[last_sys_char].extend(tokens)
                elif "END OF HEADER" in line: 
                    in_h = False
                    for sc, t in sys_tokens.items():
                        sys_idx[sc] = {
                            'C1': next((i for i, x in enumerate(t) if x.startswith('C1')), -1),
                            'L1': next((i for i, x in enumerate(t) if x.startswith('L1')), -1),
                            'C5': next((i for i, x in enumerate(t) if x.startswith('C5')), -1),
                            'L5': next((i for i, x in enumerate(t) if x.startswith('L5')), -1),
                            'S1': next((i for i, x in enumerate(t) if x.startswith('S1')), -1),
                            'S5': next((i for i, x in enumerate(t) if x.startswith('S5')), -1)
                        }
            elif line.startswith('>'):
                p = line[1:].split()
                if len(p) >= 6:
                    y, m, d, h, mn, sec = int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4]), float(p[5])
                    tow = round(gps_time_to_tow(y, m, d, h, mn, sec), 6)
                    obs[tow] = {'_meta': (y, m, d, h, mn, sec)}
            elif tow and len(line) > 3 and line[0] in 'GRECSJ':
                sys_char = line[0]
                idx_c1 = sys_idx.get(sys_char, {}).get('C1', -1)
                idx_l1 = sys_idx.get(sys_char, {}).get('L1', -1)
                idx_c5 = sys_idx.get(sys_char, {}).get('C5', -1)
                idx_l5 = sys_idx.get(sys_char, {}).get('L5', -1)
                idx_s1 = sys_idx.get(sys_char, {}).get('S1', -1)
                idx_s5 = sys_idx.get(sys_char, {}).get('S5', -1)
                
                data = {}
                if idx_c1 >= 0 and len(line) >= 17 + 16 * idx_c1:
                    v = line[3+16*idx_c1 : 17+16*idx_c1].strip()
                    if v: data['C1'] = float(v.replace('D', 'E').replace('d', 'e'))
                if idx_c5 >= 0 and len(line) >= 17 + 16 * idx_c5:
                    v = line[3+16*idx_c5 : 17+16*idx_c5].strip()
                    if v: data['C5'] = float(v.replace('D', 'E').replace('d', 'e'))
                if idx_l1 >= 0 and len(line) >= 17 + 16 * idx_l1:
                    v = line[3+16*idx_l1 : 17+16*idx_l1].strip()
                    if v: data['L1'] = float(v.replace('D', 'E').replace('d', 'e'))
                if idx_l5 >= 0 and len(line) >= 17 + 16 * idx_l5:
                    v = line[3+16*idx_l5 : 17+16*idx_l5].strip()
                    if v: data['L5'] = float(v.replace('D', 'E').replace('d', 'e'))
                if idx_s1 >= 0 and len(line) >= 17 + 16 * idx_s1:
                    v = line[3+16*idx_s1 : 17+16*idx_s1].strip()
                    if v: data['S1'] = float(v.replace('D', 'E').replace('d', 'e'))
                if idx_s5 >= 0 and len(line) >= 17 + 16 * idx_s5:
                    v = line[3+16*idx_s5 : 17+16*idx_s5].strip()
                    if v: data['S5'] = float(v.replace('D', 'E').replace('d', 'e'))
                
                valid_p = ('C1' in data and data['C1'] > 15000000.0) or ('C5' in data and data['C5'] > 15000000.0)
                if valid_p:
                    if tow not in obs: obs[tow] = {}
                    obs[tow][line[0:3].strip()] = data
    return obs

def interpolar_base_a_rover(obs_base, tr, max_gap=0.05):
    tiempos_base = sorted(list(obs_base.keys()))
    if not tiempos_base: return None
    idx = min(range(len(tiempos_base)), key=lambda i: abs(tiempos_base[i] - tr))
    if abs(tiempos_base[idx] - tr) <= max_gap:
        return obs_base[tiempos_base[idx]].copy()
    return None

def generar_rinex_sincronizado(raw_path, out_path, obs_dict):
    header_lines = []
    with open(raw_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if "SYS / # / OBS TYPES" in line: continue 
            header_lines.append(line)
            if "END OF HEADER" in line: break
    
    idx = -1
    for i, l in enumerate(header_lines):
        if "END OF HEADER" in l:
            idx = i
            break
            
    if idx != -1:
        constelaciones_requeridas = ['G', 'E', 'C', 'R', 'S', 'J']
        offset = 0
        for c in constelaciones_requeridas:
            header_lines.insert(idx + offset, f"{c}    4 C1 L1 C5 L5                                       SYS / # / OBS TYPES\n")
            offset += 1
            
    with open(out_path, 'w', encoding='utf-8') as f_out:
        for line in header_lines: f_out.write(line)
        for tow in sorted(obs_dict.keys()):
            meta = obs_dict[tow].get('_meta')
            if not meta: continue
            y, m, d, h, mn, sec = meta[0], meta[1], meta[2], meta[3], meta[4], meta[5]
            sats = [k for k in obs_dict[tow].keys() if k != '_meta']
            f_out.write(f"> {y} {m:02d} {d:02d} {h:02d} {mn:02d} {sec:11.7f}  0 {len(sats):2d}\n")
            
            for sat in sats:
                c1 = obs_dict[tow][sat].get('C1', 0.0)
                l1 = obs_dict[tow][sat].get('L1', 0.0)
                c5 = obs_dict[tow][sat].get('C5', 0.0)
                l5 = obs_dict[tow][sat].get('L5', 0.0)
                c1_s = f"{c1:14.3f}" if c1 != 0.0 else "              "
                l1_s = f"{l1:14.3f}" if l1 != 0.0 else "              "
                c5_s = f"{c5:14.3f}" if c5 != 0.0 else "              "
                l5_s = f"{l5:14.3f}" if l5 != 0.0 else "              "
                f_out.write(f"{sat}{c1_s}  {l1_s}  {c5_s}  {l5_s}  \n")

def obtener_fecha_obs(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('>'):
                partes = line[1:].strip().split()
                if len(partes) >= 6: 
                    try:
                        y = int(partes[0])
                        year = y if y > 100 else y + 2000
                        return year, int(partes[1]), int(partes[2]), int(partes[3]), int(partes[4]), float(partes[5])
                    except: pass
    return None

# =====================================================================
# PRODUCTOS IGS Y EFEMÉRIDES (HÍBRIDO NAV / SP3)
# =====================================================================
SP3_CACHE = {}
SP3_CACHE_KEYS = []
MAX_CACHE_SIZE = 2048

def parse_sp3_preciso(path):
    sp3_data = {}
    if not path or not os.path.exists(path): return sp3_data
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        current_time = None
        for line in f:
            if line.startswith('* '):
                p = line.split()
                if len(p) >= 7:
                    try:
                        y, m, d, h, mn, s = int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5]), float(p[6])
                        current_time = gps_time_to_tow(y, m, d, h, mn, s)
                    except: pass
            elif line.startswith('P') and current_time:
                sys_char = line[1]
                if sys_char in 'GECR':
                    sat_id = line[1:4].strip()
                    try:
                        x = float(line[4:18]) * 1000.0
                        y = float(line[18:32]) * 1000.0
                        z = float(line[32:46]) * 1000.0
                        clk = float(line[46:60]) / 1e6 if len(line) > 46 and line[46:60].strip() else 0.0
                        if sat_id not in sp3_data: sp3_data[sat_id] = []
                        sp3_data[sat_id].append((current_time, x, y, z, clk))
                    except: pass
    for sat in sp3_data: sp3_data[sat].sort(key=lambda item: item[0])
    return sp3_data

def lagrange_interpolate(x, x_pts, y_pts):
    n = len(x_pts)
    val = 0.0
    for i in range(n):
        p = 1.0
        for j in range(n):
            if i != j: p *= (x - x_pts[j]) / (x_pts[i] - x_pts[j])
        val += y_pts[i] * p
    return val

def interpolate_sp3(sp3_data, sat, t_emision, degree=9):
    global SP3_CACHE, SP3_CACHE_KEYS
    cache_key = f"{sat}_{t_emision}"
    with SP3_LOCK:
        if cache_key in SP3_CACHE: return SP3_CACHE[cache_key]

    if sat not in sp3_data: return None
    data = sp3_data[sat]
    if len(data) < degree + 1: return None
    
    idx = min(range(len(data)), key=lambda i: abs(data[i][0] - t_emision))
    half = degree // 2
    start = max(0, idx - half)
    end = min(len(data), start + degree + 1)
    if end - start < degree + 1: start = max(0, end - degree - 1)
        
    pts = data[start:end]
    t_pts, x_pts, y_pts, z_pts = [], [], [], []
    for p in pts:
        t_pts.append(p[0]); x_pts.append(p[1]); y_pts.append(p[2]); z_pts.append(p[3])
    
    start_clk = max(0, idx - 1)
    end_clk = min(len(data), start_clk + 2)
    if end_clk - start_clk < 2: start_clk = max(0, end_clk - 2)
    pts_clk = data[start_clk:end_clk]
    t_pts_clk, clk_pts = [], []
    for p in pts_clk:
        t_pts_clk.append(p[0]); clk_pts.append(p[4])
    
    val_x = lagrange_interpolate(t_emision, t_pts, x_pts)
    val_y = lagrange_interpolate(t_emision, t_pts, y_pts)
    val_z = lagrange_interpolate(t_emision, t_pts, z_pts)
    val_clk = lagrange_interpolate(t_emision, t_pts_clk, clk_pts)
    result = (val_x, val_y, val_z, val_clk)
    
    with SP3_LOCK:
        if len(SP3_CACHE) >= MAX_CACHE_SIZE:
            oldest_key = SP3_CACHE_KEYS.pop(0)
            SP3_CACHE.pop(oldest_key, None)
        SP3_CACHE[cache_key] = result
        SP3_CACHE_KEYS.append(cache_key)
    return result

def parse_rinex_nav_real(path):
    ephemeris = {'_iono': {'alpha': [0]*4, 'beta': [0]*4}}
    if not path or not os.path.exists(path): return ephemeris
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        in_h, sat, data = True, None, []
        for line in f:
            if in_h:
                if "IONOSPHERIC CORR" in line:
                    sys_type = line[0:4].strip()
                    vals = []
                    for i in range(4):
                        try:
                            chunk = line[5+i*12 : 5+(i+1)*12].strip().replace('D', 'E').replace('d', 'e')
                            vals.append(float(chunk) if chunk else 0.0)
                        except: vals.append(0.0)
                    if sys_type == 'GPSA': ephemeris['_iono']['alpha'] = vals
                    elif sys_type == 'GPSB': ephemeris['_iono']['beta'] = vals
                elif "END OF HEADER" in line: in_h = False
                continue
            if len(line) > 8 and line[0] in 'GECSJ' and line[1:3].isdigit():
                if sat and len(data) >= 20: 
                    ephemeris.setdefault(sat, []).append({'af0':data[0],'af1':data[1],'af2':data[2],'Crs':data[4],'Delta_n':data[5],'M0':data[6],'Cuc':data[7],'e':data[8],'Cus':data[9],'sqrtA':data[10],'Toe':data[11],'Cic':data[12],'OMEGA':data[13],'Cis':data[14],'i0':data[15],'Crc':data[16],'omega':data[17],'OMEGA_DOT':data[18],'IDOT':data[19]})
                sat = line[0:3].strip()
                data = [float(line[23:42].replace('D','E').replace('d','e')), float(line[42:61].replace('D','E').replace('d','e')), float(line[61:80].replace('D','E').replace('d','e'))]
            elif sat and line.startswith('    '): 
                data.extend([float(line[i:i+19].replace('D','E').replace('d','e').strip()) for i in range(4, 80, 19) if line[i:i+19].strip()])
        if sat and len(data) >= 20: 
            ephemeris.setdefault(sat, []).append({'af0':data[0],'af1':data[1],'af2':data[2],'Crs':data[4],'Delta_n':data[5],'M0':data[6],'Cuc':data[7],'e':data[8],'Cus':data[9],'sqrtA':data[10],'Toe':data[11],'Cic':data[12],'OMEGA':data[13],'Cis':data[14],'i0':data[15],'Crc':data[16],'omega':data[17],'OMEGA_DOT':data[18],'IDOT':data[19]})
    return ephemeris

def seleccionar_efemeride_optima(eph_list, t_target):
    if not eph_list: return None
    valid_ephs = []
    for eph in eph_list:
        dt = t_target - eph.get('Toe', 0)
        if dt > 302400: dt -= 604800
        elif dt < -302400: dt += 604800
        if abs(dt) <= 7200:
            valid_ephs.append((abs(dt), eph))
    if not valid_ephs: return None
    return min(valid_ephs, key=lambda x: x[0])[1]

# =====================================================================
# GEODESIA ESPACIAL Y CORRECCIONES ATMOSFÉRICAS
# =====================================================================
def correccion_mareas_solidas(X, Y, Z, tow, year, month, day):
    try:
        h2, l2 = 0.609, 0.085
        Re = 6378137.0
        GM_earth, GM_sun, GM_moon = 3.986004418e14, 1.327124e20, 4.902801e12
        
        jd = 367 * year - (7 * (year + (month + 9) // 12)) // 4 + (275 * month) // 9 + day + 1721013.5
        t_jc = (jd - 2451545.0 + (tow / 86400.0)) / 36525.0
        
        mean_long_sun = 280.460 + 36000.771 * t_jc
        mean_anom_sun = 357.528 + 35999.050 * t_jc
        ecl_lon_sun = mean_long_sun + 1.915 * math.sin(math.radians(mean_anom_sun)) + 0.020 * math.sin(math.radians(2 * mean_anom_sun))
        dist_sun = 1.495978707e11 * (1.00014 - 0.01671 * math.cos(math.radians(mean_anom_sun)) - 0.00014 * math.cos(math.radians(2 * mean_anom_sun)))
        obliquity = 23.439 - 0.013 * t_jc
        
        xs_sun = dist_sun * math.cos(math.radians(ecl_lon_sun))
        ys_sun = dist_sun * math.cos(math.radians(obliquity)) * math.sin(math.radians(ecl_lon_sun))
        zs_sun = dist_sun * math.sin(math.radians(obliquity)) * math.sin(math.radians(ecl_lon_sun))
        
        mean_long_moon = 218.316 + 481267.881 * t_jc
        mean_anom_moon = 134.963 + 477198.867 * t_jc
        mean_dist_moon = 93.272 + 483202.017 * t_jc
        ecl_lon_moon = mean_long_moon + 6.289 * math.sin(math.radians(mean_anom_moon))
        ecl_lat_moon = 5.128 * math.sin(math.radians(mean_dist_moon))
        dist_moon = 385000000.0 - 20905000.0 * math.cos(math.radians(mean_anom_moon))
        
        xs_moon = dist_moon * math.cos(math.radians(ecl_lon_moon)) * math.cos(math.radians(ecl_lat_moon))
        ys_moon = dist_moon * (math.cos(math.radians(obliquity)) * math.sin(math.radians(ecl_lon_moon)) * math.cos(math.radians(ecl_lat_moon)) - math.sin(math.radians(obliquity)) * math.sin(math.radians(ecl_lat_moon)))
        zs_moon = dist_moon * (math.sin(math.radians(obliquity)) * math.sin(math.radians(ecl_lon_moon)) * math.cos(math.radians(ecl_lat_moon)) + math.cos(math.radians(obliquity)) * math.sin(math.radians(ecl_lat_moon)))
        
        r_sta = math.sqrt(X**2 + Y**2 + Z**2)
        if r_sta == 0: return 0.0, 0.0, 0.0
        
        rx, ry, rz = X/r_sta, Y/r_sta, Z/r_sta
        
        def deformacion_cuerpo(mass_ratio, R_body, xs, ys, zs):
            dist_body = math.sqrt(xs**2 + ys**2 + zs**2)
            if dist_body == 0: return 0.0, 0.0, 0.0
            ux, uy, uz = xs/dist_body, ys/dist_body, zs/dist_body
            cos_theta = rx*ux + ry*uy + rz*uz
            
            p2 = 1.5 * cos_theta**2 - 0.5
            p2_prime = 3.0 * cos_theta
            
            coef = (GM_earth / Re**2) * mass_ratio * (Re / dist_body)**3 * Re
            
            dr_radial = h2 * coef * p2
            dr_tangent = l2 * coef * p2_prime
            
            dx = dr_radial * rx + dr_tangent * (ux - cos_theta * rx)
            dy = dr_radial * ry + dr_tangent * (uy - cos_theta * ry)
            dz = dr_radial * rz + dr_tangent * (uz - cos_theta * rz)
            return dx, dy, dz

        dx_sun, dy_sun, dz_sun = deformacion_cuerpo(GM_sun/GM_earth, dist_sun, xs_sun, ys_sun, zs_sun)
        dx_moon, dy_moon, dz_moon = deformacion_cuerpo(GM_moon/GM_earth, dist_moon, xs_moon, ys_moon, zs_moon)
        
        return dx_sun + dx_moon, dy_sun + dy_moon, dz_sun + dz_moon
    except:
        return 0.0, 0.0, 0.0 

def calcular_saastamoinen(lat_deg, alt, elev_deg):
    if elev_deg < 5.0: elev_deg = 5.0
    lat_rad, elev_rad = max(math.radians(lat_deg), -math.pi/2), math.radians(elev_deg)
    H = max(0.0, min(alt, 40000.0))
    P = 1013.25 * ((1.0 - 2.2557e-5 * H) ** 5.2568)
    T = 288.15 - 0.0065 * H
    e = 6.11 * 0.5 * (10.0 ** (7.5 * (T - 273.15) / (T - 273.15 + 237.3))) * ((1.0 - 2.2557e-5 * H) ** 5.2568)
    zhd = (0.0022768 * P) / (1.0 - 0.00266 * math.cos(2.0 * lat_rad) - 0.00028 * (H / 1000.0))
    zwd = 0.0022768 * ((1255.0 / T) + 0.05) * e
    return (zhd + zwd) * (1.0 / math.sin(elev_rad))

def geodesicas_a_ecef(lat_deg, lon_deg, alt):
    a, e2 = 6378137.0, 0.0066943799901413155
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    N = a / math.sqrt(1 - e2 * (math.sin(lat) ** 2))
    return (N + alt) * math.cos(lat) * math.cos(lon), (N + alt) * math.cos(lat) * math.sin(lon), (N * (1 - e2) + alt) * math.sin(lat)

def ecef_a_geodesicas(x, y, z):
    a, e2 = 6378137.0, 0.0066943799901413155
    b = math.sqrt(a**2 * (1 - e2)); ep2 = (a**2 - b**2) / b**2
    p = math.sqrt(x**2 + y**2); th = math.atan2(a * z, b * p)
    lat = math.atan2((z + ep2 * b * (math.sin(th) ** 3)), (p - e2 * a * (math.cos(th) ** 3)))
    N = a / math.sqrt(1 - e2 * (math.sin(lat) ** 2))
    return math.degrees(lat), math.degrees(math.atan2(y, x)), p / math.cos(lat) - N

def geodesicas_a_utm(lat, lon, force_zone=19):
    a, e2 = 6378137.0, 0.0066943799901413155
    lat_r, lon_r = math.radians(lat), math.radians(lon)
    LongOrig = math.radians((force_zone - 1) * 6 - 180 + 3)
    ep2 = e2 / (1 - e2)
    N = a / math.sqrt(1 - e2 * math.sin(lat_r)**2)
    T = math.tan(lat_r)**2; C = ep2 * math.cos(lat_r)**2; A = math.cos(lat_r) * (lon_r - LongOrig)
    M = a * ((1 - e2/4 - 3*e2**2/64 - 5*e2**3/256)*lat_r - (3*e2/8 + 3*e2**2/32 + 45*e2**3/1024)*math.sin(2*lat_r) + (15*e2**2/256 + 45*e2**3/1024)*math.sin(4*lat_r) - (35*e2**3/3072)*math.sin(6*lat_r))
    Easting = 0.9996 * N * (A + (1-T+C)*A**3/6 + (5-18*T+T**2+72*C-58*ep2)*A**5/120) + 500000.0
    Northing = 0.9996 * (M + N*math.tan(lat_r)*(A**2/2 + (5-T+9*C+4*C**2)*A**4/24 + (61-58*T+T**2+600*C-330*ep2)*A**6/720))
    return (Northing + 10000000.0 if lat < 0 else Northing), Easting

def utm_a_geodesicas(easting, northing, zone=19, hemisferio='N'):
    a, e2 = 6378137.0, 0.0066943799901413155
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    x, y = easting - 500000.0, northing if hemisferio.upper() == 'N' else northing - 10000000.0
    m = y / 0.9996; mu = m / (a * (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256))
    phi1_rad = mu + (3*e1/2 - 27*e1**3/32)*math.sin(2*mu) + (21*e1**2/16 - 55*e1**4/32)*math.sin(4*mu)
    n1 = a / math.sqrt(1 - e2*math.sin(phi1_rad)**2)
    t1, c1 = math.tan(phi1_rad)**2, e2 / (1 - e2) * math.cos(phi1_rad)**2
    r1 = a * (1 - e2) / ((1 - e2*math.sin(phi1_rad)**2)**1.5)
    d = x / (n1 * 0.9996)
    lat_rad = phi1_rad - (n1*math.tan(phi1_rad)/r1) * (d**2/2 - (5 + 3*t1 + 10*c1)*d**4/24)
    lon_rad = (d - (1 + 2*t1 + c1)*d**3/6) / math.cos(phi1_rad)
    lon_origen = math.radians((zone - 1) * 6 - 180 + 3)
    return math.degrees(lat_rad), math.degrees(lon_rad + lon_origen), 0.0

def calcular_topocentricas(xs, ys, zs, X_usr, Y_usr, Z_usr):
    lat_val, lon_val, alt_val = ecef_a_geodesicas(X_usr, Y_usr, Z_usr)
    lat_r = math.radians(lat_val)
    lon_r = math.radians(lon_val)
    dx, dy, dz = xs - X_usr, ys - Y_usr, zs - Z_usr
    sin_lat, cos_lat = math.sin(lat_r), math.cos(lat_r)
    sin_lon, cos_lon = math.sin(lon_r), math.cos(lon_r)
    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    dist = math.sqrt(dx**2 + dy**2 + dz**2)
    if dist < 1e-6: return 0.0, 0.0
    val_asin = max(-1.0, min(1.0, u / dist))
    el = math.degrees(math.asin(val_asin))
    az = math.degrees(math.atan2(e, n))
    if az < 0: az += 360.0
    return el, az

def calcular_klobuchar(lat_deg, lon_deg, el_deg, az_deg, tow, alpha, beta):
    if not any(alpha) and not any(beta): return 0.0
    phi_u, lam_u = lat_deg / 180.0, lon_deg / 180.0
    E, A = el_deg / 180.0, az_deg / 180.0
    psi = 0.0137 / (E + 0.11) - 0.022
    phi_i = phi_u + psi * math.cos(A * math.pi)
    if phi_i > 0.416: phi_i = 0.416
    elif phi_i < -0.416: phi_i = -0.416
    lam_i = lam_u + (psi * math.sin(A * math.pi)) / math.cos(phi_i * math.pi)
    phi_m = phi_i + 0.064 * math.cos((lam_i - 1.617) * math.pi)
    t = 43200.0 * lam_i + tow
    t = t % 86400.0
    if t < 0: t += 86400.0
    F = 1.0 + 16.0 * (0.53 - E) ** 3
    PER = beta[0] + beta[1]*phi_m + beta[2]*(phi_m**2) + beta[3]*(phi_m**3)
    if PER < 72000.0: PER = 72000.0
    AMP = alpha[0] + alpha[1]*phi_m + alpha[2]*(phi_m**2) + alpha[3]*(phi_m**3)
    if AMP < 0.0: AMP = 0.0
    x = (2.0 * math.pi * (t - 50400.0)) / PER
    if abs(x) < 1.5707963267948966:
        return F * (5e-9 + AMP * (1.0 - (x**2)/2.0 + (x**4)/24.0)) * C_LIGHT
    return F * 5e-9 * C_LIGHT

def calcular_posicion_satelite_wgs84(eph, t_emision, tau_vuelo, sys_char='G'):
    if not eph or eph['sqrtA'] <= 0.0: return None
    mu_sys = 3.986004418e14 if sys_char in 'EC' else MU
    omega_e_sys = 7.292115e-5 if sys_char == 'C' else OMEGA_E
    A = eph['sqrtA'] ** 2
    n0 = math.sqrt(mu_sys / (A ** 3))
    t_k = t_emision - eph['Toe']
    if sys_char == 'C': t_k -= 14.0
    if t_k > 302400: t_k -= 604800
    elif t_k < -302400: t_k += 604800
    M_k = eph['M0'] + (n0 + eph['Delta_n']) * t_k; E_k = M_k
    for _ in range(5): E_k = M_k + eph['e'] * math.sin(E_k)
    dt_sat = eph['af0'] + eph['af1'] * t_k + eph['af2'] * (t_k ** 2)
    nu_k = math.atan2((math.sqrt(1 - eph['e']**2) * math.sin(E_k)), (math.cos(E_k) - eph['e']))
    phi_k = nu_k + eph['omega']
    u_k = phi_k + eph['Cus'] * math.sin(2 * phi_k) + eph['Cuc'] * math.cos(2 * phi_k)
    r_k = A * (1 - eph['e'] * math.cos(E_k)) + eph['Crs'] * math.sin(2 * phi_k) + eph['Crc'] * math.cos(2 * phi_k)
    i_k = eph['i0'] + eph['Cic'] * math.cos(2 * phi_k) + eph['Cis'] * math.sin(2 * phi_k) + eph['IDOT'] * t_k
    x_k, y_k = r_k * math.cos(u_k), r_k * math.sin(u_k)
    omega_k = eph['OMEGA'] + (eph['OMEGA_DOT'] - omega_e_sys) * t_k - omega_e_sys * eph['Toe']
    xs = x_k * math.cos(omega_k) - y_k * math.cos(i_k) * math.sin(omega_k)
    ys = x_k * math.sin(omega_k) + y_k * math.cos(i_k) * math.cos(omega_k)
    zs = y_k * math.sin(i_k)
    theta = omega_e_sys * tau_vuelo
    return (xs * math.cos(theta) + ys * math.sin(theta), -xs * math.sin(theta) + ys * math.cos(theta), zs, dt_sat)

# =====================================================================
# ENRUTADOR AUTOMÁTICO (REGLAS EXCLUYENTES V19)
# =====================================================================
def analizar_calidad_y_senales_rinex(obs_b, obs_r, max_gap_tolerado=0.05):
    tows_b = sorted(list(obs_b.keys()))
    tows_r = sorted(list(obs_r.keys()))
    
    if not tows_b or not tows_r: 
        return "MODO_C_SPP", 0.0, "Cero épocas. Archivo vacío o corrupto."
    
    t_ini_b, t_fin_b = tows_b[0], tows_b[-1]
    t_ini_r, t_fin_r = tows_r[0], tows_r[-1]
    
    overlap_ini = max(t_ini_b, t_ini_r)
    overlap_fin = min(t_fin_b, t_fin_r)
    
    if overlap_ini > overlap_fin:
        return "MODO_C_SPP", 0.0, "Cero solapamiento temporal entre Base y Rover. Requiere cálculo autónomo."
        
    sync_epochs = 0
    total_eval = 0
    
    base_L1_count = 0
    base_L5_count = 0
    rover_L1_count = 0
    rover_L5_count = 0
    
    for tr in tows_r:
        if tr < overlap_ini or tr > overlap_fin: continue
        total_eval += 1
        idx = min(range(len(tows_b)), key=lambda i: abs(tows_b[i] - tr))
        gap = abs(tows_b[idx] - tr)
        
        if gap <= max_gap_tolerado:
            sync_epochs += 1
            
        d_r = obs_r[tr]
        d_b = obs_b[tows_b[idx]]
        
        for s in d_r:
            if s == '_meta' or s not in d_b: continue
            
            if 'L1' in d_b[s] and d_b[s]['L1'] != 0.0: base_L1_count += 1
            if 'L5' in d_b[s] and d_b[s]['L5'] != 0.0: base_L5_count += 1
            
            if 'L1' in d_r[s] and d_r[s]['L1'] != 0.0: rover_L1_count += 1
            if 'L5' in d_r[s] and d_r[s]['L5'] != 0.0: rover_L5_count += 1
                
    if total_eval == 0: 
        return "MODO_C_SPP", 0.0, "Sin épocas en la ventana de solapamiento."
    
    ratio_sync = sync_epochs / total_eval
    
    tiene_doble_fase = (base_L1_count > 0 and base_L5_count > 0 and 
                        rover_L1_count > 0 and rover_L5_count > 0)
                        
    tiene_fase_mono = (base_L1_count > 0 and rover_L1_count > 0 and 
                       (base_L5_count == 0 or rover_L5_count == 0))
                       
    no_tiene_fase = (base_L1_count == 0 or rover_L1_count == 0)

    if tiene_doble_fase:
        return "MODO_C_PPK", ratio_sync, "Fase Dual detectada en Base y Rover (L1+L5). Enrutando a MÓDULO C (PPK Dual)."
    elif tiene_fase_mono:
        return "MODO_B_FASE", ratio_sync, "Fase Monofrecuencia (L1) detectada. Enrutando a MÓDULO B (Blindaje IRLS para evitar divergencia EKF)."
    elif no_tiene_fase:
        if ratio_sync > 0.5:
            return "MODO_A_CODIGO", ratio_sync, "Código Puro homogéneo (>50% sinc). Enrutando a MÓDULO A (EKF de Código)."
        else:
            return "MODO_B_ASINCRONO", ratio_sync, "Código Puro con asincronía severa (<50% sinc). Enrutando a MÓDULO B (IRLS Asíncrono)."
    else:
        return "MODO_B_ASINCRONO", ratio_sync, "Fallo en clasificación lógica. Enrutando a Módulo de rescate B."

# =====================================================================
# AISLAMIENTO DE OBSERVABLES (MÓDULOS A, B Y C)
# =====================================================================
def aislar_diferencias_simples_ppk(obs_b, obs_r):
    sd_suavizada = {}
    for tow in sorted(list(obs_r.keys())):
        if tow not in obs_b: continue
        
        l1_count, l5_count = 0, 0
        for s, d_r in obs_r[tow].items():
            if s == '_meta' or s not in obs_b[tow]: continue
            d_b = obs_b[tow][s]
            if d_b.get('C5') and d_r.get('C5') and d_b.get('L5') and d_r.get('L5'): l5_count += 1
            if d_b.get('C1') and d_r.get('C1') and d_b.get('L1') and d_r.get('L1'): l1_count += 1
        
        use_l5 = (l5_count >= 4) or (l5_count >= l1_count and l5_count >= 3)
        
        sd_epoca = {'_meta': obs_r[tow]['_meta'], '_tow_b': tow}
        for s, d_r in obs_r[tow].items():
            if s == '_meta' or s not in obs_b[tow]: continue
            d_b = obs_b[tow][s]
            pr_b, pr_r, cp_b, cp_r, wave_sys = None, None, None, None, None
            
            if use_l5 and d_b.get('C5') and d_r.get('C5'):
                pr_b, pr_r = d_b['C5'], d_r['C5']
                cp_b, cp_r = d_b.get('L5'), d_r.get('L5')
                wave_sys = WAVE_L5
            elif not use_l5 and d_b.get('C1') and d_r.get('C1'):
                pr_b, pr_r = d_b['C1'], d_r['C1']
                cp_b, cp_r = d_b.get('L1'), d_r.get('L1')
                wave_sys = WAVE_L1
                
            if not pr_b or not pr_r or not wave_sys: continue
            
            snr_b = d_b.get('S5', d_b.get('S1', 30.0)) if use_l5 else d_b.get('S1', d_b.get('S5', 30.0))
            snr_r = d_r.get('S5', d_r.get('S1', 30.0)) if use_l5 else d_r.get('S1', d_r.get('S5', 30.0))
            
            sd_epoca[s] = {
                'sd_P': pr_r - pr_b, 
                'pr_b': pr_b, 'pr_r': pr_r, 
                'cp_b': cp_b, 'cp_r': cp_r,
                'wave': wave_sys,
                'snr': min(snr_b, snr_r),
                'sys': s[0]
            }
        if len(sd_epoca) > 2: sd_suavizada[tow] = sd_epoca
    return sd_suavizada

def aislar_diferencias_MODO_B(obs_b, obs_r):
    sd_suavizada = {}
    for tow in sorted(list(obs_r.keys())):
        if tow not in obs_b: continue
        
        sd_epoca = {'_meta': obs_r[tow]['_meta']}
        for s, d_r in obs_r[tow].items():
            if s == '_meta' or s not in obs_b[tow]: continue
            d_b = obs_b[tow]
            
            freq = 'L1' 
            if 'C5' in d_b[s] and 'C5' in d_r and 'L5' in d_b[s] and 'L5' in d_r:
                freq = 'L5' 
            elif not ('C1' in d_b[s] and 'C1' in d_r): continue
            
            pr_b = d_b[s]['C5'] if freq == 'L5' else d_b[s]['C1']
            pr_r = d_r['C5'] if freq == 'L5' else d_r['C1']
            
            snr_b = d_b[s].get('S5', 30.0) if freq == 'L5' else d_b[s].get('S1', 30.0)
            snr_r = d_r.get('S5', 30.0) if freq == 'L5' else d_r.get('S1', 30.0)
            
            sd_P = pr_r - pr_b
            
            sd_epoca[s] = {
                'sd_P': sd_P,
                'pr_b': pr_b, 'pr_r': pr_r,
                'snr': min(snr_b, snr_r)
            }
        if len(sd_epoca) > 1: sd_suavizada[tow] = sd_epoca
    return sd_suavizada

def aislar_diferencias_MODO_C(obs_b, obs_r):
    sd_suavizada = {}
    for tow in sorted(list(obs_r.keys())):
        if tow not in obs_b: continue
        
        sd_epoca = {'_meta': obs_r[tow]['_meta'], '_tow_b': tow}
        for s, d_r in obs_r[tow].items():
            if s == '_meta' or s not in obs_b[tow]: continue
            d_b = obs_b[tow][s]
            
            c1_r = d_r.get('C1')
            l1_r = d_r.get('L1')
            c5_r = d_r.get('C5')
            l5_r = d_r.get('L5')
            
            c1_b = d_b.get('C1')
            l1_b = d_b.get('L1')
            c5_b = d_b.get('C5')
            l5_b = d_b.get('L5')
            
            if c1_r is not None and c1_b is not None and c5_r is not None and c5_b is not None:
                sd_epoca[s] = {
                    'C1': {
                        'sd_P': c1_r - c1_b, 'pr_b': c1_b, 'pr_r': c1_r, 
                        'cp_b': l1_b, 'cp_r': l1_r, 'wave': WAVE_L1, 
                        'snr': min(d_b.get('S1', 30.0), d_r.get('S1', 30.0))
                    },
                    'C5': {
                        'sd_P': c5_r - c5_b, 'pr_b': c5_b, 'pr_r': c5_r, 
                        'cp_b': l5_b, 'cp_r': l5_r, 'wave': WAVE_L5, 
                        'snr': min(d_b.get('S5', 30.0), d_r.get('S5', 30.0))
                    },
                    'sys': s[0]
                }
        if len(sd_epoca) > 2: sd_suavizada[tow] = sd_epoca
    return sd_suavizada

# =====================================================================
# VÍA 1 -> MÓDULO A: MOTOR EKF + LAMBDA + RTS
# =====================================================================
def decorrelacion_lambda_z(Q):
    n = len(Q)
    Z = matid(n)
    try:
        L = cholesky_decompose(Q)
    except:
        return Z, Q 
    for i in range(n - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            mu = round(L[i][j] / L[j][j])
            if mu != 0:
                for k in range(j + 1): L[i][k] -= mu * L[j][k]
                for k in range(n): Z[k][i] -= mu * Z[k][j]
    Z_T = transpose_matrix(Z)
    Q_z = matmul(matmul(Z_T, Q), Z)
    return Z, Q_z

def suavizador_rts_backward(forward_states):
    n = len(forward_states)
    if n == 0: return []
    
    smoothed_states = [None] * n
    smoothed_states[-1] = forward_states[-1]['X_post']
    
    for k in range(n-2, -1, -1):
        P_post_k = forward_states[k]['P_post']
        P_pri_k1 = forward_states[k+1]['P_pri']
        
        P_pri_inv = invert_matrix_nxn(P_pri_k1)
        if not P_pri_inv:
            smoothed_states[k] = forward_states[k]['X_post']
            continue
            
        C_k = matmul(P_post_k, P_pri_inv)
        
        X_smooth_k1 = smoothed_states[k+1]
        X_pri_k1 = forward_states[k+1]['X_pri']
        
        dx = [[X_smooth_k1[i][0] - X_pri_k1[i][0]] for i in range(3)]
        correction = matmul(C_k, dx)
        
        X_post_k = forward_states[k]['X_post']
        smoothed_states[k] = [[X_post_k[i][0] + correction[i][0]] for i in range(3)]
        
    return smoothed_states

def procesar_ekF_lambda(sd_epoca, nav, sp3, kf_estado, tr, mask_angle, snr_mask):
    try:
        tow_b = sd_epoca.get('_tow_b', tr)
        
        X_pri = [[kf_estado['X'][0][0]], [kf_estado['X'][1][0]], [kf_estado['X'][2][0]]]
        P_pri = [row[:] for row in kf_estado['P']]
        h_r = kf_estado.get('h_r', 0.0)
        
        if 'prev_cp' not in kf_estado: kf_estado['prev_cp'] = {}
        
        X_iter = X_pri[0][0]
        Y_iter = X_pri[1][0]
        Z_iter = X_pri[2][0]
        
        lat_r, lon_r, alt_r = ecef_a_geodesicas(X_iter, Y_iter, Z_iter)
        lat_rad, lon_rad = math.radians(lat_r), math.radians(lon_r)
        
        X_apc = X_iter + h_r * math.cos(lat_rad) * math.cos(lon_rad)
        Y_apc = Y_iter + h_r * math.cos(lat_rad) * math.sin(lon_rad)
        Z_apc = Z_iter + h_r * math.sin(lat_rad)
        
        alpha = nav.get('_iono', {}).get('alpha', [0]*4)
        beta = nav.get('_iono', {}).get('beta', [0]*4)
        
        y_m, m_m, d_m, h_m, mn_m, sec_m = sd_epoca['_meta']
        
        dx_tide, dy_tide, dz_tide = correccion_mareas_solidas(
            kf_estado['X_base'][0], kf_estado['X_base'][1], kf_estado['X_base'][2], 
            tow_b, y_m, m_m, d_m
        )
        
        X_base_corr = kf_estado['X_base'][0] + dx_tide
        Y_base_corr = kf_estado['X_base'][1] + dy_tide
        Z_base_corr = kf_estado['X_base'][2] + dz_tide
        
        lat_base, lon_base, alt_base = ecef_a_geodesicas(X_base_corr, Y_base_corr, Z_base_corr)
        
        sat_positions = {}
        for s, d in sd_epoca.items():
            if s == '_meta' or s == '_tow_b' or d['sd_P'] is None: continue 
            tau_r = d['pr_r'] / C_LIGHT
            tau_b = d['pr_b'] / C_LIGHT
            
            t_emision_r = tr - tau_r
            t_emision_b = tow_b - tau_b
            
            sp_r, sp_b = None, None
            
            if sp3 and s in sp3:
                sp3_res_r = interpolate_sp3(sp3, s, t_emision_r)
                sp3_res_b = interpolate_sp3(sp3, s, t_emision_b)
                if sp3_res_r and sp3_res_b:
                    theta_r = OMEGA_E * tau_r
                    xs_r = sp3_res_r[0] * math.cos(theta_r) + sp3_res_r[1] * math.sin(theta_r)
                    ys_r = -sp3_res_r[0] * math.sin(theta_r) + sp3_res_r[1] * math.cos(theta_r)
                    sp_r = (xs_r, ys_r, sp3_res_r[2], sp3_res_r[3]) 
                    
                    theta_b = OMEGA_E * tau_b
                    xs_b = sp3_res_b[0] * math.cos(theta_b) + sp3_res_b[1] * math.sin(theta_b)
                    ys_b = -sp3_res_b[0] * math.sin(theta_b) + sp3_res_b[1] * math.cos(theta_b)
                    sp_b = (xs_b, ys_b, sp3_res_b[2], sp3_res_b[3])
            
            if not sp_r or not sp_b:
                sp_r = calcular_posicion_satelite_wgs84(seleccionar_efemeride_optima(nav.get(s), t_emision_r), t_emision_r, tau_r, s[0])
                sp_b = calcular_posicion_satelite_wgs84(seleccionar_efemeride_optima(nav.get(s), t_emision_b), t_emision_b, tau_b, s[0])
            
            if sp_r and sp_b:
                el_r, az_r = calcular_topocentricas(sp_r[0], sp_r[1], sp_r[2], X_apc, Y_apc, Z_apc)
                if el_r >= mask_angle and d.get('snr', 30.0) >= snr_mask:
                    sat_positions[s] = {'sp_r': sp_r, 'sp_b': sp_b, 'sd_P': d['sd_P'], 'cp_r': d['cp_r'], 'cp_b': d['cp_b'], 'wave': d['wave'], 'snr': d['snr'], 'sys': d['sys']}
        
        if len(sat_positions) < 4: return None, "FAILED", kf_estado, None
        
        sat_list_full = list(sat_positions.keys())
        constellations = set([s[0] for s in sat_list_full])
        ref_sats = {}
        sat_list = []
        
        for c in constellations:
            c_sats = [s for s in sat_list_full if s[0] == c]
            if len(c_sats) >= 2:
                r_candidate = max(c_sats, key=lambda k: calcular_topocentricas(sat_positions[k]['sp_r'][0], sat_positions[k]['sp_r'][1], sat_positions[k]['sp_r'][2], X_apc, Y_apc, Z_apc)[0])
                ref_sats[c] = r_candidate
                c_sats.remove(ref_sats[c])
                sat_list.extend(c_sats)
        
        if len(sat_list) < 3: return None, "FAILED", kf_estado, None
        
        def calc_rho(sp, X, Y, Z, lat, lon, alt, el, az, wave, t_eval):
            dist = math.sqrt((sp[0]-X)**2 + (sp[1]-Y)**2 + (sp[2]-Z)**2)
            tropo = calcular_saastamoinen(lat, alt, el)
            iono_m = calcular_klobuchar(lat, lon, el, az, t_eval, alpha, beta)
            if wave == WAVE_L5: iono_m *= 1.79327 
            return dist + tropo, iono_m, dist

        base_calcs = {}
        for s, data in sat_positions.items():
            el_b, az_b = calcular_topocentricas(data['sp_b'][0], data['sp_b'][1], data['sp_b'][2], X_base_corr, Y_base_corr, Z_base_corr)
            rho_b, iono_b, dist_b = calc_rho(data['sp_b'], X_base_corr, Y_base_corr, Z_base_corr, lat_base, lon_base, alt_base, el_b, az_b, data['wave'], tow_b)
            base_calcs[s] = {'P': rho_b + iono_b, 'CP': rho_b - iono_b}

        H = []; L = []; R_diag = []
        
        c_ref = {}
        for c, r_sat in ref_sats.items():
            r_data = sat_positions[r_sat]
            el_r, az_r = calcular_topocentricas(r_data['sp_r'][0], r_data['sp_r'][1], r_data['sp_r'][2], X_apc, Y_apc, Z_apc)
            rho_r, iono_r, dist_r = calc_rho(r_data['sp_r'], X_apc, Y_apc, Z_apc, lat_r, lon_r, alt_r + h_r, el_r, az_r, r_data['wave'], tr)
            
            SD_P_calc_ref = (rho_r + iono_r) - base_calcs[r_sat]['P']
            SD_CP_calc_ref = (rho_r - iono_r) - base_calcs.get(r_sat, {}).get('CP', 0.0)
            
            c_ref[c] = {'dist_r': dist_r, 'SD_P_calc_ref': SD_P_calc_ref, 'SD_CP_calc_ref': SD_CP_calc_ref, 'sp_r': r_data['sp_r'], 'el_r': el_r, 'snr': r_data['snr'], 'sd_P': r_data['sd_P'], 'cp_r': r_data['cp_r'], 'cp_b': r_data['cp_b']}
        
        for s in sat_list:
            c = s[0]
            data = sat_positions[s]
            rc = c_ref[c]
            
            el_i_r, az_i_r = calcular_topocentricas(data['sp_r'][0], data['sp_r'][1], data['sp_r'][2], X_apc, Y_apc, Z_apc)
            rho_i_r, iono_i_r, dist_i_r = calc_rho(data['sp_r'], X_apc, Y_apc, Z_apc, lat_r, lon_r, alt_r + h_r, el_i_r, az_i_r, data['wave'], tr)
            
            SD_P_calc_i = (rho_i_r + iono_i_r) - base_calcs[s]['P']
            DD_P_calc = SD_P_calc_i - rc['SD_P_calc_ref']
            
            dx_geom = [
                -(data['sp_r'][0] - X_apc) / dist_i_r - (-(rc['sp_r'][0] - X_apc) / rc['dist_r']),
                -(data['sp_r'][1] - Y_apc) / dist_i_r - (-(rc['sp_r'][1] - Y_apc) / rc['dist_r']),
                -(data['sp_r'][2] - Z_apc) / dist_i_r - (-(rc['sp_r'][2] - Z_apc) / rc['dist_r'])
            ]
            
            var_base = (10.0 ** (-data['snr'] / 10.0)) * 100.0
            
            DD_P_obs = data['sd_P'] - rc['sd_P']
            L.append([DD_P_obs - DD_P_calc])
            H.append(dx_geom)
            R_diag.append(var_base * 9.0)
            
            if data['cp_r'] is not None and data['cp_b'] is not None and rc['cp_r'] is not None and rc['cp_b'] is not None:
                wave = data['wave']
                SD_CP_calc_i = (rho_i_r - iono_i_r) - base_calcs[s]['CP']
                DD_CP_calc = SD_CP_calc_i - rc['SD_CP_calc_ref']
                
                cp_valid = True
                if s in kf_estado['prev_cp']:
                    delta_P = data['sd_P'] - kf_estado['prev_cp'][s]['P']
                    delta_CP = (data['cp_r'] - data['cp_b']) - kf_estado['prev_cp'][s]['CP']
                    if abs(delta_P - delta_CP * wave) > 5.0: 
                        cp_valid = False
                
                kf_estado['prev_cp'][s] = {'P': data['sd_P'], 'CP': data['cp_r'] - data['cp_b']}
                
                if cp_valid:
                    DD_CP_obs = (data['cp_r'] - data['cp_b']) - (rc['cp_r'] - rc['cp_b'])
                    DD_CP_m = DD_CP_obs * wave
                    
                    var_amb = [[var_base * 0.0001]]
                    Z_trans, Q_z = decorrelacion_lambda_z(var_amb)
                    
                    ambiguity_float = (DD_CP_m - DD_CP_calc) / wave
                    amb_z = ambiguity_float * Z_trans[0][0]
                    amb_round = round(amb_z)
                    amb_restored = amb_round / Z_trans[0][0]
                    
                    if abs(ambiguity_float - amb_restored) < 0.20:
                        L.append([(DD_CP_m - amb_restored * wave) - DD_CP_calc])
                        H.append(dx_geom)
                        R_diag.append(var_base * 0.0001)
                        kf_estado['fix_flags'] += 1

        if not H: return None, "FAILED", kf_estado, None
        
        H_T = transpose_matrix(H)
        R_inv = matid(len(R_diag))
        for i in range(len(R_diag)): R_inv[i][i] = 1.0 / max(1e-6, R_diag[i])
        
        P_inv = invert_matrix_nxn(P_pri)
        if not P_inv: return None, "FAILED", kf_estado, None
        
        H_T_R_inv = matmul(H_T, R_inv)
        N_mat = matadd(matmul(H_T_R_inv, H), P_inv)
        U_vec = matmul(H_T_R_inv, L)
        
        Q_cov = invert_matrix_nxn(N_mat)
        if not Q_cov: return None, "FAILED", kf_estado, None
        
        Delta_X = matmul(Q_cov, U_vec)
        
        X_post = [
            [X_pri[0][0] + Delta_X[0][0]],
            [X_pri[1][0] + Delta_X[1][0]],
            [X_pri[2][0] + Delta_X[2][0]]
        ]
        
        kf_estado['X'] = X_post
        kf_estado['P'] = Q_cov
        
        status = "FIXED (PPK)" if kf_estado['fix_flags'] > 4 else "FLOAT (DGPS)"
        kf_estado['fix_flags'] = 0 
        
        state_dict = {
            'tow': tr, 'X_pri': X_pri, 'P_pri': P_pri, 'X_post': X_post, 'P_post': Q_cov, 'tide': (dx_tide, dy_tide, dz_tide)
        }
        
        return (X_post[0][0] - dx_tide, X_post[1][0] - dy_tide, X_post[2][0] - dz_tide), status, kf_estado, state_dict

    except Exception as e:
        return None, f"FAILED_EXCEPTION:_{str(e)}", kf_estado, None

# =====================================================================
# VÍA 2 -> MÓDULO B: MOTOR IRLS CLÁSICO
# =====================================================================
def calcular_IRLS_MODO_B(sd_epoca, nav, X_b, Y_b, Z_b, tr, mask_angle):
    try:
        y_m, m_m, d_m, h_m, mn_m, sec_m = sd_epoca['_meta']
        dx_tide, dy_tide, dz_tide = correccion_mareas_solidas(X_b, Y_b, Z_b, tr, y_m, m_m, d_m)
        X_base_corr = X_b + dx_tide
        Y_base_corr = Y_b + dy_tide
        Z_base_corr = Z_b + dz_tide
        
        X_iter, Y_iter, Z_iter = X_base_corr, Y_base_corr, Z_base_corr 
        lat_b, lon_b, alt_b = ecef_a_geodesicas(X_base_corr, Y_base_corr, Z_base_corr)
        
        iono = nav.get('_iono', {'alpha': [0]*4, 'beta': [0]*4})
        alpha, beta = iono['alpha'], iono['beta']
        
        sat_positions = {}
        for s, d in sd_epoca.items():
            if s == '_meta' or d['sd_P'] is None: continue 
            tau = d['pr_r'] / C_LIGHT
            sp = calcular_posicion_satelite_wgs84(seleccionar_efemeride_optima(nav.get(s), tr-tau), tr-tau, tau, s[0])
            if sp:
                el_r, az_r = calcular_topocentricas(sp[0], sp[1], sp[2], X_iter, Y_iter, Z_iter)
                if el_r >= mask_angle:
                    sat_positions[s] = {'sp': sp, 'el': el_r, 'az': az_r, 'sd_P': d['sd_P'], 'snr': d.get('snr', 30.0)}
        
        if len(sat_positions) < 4: return None, "FAILED"
        
        sat_list_full = list(sat_positions.keys())
        constellations = set([s[0] for s in sat_list_full])
        ref_sats = {}
        sat_list = []
        
        for c in constellations:
            c_sats = [s for s in sat_list_full if s[0] == c]
            if len(c_sats) >= 2:
                ref_sats[c] = max(c_sats, key=lambda k: sat_positions[k]['el'])
                c_sats.remove(ref_sats[c])
                sat_list.extend(c_sats)
        
        if len(sat_list) < 3: return None, "FAILED" 
        
        def calc_rho(sp, X, Y, Z, lat, lon, alt, el, az):
            dist = math.sqrt((sp[0]-X)**2 + (sp[1]-Y)**2 + (sp[2]-Z)**2)
            tropo = calcular_saastamoinen(lat, alt, el)
            iono_m = calcular_klobuchar(lat, lon, el, az, tr, alpha, beta)
            return dist + tropo, iono_m, dist

        prev_residuals = [0.0] * len(sat_list)

        for iteracion in range(8):
            lat_it, lon_it, alt_it = ecef_a_geodesicas(X_iter, Y_iter, Z_iter)
            
            H = []      
            L = []      
            W_diag = [] 
            
            ref_calcs = {}
            for c, r_sat in ref_sats.items():
                r_data = sat_positions[r_sat]
                rho_ref_r_base, iono_ref_r, dist_ref_r = calc_rho(r_data['sp'], X_iter, Y_iter, Z_iter, lat_it, lon_it, alt_it, r_data['el'], r_data['az'])
                el_ref_b, az_ref_b = calcular_topocentricas(r_data['sp'][0], r_data['sp'][1], r_data['sp'][2], X_base_corr, Y_base_corr, Z_base_corr)
                rho_ref_b_base, iono_ref_b, _ = calc_rho(r_data['sp'], X_base_corr, Y_base_corr, Z_base_corr, lat_b, lon_b, alt_b, el_ref_b, az_ref_b)
                
                ref_calcs[c] = {
                    'dist_ref_r': dist_ref_r,
                    'SD_P_calc_ref': (rho_ref_r_base + iono_ref_r) - (rho_ref_b_base + iono_ref_b),
                    'sp': r_data['sp'],
                    'el': r_data['el'],
                    'snr': r_data.get('snr', 30.0),
                    'sd_P': r_data['sd_P']
                }
            
            res_idx = 0
            for i, s in enumerate(sat_list):
                c = s[0]
                data = sat_positions[s]
                rc = ref_calcs[c]
                
                rho_i_r_base, iono_i_r, dist_i_r = calc_rho(data['sp'], X_iter, Y_iter, Z_iter, lat_it, lon_it, alt_it, data['el'], data['az'])
                el_i_b, az_i_b = calcular_topocentricas(data['sp'][0], data['sp'][1], data['sp'][2], X_base_corr, Y_base_corr, Z_base_corr)
                rho_i_b_base, iono_i_b, _ = calc_rho(data['sp'], X_base_corr, Y_base_corr, Z_base_corr, lat_b, lon_b, alt_b, el_i_b, az_i_b)
                
                SD_P_calc_i = (rho_i_r_base + iono_i_r) - (rho_i_b_base + iono_i_b)
                DD_P_calc = SD_P_calc_i - rc['SD_P_calc_ref']
                
                dx_geom = [
                    -(data['sp'][0] - X_iter) / dist_i_r - (-(rc['sp'][0] - X_iter) / rc['dist_ref_r']),
                    -(data['sp'][1] - Y_iter) / dist_i_r - (-(rc['sp'][1] - Y_iter) / rc['dist_ref_r']),
                    -(data['sp'][2] - Z_iter) / dist_i_r - (-(rc['sp'][2] - Z_iter) / rc['dist_ref_r'])
                ]
                
                sin_el_i_sq = math.sin(math.radians(data['el']))**2
                sin_el_ref_sq = math.sin(math.radians(rc['el']))**2
                snr_i_pow = 10.0 ** (data.get('snr', 30.0) / 10.0)
                snr_ref_pow = 10.0 ** (rc['snr'] / 10.0)
                
                w_i_ref = (sin_el_i_sq * snr_i_pow * sin_el_ref_sq * snr_ref_pow) / max(1.0, (sin_el_i_sq * snr_i_pow) + (sin_el_ref_sq * snr_ref_pow))

                DD_P_obs = data['sd_P'] - rc['sd_P']
                res_P = DD_P_obs - DD_P_calc
                
                L.append([res_P])
                H.append(dx_geom)
                
                if iteracion == 0:
                    w_P = w_i_ref * 1.0
                else:
                    w_P = w_i_ref * 1.0 / max(1.0, abs(prev_residuals[res_idx]) / 2.0)
                W_diag.append(w_P)
                res_idx += 1

            H_T = transpose_matrix(H)
            if not H_T or not W_diag: return None, "FAILED" 
            
            try:
                H_T_W = [[H_T[r][idx] * W_diag[idx] for idx in range(len(W_diag))] for r in range(len(H_T))]
            except IndexError:
                return None, "FAILED"

            N_mat = matmul(H_T_W, H)
            
            for r in range(len(N_mat)):
                N_mat[r][r] += abs(N_mat[r][r]) * 1e-6 + 1e-6
                
            U_vec = matmul(H_T_W, L)
            
            Q = invert_matrix_nxn(N_mat)
            if not Q: return None, "FAILED"
            
            Delta_X = matmul(Q, U_vec)
            if not Delta_X or len(Delta_X) < 3 or not Delta_X[0]: return None, "FAILED" 

            X_iter += Delta_X[0][0]; Y_iter += Delta_X[1][0]; Z_iter += Delta_X[2][0]
                
            prev_residuals = []
            for r in range(len(H)):
                v_val = sum(H[r][idx] * Delta_X[idx][0] for idx in range(len(H[0]))) - L[r][0]
                prev_residuals.append(v_val)
            
            if max(abs(Delta_X[0][0]), abs(Delta_X[1][0]), abs(Delta_X[2][0])) < 1e-3:
                return (X_iter - dx_tide, Y_iter - dy_tide, Z_iter - dz_tide), "FLOAT (DGPS Clásico Termux)"
                
        return (X_iter - dx_tide, Y_iter - dy_tide, Z_iter - dz_tide), "FLOAT (DGPS Clásico Termux)"
    except Exception as e:
        return None, f"FAILED_EXCEPTION:_{str(e)}"

# =====================================================================
# VÍA 3 -> MÓDULO C: NUEVO MOTOR PPK L1+L5 (DOBLE FRECUENCIA SIMULTÁNEA)
# =====================================================================
def procesar_ekF_PPK_L1_L5(sd_epoca, nav, sp3, kf_estado, tr, mask_angle, snr_mask):
    try:
        tow_b = sd_epoca.get('_tow_b', tr)
        
        X_pri = [[kf_estado['X'][0][0]], [kf_estado['X'][1][0]], [kf_estado['X'][2][0]]]
        P_pri = [row[:] for row in kf_estado['P']]
        h_r = kf_estado.get('h_r', 0.0)
        
        if 'prev_cp' not in kf_estado: kf_estado['prev_cp'] = {}
        
        X_iter, Y_iter, Z_iter = X_pri[0][0], X_pri[1][0], X_pri[2][0]
        lat_r, lon_r, alt_r = ecef_a_geodesicas(X_iter, Y_iter, Z_iter)
        lat_rad, lon_rad = math.radians(lat_r), math.radians(lon_r)
        
        X_apc = X_iter + h_r * math.cos(lat_rad) * math.cos(lon_rad)
        Y_apc = Y_iter + h_r * math.cos(lat_rad) * math.sin(lon_rad)
        Z_apc = Z_iter + h_r * math.sin(lat_rad)
        
        alpha, beta = nav.get('_iono', {}).get('alpha', [0]*4), nav.get('_iono', {}).get('beta', [0]*4)
        
        y_m, m_m, d_m, h_m, mn_m, sec_m = sd_epoca['_meta']
        dx_tide, dy_tide, dz_tide = correccion_mareas_solidas(kf_estado['X_base'][0], kf_estado['X_base'][1], kf_estado['X_base'][2], tow_b, y_m, m_m, d_m)
        X_base_corr, Y_base_corr, Z_base_corr = kf_estado['X_base'][0] + dx_tide, kf_estado['X_base'][1] + dy_tide, kf_estado['X_base'][2] + dz_tide
        lat_base, lon_base, alt_base = ecef_a_geodesicas(X_base_corr, Y_base_corr, Z_base_corr)
        
        sat_positions = {}
        for s, d in sd_epoca.items():
            if s == '_meta' or s == '_tow_b' or 'C1' not in d or d['C1']['sd_P'] is None: continue 
            
            tau_r = d['C1']['pr_r'] / C_LIGHT 
            tau_b = d['C1']['pr_b'] / C_LIGHT
            t_emision_r, t_emision_b = tr - tau_r, tow_b - tau_b
            
            sp_r, sp_b = None, None
            if sp3 and s in sp3:
                sp3_res_r, sp3_res_b = interpolate_sp3(sp3, s, t_emision_r), interpolate_sp3(sp3, s, t_emision_b)
                if sp3_res_r and sp3_res_b:
                    theta_r, theta_b = OMEGA_E * tau_r, OMEGA_E * tau_b
                    xs_r, ys_r = sp3_res_r[0] * math.cos(theta_r) + sp3_res_r[1] * math.sin(theta_r), -sp3_res_r[0] * math.sin(theta_r) + sp3_res_r[1] * math.cos(theta_r)
                    xs_b, ys_b = sp3_res_b[0] * math.cos(theta_b) + sp3_res_b[1] * math.sin(theta_b), -sp3_res_b[0] * math.sin(theta_b) + sp3_res_b[1] * math.cos(theta_b)
                    sp_r, sp_b = (xs_r, ys_r, sp3_res_r[2], sp3_res_r[3]), (xs_b, ys_b, sp3_res_b[2], sp3_res_b[3])
            
            if not sp_r or not sp_b:
                sp_r = calcular_posicion_satelite_wgs84(seleccionar_efemeride_optima(nav.get(s), t_emision_r), t_emision_r, tau_r, s[0])
                sp_b = calcular_posicion_satelite_wgs84(seleccionar_efemeride_optima(nav.get(s), t_emision_b), t_emision_b, tau_b, s[0])
            
            if sp_r and sp_b:
                el_r, az_r = calcular_topocentricas(sp_r[0], sp_r[1], sp_r[2], X_apc, Y_apc, Z_apc)
                if el_r >= mask_angle and d['C1'].get('snr', 30.0) >= snr_mask and d['C5'].get('snr', 30.0) >= snr_mask:
                    sat_positions[s] = {'sp_r': sp_r, 'sp_b': sp_b, 'data': d}
        
        if len(sat_positions) < 4: return None, "FAILED", kf_estado, None
        
        sat_list_full = list(sat_positions.keys())
        constellations = set([s[0] for s in sat_list_full])
        ref_sats = {}
        sat_list = []
        
        for c in constellations:
            c_sats = [s for s in sat_list_full if s[0] == c]
            if len(c_sats) >= 2:
                r_candidate = max(c_sats, key=lambda k: calcular_topocentricas(sat_positions[k]['sp_r'][0], sat_positions[k]['sp_r'][1], sat_positions[k]['sp_r'][2], X_apc, Y_apc, Z_apc)[0])
                ref_sats[c] = r_candidate
                c_sats.remove(ref_sats[c])
                sat_list.extend(c_sats)
        
        if len(sat_list) < 3: return None, "FAILED", kf_estado, None
        
        def calc_rho(sp, X, Y, Z, lat, lon, alt, el, az, wave, t_eval):
            dist = math.sqrt((sp[0]-X)**2 + (sp[1]-Y)**2 + (sp[2]-Z)**2)
            tropo = calcular_saastamoinen(lat, alt, el)
            iono_m = calcular_klobuchar(lat, lon, el, az, t_eval, alpha, beta)
            if wave == WAVE_L5: iono_m *= 1.79327 
            return dist + tropo, iono_m, dist

        base_calcs = {}
        for s, info in sat_positions.items():
            el_b, az_b = calcular_topocentricas(info['sp_b'][0], info['sp_b'][1], info['sp_b'][2], X_base_corr, Y_base_corr, Z_base_corr)
            rho_b, iono_b_L1, _ = calc_rho(info['sp_b'], X_base_corr, Y_base_corr, Z_base_corr, lat_base, lon_base, alt_base, el_b, az_b, WAVE_L1, tow_b)
            rho_b, iono_b_L5, _ = calc_rho(info['sp_b'], X_base_corr, Y_base_corr, Z_base_corr, lat_base, lon_base, alt_base, el_b, az_b, WAVE_L5, tow_b)
            base_calcs[s] = {'P1': rho_b + iono_b_L1, 'CP1': rho_b - iono_b_L1, 'P5': rho_b + iono_b_L5, 'CP5': rho_b - iono_b_L5}

        H = []; L = []; R_diag = []
        c_ref = {}
        for c, r_sat in ref_sats.items():
            info = sat_positions[r_sat]
            el_r, az_r = calcular_topocentricas(info['sp_r'][0], info['sp_r'][1], info['sp_r'][2], X_apc, Y_apc, Z_apc)
            rho_r, iono_r_L1, dist_r = calc_rho(info['sp_r'], X_apc, Y_apc, Z_apc, lat_r, lon_r, alt_r + h_r, el_r, az_r, WAVE_L1, tr)
            rho_r, iono_r_L5, _ = calc_rho(info['sp_r'], X_apc, Y_apc, Z_apc, lat_r, lon_r, alt_r + h_r, el_r, az_r, WAVE_L5, tr)
            
            c_ref[c] = {
                'dist_r': dist_r, 'sp_r': info['sp_r'], 'data': info['data'],
                'SD_P1_ref': (rho_r + iono_r_L1) - base_calcs[r_sat]['P1'], 'SD_CP1_ref': (rho_r - iono_r_L1) - base_calcs[r_sat]['CP1'],
                'SD_P5_ref': (rho_r + iono_r_L5) - base_calcs[r_sat]['P5'], 'SD_CP5_ref': (rho_r - iono_r_L5) - base_calcs[r_sat]['CP5']
            }
        
        for s in sat_list:
            info, rc = sat_positions[s], c_ref[s[0]]
            el_i_r, az_i_r = calcular_topocentricas(info['sp_r'][0], info['sp_r'][1], info['sp_r'][2], X_apc, Y_apc, Z_apc)
            rho_i_r, iono_i_L1, dist_i_r = calc_rho(info['sp_r'], X_apc, Y_apc, Z_apc, lat_r, lon_r, alt_r + h_r, el_i_r, az_i_r, WAVE_L1, tr)
            rho_i_r, iono_i_L5, _ = calc_rho(info['sp_r'], X_apc, Y_apc, Z_apc, lat_r, lon_r, alt_r + h_r, el_i_r, az_i_r, WAVE_L5, tr)
            
            dx_geom = [
                -(info['sp_r'][0] - X_apc) / dist_i_r - (-(rc['sp_r'][0] - X_apc) / rc['dist_r']),
                -(info['sp_r'][1] - Y_apc) / dist_i_r - (-(rc['sp_r'][1] - Y_apc) / rc['dist_r']),
                -(info['sp_r'][2] - Z_apc) / dist_i_r - (-(rc['sp_r'][2] - Z_apc) / rc['dist_r'])
            ]
            
            var_base_1 = (10.0 ** (-info['data']['C1']['snr'] / 10.0)) * 100.0
            var_base_5 = (10.0 ** (-info['data']['C5']['snr'] / 10.0)) * 100.0
            
            L.append([(info['data']['C1']['sd_P'] - rc['data']['C1']['sd_P']) - ((rho_i_r + iono_i_L1) - base_calcs[s]['P1'] - rc['SD_P1_ref'])])
            H.append(dx_geom); R_diag.append(var_base_1 * 9.0)
            
            L.append([(info['data']['C5']['sd_P'] - rc['data']['C5']['sd_P']) - ((rho_i_r + iono_i_L5) - base_calcs[s]['P5'] - rc['SD_P5_ref'])])
            H.append(dx_geom); R_diag.append(var_base_5 * 9.0)
            
            if info['data']['C1'].get('cp_r') is not None and info['data']['C5'].get('cp_r') is not None:
                cp1_obs = (info['data']['C1']['cp_r'] - info['data']['C1']['cp_b']) - (rc['data']['C1']['cp_r'] - rc['data']['C1']['cp_b'])
                cp5_obs = (info['data']['C5']['cp_r'] - info['data']['C5']['cp_b']) - (rc['data']['C5']['cp_r'] - rc['data']['C5']['cp_b'])
                
                amb_float_1 = (cp1_obs * WAVE_L1 - ((rho_i_r - iono_i_L1) - base_calcs[s]['CP1'] - rc['SD_CP1_ref'])) / WAVE_L1
                amb_float_5 = (cp5_obs * WAVE_L5 - ((rho_i_r - iono_i_L5) - base_calcs[s]['CP5'] - rc['SD_CP5_ref'])) / WAVE_L5
                
                Z_trans_1, _ = decorrelacion_lambda_z([[var_base_1 * 0.0001]])
                Z_trans_5, _ = decorrelacion_lambda_z([[var_base_5 * 0.0001]])
                
                amb_restored_1 = round(amb_float_1 * Z_trans_1[0][0]) / Z_trans_1[0][0]
                amb_restored_5 = round(amb_float_5 * Z_trans_5[0][0]) / Z_trans_5[0][0]
                
                if abs(amb_float_1 - amb_restored_1) < 0.20 and abs(amb_float_5 - amb_restored_5) < 0.20:
                    L.append([(cp1_obs * WAVE_L1 - amb_restored_1 * WAVE_L1) - ((rho_i_r - iono_i_L1) - base_calcs[s]['CP1'] - rc['SD_CP1_ref'])])
                    H.append(dx_geom); R_diag.append(var_base_1 * 0.0001)
                    
                    L.append([(cp5_obs * WAVE_L5 - amb_restored_5 * WAVE_L5) - ((rho_i_r - iono_i_L5) - base_calcs[s]['CP5'] - rc['SD_CP5_ref'])])
                    H.append(dx_geom); R_diag.append(var_base_5 * 0.0001)
                    
                    kf_estado['fix_flags'] += 1

        if not H: return None, "FAILED", kf_estado, None
        
        H_T = transpose_matrix(H)
        R_inv = matid(len(R_diag))
        for i in range(len(R_diag)): R_inv[i][i] = 1.0 / max(1e-6, R_diag[i])
        
        P_inv = invert_matrix_nxn(P_pri)
        if not P_inv: return None, "FAILED", kf_estado, None
        
        H_T_R_inv = matmul(H_T, R_inv)
        N_mat = matadd(matmul(H_T_R_inv, H), P_inv)
        U_vec = matmul(H_T_R_inv, L)
        
        Q_cov = invert_matrix_nxn(N_mat)
        if not Q_cov: return None, "FAILED", kf_estado, None
        
        Delta_X = matmul(Q_cov, U_vec)
        X_post = [[X_pri[0][0] + Delta_X[0][0]], [X_pri[1][0] + Delta_X[1][0]], [X_pri[2][0] + Delta_X[2][0]]]
        
        kf_estado['X'] = X_post
        kf_estado['P'] = Q_cov
        
        status = "FIXED (PPK DUAL)" if kf_estado['fix_flags'] > 4 else "FLOAT (DGPS DUAL)"
        kf_estado['fix_flags'] = 0 
        
        state_dict = {
            'tow': tr, 'X_pri': X_pri, 'P_pri': P_pri, 'X_post': X_post, 'P_post': Q_cov, 'tide': (dx_tide, dy_tide, dz_tide)
        }
        
        return (X_post[0][0] - dx_tide, X_post[1][0] - dy_tide, X_post[2][0] - dz_tide), status, kf_estado, state_dict

    except Exception as e:
        return None, f"FAILED_EXCEPTION:_{str(e)}", kf_estado, None

# =====================================================================
# ESTADÍSTICAS Y FILTRADO VINCULANTE (HARD FILTER)
# =====================================================================
def estadistica_desacoplada(coordenadas, conf_plani, conf_alti, err_hor_max, err_ver_max):
    if not coordenadas: return None, None, None, 0, 0, 0, 0, 0.0
    
    N_list = [c[0] for c in coordenadas]
    E_list = [c[1] for c in coordenadas]
    Z_list = [c[2] for c in coordenadas]

    def get_median(lst):
        s = sorted(lst); n = len(s)
        if n == 0: return 0
        return s[n//2] if n % 2 == 1 else (s[n//2 - 1] + s[n//2]) / 2.0

    med_N = get_median(N_list)
    med_E = get_median(E_list)
    med_Z = get_median(Z_list)
    
    valid_coords = []
    for c in coordenadas:
        dh = math.hypot(c[0] - med_N, c[1] - med_E)
        dv = abs(c[2] - med_Z)
        
        if (err_hor_max > 0.0 and dh > err_hor_max) or (err_ver_max > 0.0 and dv > err_ver_max):
            continue
        valid_coords.append(c)

    if not valid_coords: return None, None, None, 0, 0, 0, 0, 0.0
    
    N_v = [c[0] for c in valid_coords]
    E_v = [c[1] for c in valid_coords]
    Z_v = [c[2] for c in valid_coords]
    f_v = [c[3] for c in valid_coords if len(c) > 3 and "FIXED" in c[3]]

    def calc_mean_std(arr):
        n = len(arr)
        m = sum(arr) / n
        return m, (math.sqrt(sum((x - m)**2 for x in arr) / n) if n > 1 else 0.0)

    N_m, N_s = calc_mean_std(N_v)
    E_m, E_s = calc_mean_std(E_v)
    Z_m, Z_s = calc_mean_std(Z_v)
    
    N_f = [x for x in N_v if abs(x - N_m) <= conf_plani * N_s] if N_s > 0 else N_v
    E_f = [x for x in E_v if abs(x - E_m) <= conf_plani * E_s] if E_s > 0 else E_v
    Z_f = [x for x in Z_v if abs(x - Z_m) <= conf_alti * Z_s] if Z_s > 0 else Z_v

    fix_ratio = (len(f_v) / len(valid_coords)) * 100.0 if valid_coords else 0.0
    return sum(N_f)/max(1, len(N_f)), sum(E_f)/max(1, len(E_f)), sum(Z_f)/max(1, len(Z_f)), N_s, E_s, Z_s, min(len(N_f), len(E_f), len(Z_f)), fix_ratio

# =====================================================================
# GENERADORES DE INFORMES (FRONTEND)
# =====================================================================
def generar_informe_homogeneizacion_detallado(base_name, rover_name, base_raw, rover_raw, rover_sinc, modo_str, msg, c_base, c_rover):
    def get_stats(obs):
        c = {'G':0, 'E':0, 'C':0, 'R':0, 'S':0, 'J':0}
        tiempos = sorted(list(obs.keys()))
        if not tiempos: return c, 0, None, None, 0.0, 0, "Desconocida", 0, 0.0
        
        epocas = len(obs)
        t_ini, t_fin = obs[tiempos[0]]['_meta'], obs[tiempos[-1]]['_meta']
        intervalos = [tiempos[i] - tiempos[i-1] for i in range(1, epocas)]
        tasa_muestreo = sum(intervalos)/len(intervalos) if intervalos else 0.0
        gaps = sum(1 for i in intervalos if i > tasa_muestreo * 1.5)
        
        sats_unicos = set()
        tiene_l1 = False
        tiene_l5 = False
        snr_total = 0.0
        snr_count = 0
        
        for t in tiempos:
            for s, data in obs[t].items():
                if s != '_meta':
                    if s[0] in c: c[s[0]] += 1
                    sats_unicos.add(s)
                    if 'C1' in data or 'L1' in data: tiene_l1 = True
                    if 'C5' in data or 'L5' in data: tiene_l5 = True
                    if 'S1' in data and data['S1'] > 0:
                        snr_total += data['S1']
                        snr_count += 1
                    if 'S5' in data and data['S5'] > 0:
                        snr_total += data['S5']
                        snr_count += 1
                        
        tipo_senal = "L1+L5 (Doble Frecuencia)" if (tiene_l1 and tiene_l5) else ("L1 (Monofrecuencia)" if tiene_l1 else "C1 (Solo Código)")
        avg_snr = (snr_total / snr_count) if snr_count > 0 else 0.0
        total_sats = len(sats_unicos)
        
        return {k: v/epocas for k, v in c.items()}, epocas, t_ini, t_fin, tasa_muestreo, gaps, tipo_senal, total_sats, avg_snr
    
    cb, eb, b_ini, b_fin, tr_b, g_b, senal_b, sats_b, snr_b = get_stats(base_raw)
    cr, er, r_ini, r_fin, tr_r, g_r, senal_r, sats_r, snr_r = get_stats(rover_raw)
    cs, es, s_ini, s_fin, tr_s, _, senal_s, sats_s, snr_s = get_stats(rover_sinc)
    t_exito = (es / er * 100) if er > 0 else 0.0
    
    sug_iter = 4
    if es < 150: sug_iter = 8
    elif es < 300: sug_iter = 6
    elif es < 500: sug_iter = 5
    
    b_ini_str = f"{b_ini[0]}-{b_ini[1]:02d}-{b_ini[2]:02d} {b_ini[3]:02d}:{b_ini[4]:02d}:{b_ini[5]}" if b_ini else "N/A"
    b_fin_str = f"{b_fin[0]}-{b_fin[1]:02d}-{b_fin[2]:02d} {b_fin[3]:02d}:{b_fin[4]:02d}:{b_fin[5]}" if b_fin else "N/A"
    r_ini_str = f"{r_ini[0]}-{r_ini[1]:02d}-{r_ini[2]:02d} {r_ini[3]:02d}:{r_ini[4]:02d}:{r_ini[5]}" if r_ini else "N/A"
    r_fin_str = f"{r_fin[0]}-{r_fin[1]:02d}-{r_fin[2]:02d} {r_fin[3]:02d}:{r_fin[4]:02d}:{r_fin[5]}" if r_fin else "N/A"
    
    fecha_calculo = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    informe = f"""
========================================================================
    AUDITORÍA FORENSE DE EMPAREJAMIENTO DE ÉPOCAS
========================================================================
[0] TRAZABILIDAD TEMPORAL Y ESPACIAL
  [-] Fecha y Hora de Cálculo   : {fecha_calculo}
  [-] Coord. Base Fija (N,E,Z)  : {f_14(c_base['N'])}, {f_14(c_base['E'])}, {f_14(c_base['Z'])}
  [-] Coord. Rover Calib (N,E,Z): {f_14(c_rover['N'])}, {f_14(c_rover['E'])}, {f_14(c_rover['Z'])}

[1] PARÁMETROS DE CONTROL (BASE) : {base_name}
  [-] Tipo de Señal GNSS        : {senal_b}
  [-] Satélites Únicos Vistos   : {sats_b}
  [-] Potencia Promedio (SNR)   : {snr_b:.1f} dBHz
  [-] Épocas Crudas Registradas : {eb}
  [-] Ventana de Medición       : {b_ini_str} al {b_fin_str}

[2] PARÁMETROS DEL MÓVIL (ROVER) : {rover_name}
  [-] Tipo de Señal GNSS        : {senal_r}
  [-] Satélites Únicos Vistos   : {sats_r}
  [-] Potencia Promedio (SNR)   : {snr_r:.1f} dBHz
  [-] Épocas Crudas Registradas : {er}
  [-] Ventana de Medición       : {r_ini_str} al {r_fin_str}

[3] MATRIZ RESULTANTE (ESTRICTA, SIN INTERPOLACIÓN)
  [-] Épocas Útiles Sincronizadas: {es}
  [-] Tasa de Éxito sobre Rover  : {f_14(t_exito)}%
  [-] Iteraciones Geométricas Sug: {sug_iter} (Basado en densidad)

[4] ENRUTADOR AUTOMÁTICO DE CÁLCULO
  [-] Módulo Asignado           : {modo_str}
  [-] Justificación Técnica     : {msg}
========================================================================
"""
    return informe

def generar_informe_ascii(tipo, p_dict):
    estado_sol = f"HÍBRIDO PPK/EKF ({p_dict.get('fix_r', 0.0):.1f}% FIXED)" if p_dict.get('fix_r', 0.0) > 0 else 'FLOAT'
    
    if p_dict.get('estrategia') in ['MODO_B_ASINCRONO', 'MODO_B_FASE']:
        if p_dict.get('shift_applied'):
            estado_sol = 'FLOAT (DGPS IRLS + Site Calibration)'
        else:
            estado_sol = 'FLOAT (DGPS Clásico Termux)'
    elif p_dict.get('estrategia') == 'MODO_C_PPK':
        if p_dict.get('fix_r', 0.0) > 0:
            estado_sol = f"PPK DUAL FRECUENCIA L1+L5 ({p_dict.get('fix_r', 0.0):.1f}% FIXED)"
        else:
            estado_sol = 'FLOAT DUAL'
    elif p_dict.get('fix_r', 0.0) == 0.0:
        estado_sol = 'FLOAT (EKF)'
        
    err_h_str = f"± {f_14(p_dict['err_h'])} m (Vinculante)" if p_dict['err_h'] > 0 else 'Inactiva'
    err_v_str = f"± {f_14(p_dict['err_v'])} m (Vinculante)" if p_dict['err_v'] > 0 else 'Inactiva'
    sp3_str = p_dict.get('sp3_file') if p_dict.get('sp3_file') else "No provisto (Fallback a Broadcast NAV)"
    nav_str = p_dict.get('nav_file', "auto_nav.nav")
    
    fecha_calculo = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    shift_bloque = ""
    if p_dict.get('shift_applied'):
        shift_bloque = f"""
[4] COMPENSACIÓN DE SITIO EMPÍRICA (SITE CALIBRATION)
------------------------------------------------------------------------
  [-] Sesgo Norte Descontado : {f_14(p_dict.get('bias_n', 0.0))} m
  [-] Sesgo Este Descontado  : {f_14(p_dict.get('bias_e', 0.0))} m
  [-] Sesgo Cota Descontado  : {f_14(p_dict.get('bias_z', 0.0))} m
"""
    
    informe = f"""
========================================================================
             INFORME DE PROCESAMIENTO GNSSJP PRO (V19)
========================================================================

[*] RESULTADO DE MEDICIÓN ABSOLUTA ({estado_sol})
------------------------------------------------------------------------
  [-] Fecha y Hora de Cálculo: {fecha_calculo}
  [-] Tolerancia Horizontal  : {err_h_str}
  [-] Tolerancia Vertical    : {err_v_str}
  [-] Máscara Elevación      : {f_14(p_dict['mask'])}°
  [-] Filtro Planimétrico    : {f_14(p_dict['cp'])} Sigma
  [-] Filtro Altimétrico     : {f_14(p_dict['ca'])} Sigma
  [-] Tolerancia Sync        : {f_14(p_dict.get('max_gap', 0.5))} s
  [-] Tasa Ambiguity FIX     : {p_dict.get('fix_r', 0.0):.2f}% (Resolución Entera)
  [-] Épocas Útiles Retenidas: {p_dict['ret']} ({(p_dict['ret']/max(1, p_dict['total']))*100:.2f}% del total)
  [-] Motor Matemático Activo: {p_dict.get('estrategia', 'Módulo A (EKF Homogéneo)')}

[1] TRAZABILIDAD DEL PROYECTO Y ARCHIVOS
------------------------------------------------------------------------
  [-] Archivo Control (Base) : {p_dict['base_file']}
  [-] Archivo Móvil (Rover)  : {p_dict['rover_file']}
  [-] Archivo Efemérides NAV : {nav_str}
  [-] Archivo Preciso SP3    : {sp3_str}

[2] CALIDAD GEOMÉTRICA (QA / QC)
------------------------------------------------------------------------
  [-] Error Horizontal (RMS) : ± {f_14(math.hypot(p_dict['std_n'], p_dict['std_e']))} m
  [-] Error Espacial (3D RMS): ± {f_14(math.sqrt(p_dict['std_n']**2 + p_dict['std_e']**2 + p_dict['std_z']**2))} m

[3] RESULTADOS VECTORIALES FINALES
------------------------------------------------------------------------
  * COORDENADA DE CONTROL (BASE FIJA):
      Norte : {f_14(p_dict['b_n'])} m
      Este  : {f_14(p_dict['b_e'])} m
      Cota  : {f_14(p_dict['b_z'])} m

  * COORDENADA CALCULADA (AJUSTE {estado_sol}):
      Norte : {f_14(p_dict['r_n_calc'])} m
      Este  : {f_14(p_dict['r_e_calc'])} m
      Cota  : {f_14(p_dict['r_z_calc'])} m
{shift_bloque}========================================================================
"""
    return informe

# =====================================================================
# RUTAS FLASK (ENRUTADOR AUTÓNOMO Y TRIPLE VÍA AISLADA)
# =====================================================================
@app.route('/')
def index():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_path = os.path.join(base_dir, 'index.html')
    return send_file(index_path)

@app.route('/API/tab1_homogenizar', methods=['POST'])
def tab1_homogenizar():
    with STATE_LOCK:
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    url_base = request.form.get('url_base')
    url_rover = request.form.get('url_rover')
    
    utm_n = safe_f(request.form.get('utm_norte'), 0.0)
    utm_e = safe_f(request.form.get('utm_este'), 0.0)
    utm_c = safe_f(request.form.get('utm_cota'), 0.0)
    utm_h = safe_i(request.form.get('utm_huso'), 19)
    utm_hem = request.form.get('utm_hemisferio', 'N')
    
    utm_n_r = safe_f(request.form.get('utm_norte_r'), 0.0)
    utm_e_r = safe_f(request.form.get('utm_este_r'), 0.0)
    utm_c_r = safe_f(request.form.get('utm_cota_r'), 0.0)
    
    h_b = safe_f(request.form.get('altura_base'), 0.0)
    h_r = safe_f(request.form.get('altura_rover'), 0.0)

    guardar_estado('utm_norte', utm_n)
    guardar_estado('utm_este', utm_e)
    guardar_estado('utm_cota', utm_c)
    guardar_estado('utm_huso', utm_h)
    guardar_estado('utm_hemisferio', utm_hem)
    guardar_estado('utm_norte_r', utm_n_r)
    guardar_estado('utm_este_r', utm_e_r)
    guardar_estado('utm_cota_r', utm_c_r)
    guardar_estado('altura_base', h_b)
    guardar_estado('altura_rover', h_r)
    
    if not url_base or not url_rover: 
        return Response("> [ERROR CRÍTICO] Enlaces de Google Drive faltantes.\n", mimetype='text/plain')
    
    p_b_raw = os.path.join(UPLOAD_FOLDER, 'base_raw.obs')
    p_r_raw = os.path.join(UPLOAD_FOLDER, 'rover_calibracion_raw.obs')

    def procesar():
        try:
            yield "> [RED] Descargando RINEX Base desde Google Drive...\n"
            descargar_desde_gdrive(url_base, p_b_raw)
            yield "> [RED] Descargando RINEX Rover desde Google Drive...\n"
            descargar_desde_gdrive(url_rover, p_r_raw)

            yield f"\n> [SISTEMA] Iniciando Etapa 1: Emparejamiento Base Pivote y Rover de Calibración...\n"
            base_raw_dict = parse_rinex_obs_completo(p_b_raw)
            rover_raw_dict = parse_rinex_obs_completo(p_r_raw)
            
            yield "> [ENRUTADOR] Evaluando calidad de señales y sincronía geométrica...\n"
            modo_str, ratio, msg = analizar_calidad_y_senales_rinex(base_raw_dict, rover_raw_dict, max_gap_tolerado=0.5)
            yield f"  [-] Módulo pre-asignado: {modo_str}\n"
            yield f"  [-] Justificación: {msg}\n\n"
            
            base_sinc, rover_sinc = {}, {}
            total_epochs = len(rover_raw_dict)
            c = 0
            for tr in sorted(list(rover_raw_dict.keys())):
                c += 1
                if total_epochs > 0 and c % max(1, total_epochs // 10) == 0: 
                    yield f"[PROGRESO] Cotejando épocas sin distorsión... {int((c / total_epochs) * 100)}%\n"
                
                base_interp = interpolar_base_a_rover(base_raw_dict, tr, max_gap=float('inf'))
                
                if base_interp:
                    base_sinc[tr] = base_interp
                    base_sinc[tr]['_meta'] = rover_raw_dict[tr]['_meta']
                    rover_sinc[tr] = rover_raw_dict[tr]
            
            if not base_sinc: yield "\n> [ERROR FATAL] Cero épocas en común. Revisar rango horario."; return
            p_b_h = os.path.join(UPLOAD_FOLDER, 'base_calib_homo.obs')
            p_r_h = os.path.join(UPLOAD_FOLDER, 'rover_calib_homo.obs')
            generar_rinex_sincronizado(p_b_raw, p_b_h, base_sinc)
            generar_rinex_sincronizado(p_r_raw, p_r_h, rover_sinc)
            
            guardar_estado('base_raw', p_b_raw)
            guardar_estado('base_calib_homo', p_b_h)
            guardar_estado('rover_calib_homo', p_r_h)
            
            name_base = "Drive_Base_Pivote.obs"
            name_rover = "Drive_Rover_Calib.obs"
            guardar_estado('name_base_raw', name_base)
            guardar_estado('name_rover_calib_raw', name_rover)
            
            c_base = {'N': utm_n, 'E': utm_e, 'Z': utm_c}
            c_rover = {'N': utm_n_r, 'E': utm_e_r, 'Z': utm_c_r}
            
            yield generar_informe_homogeneizacion_detallado(name_base, name_rover, base_raw_dict, rover_raw_dict, rover_sinc, modo_str, msg, c_base, c_rover)
            yield "\n[SUCCESS]"
        except Exception as e: yield f"\n> [ERROR] Falla estructural: {str(e)}"
    return Response(procesar(), mimetype='text/plain')

@app.route('/API/tab2_efemerides', methods=['POST'])
def tab2_efemerides():
    f_sp3 = request.files.get('file_sp3')
    
    sp3_path = None
    if f_sp3 and f_sp3.filename != '':
        sp3_path = os.path.join(UPLOAD_FOLDER, 'manual_sp3.sp3')
        f_sp3.save(sp3_path)
        guardar_estado('sp3_path', sp3_path)
        guardar_estado('name_sp3_file', f_sp3.filename)
    else:
        guardar_estado('sp3_path', None)
        guardar_estado('name_sp3_file', None)

    def procesar():
        try:
            yield "> [SISTEMA] Iniciando Inyección Híbrida de Efemérides...\n"
            if sp3_path: yield f"  [-] Archivo SP3 Preciso cargado manualmente: {f_sp3.filename}\n"
            else: yield "  [!] No se detectó archivo SP3 manual. Se usará solo Broadcast NAV.\n"

            yield "\n> [RED] Conectando con Red Global de Repositorios GNSS...\n"
            bp = leer_estado('base_raw')
            if not bp or not os.path.exists(bp): 
                yield "> [ERROR FATAL] Falta RINEX Base en memoria para extraer fecha.\n"; return
            
            ft = obtener_fecha_obs(bp)
            if not ft: yield "> [ERROR FATAL] Imposible extraer la fecha del RINEX Base.\n"; return
            
            year, month, day = ft[0], ft[1], ft[2]
            dt = datetime.datetime(year, month, day)
            doy = dt.timetuple().tm_yday
            yy = str(year)[-2:]
            
            nav_gz = os.path.join(UPLOAD_FOLDER, f"auto_nav_{year}_{doy:03d}.nav.gz")
            nav_path = os.path.join(UPLOAD_FOLDER, f"auto_nav_{year}_{doy:03d}.nav")
            
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            if not os.path.exists(nav_path):
                urls_to_try = [
                    f"https://garner.ucsd.edu/pub/rinex/{year}/{doy:03d}/brdc{doy:03d}0.{yy}n.gz",
                    f"http://igs.bkg.bund.de/root_ftp/IGS/BRDC/{year}/{doy:03d}/brdc{doy:03d}0.{yy}n.gz",
                    f"https://www.epncb.oma.be/ftp/obs/BRDC/{year}/{doy:03d}/brdc{doy:03d}0.{yy}n.gz",
                    f"http://igs.bkg.bund.de/root_ftp/IGS/BRDC/{year}/{doy:03d}/BRDC00IGS_R_{year}{doy:03d}0000_01D_MN.rnx.gz"
                ]
                
                descargado = False
                for url_nav in urls_to_try:
                    try:
                        protocolo = "HTTP PURO" if url_nav.startswith("http://") else "HTTPS"
                        yield f"  [-] Intentando espejo ({protocolo}): {url_nav.split('/')[-1]}...\n"
                        req = urllib.request.Request(url_nav, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(req, context=ctx, timeout=5) as res:
                            with open(nav_gz, 'wb') as f: f.write(res.read())
                        descargado = True
                        yield f"  [+] Descarga exitosa verificada mediante {protocolo}.\n"
                        break 
                    except Exception as e:
                        err_limpio = str(e).replace('<', '[').replace('>', ']')
                        yield f"  [!] Falló ({err_limpio}). Saltando a siguiente servidor...\n"
                        continue
                
                if not descargado:
                    raise Exception("HTTP 404/Timeout Total: La red IGS global bloqueó la conexión o el archivo no existe.")
                
                yield "  [-] Descomprimiendo archivo NAV...\n"
                with gzip.open(nav_gz, 'rb') as f_in, open(nav_path, 'wb') as f_out: 
                    shutil.copyfileobj(f_in, f_out)
                
                if os.path.exists(nav_gz): os.remove(nav_gz)
            
            guardar_estado('nav_path', nav_path)
            guardar_estado('name_nav_file', os.path.basename(nav_path))
            yield f"  [-] Archivo NAV listo y ensamblado en memoria.\n\n[SUCCESS]"
        except Exception as e:
            yield f"\n> [ERROR FATAL] Fallo en descarga automática NAV: {str(e)}\n"

    return Response(procesar(), mimetype='text/plain')

@app.route('/API/tab3_calibrar', methods=['POST'])
def tab3_calibrar():
    utm_n = leer_estado('utm_norte')
    utm_e = leer_estado('utm_este')
    utm_c = leer_estado('utm_cota')
    utm_h = leer_estado('utm_huso')
    utm_hem = leer_estado('utm_hemisferio')
    utm_n_r = leer_estado('utm_norte_r')
    utm_e_r = leer_estado('utm_este_r')
    utm_c_r = leer_estado('utm_cota_r')
    h_b = leer_estado('altura_base')
    h_r = leer_estado('altura_rover')

    p_max_gap = safe_f(request.form.get('param_max_gap'), 0.5)
    p_snr = safe_f(request.form.get('param_snr'), 25.0)
    p_iter = safe_i(request.form.get('param_iter'), 6)
    p_iter = max(1, p_iter) 

    def procesar():
        try:
            if utm_e == 0.0 or utm_n == 0.0 or utm_n_r == 0.0 or utm_e_r == 0.0: 
                yield "> [ERROR] Coordenadas Base y Rover no inyectadas correctamente.\n"; return
            
            nav_path = leer_estado('nav_path')
            sp3_path = leer_estado('sp3_path')
            p_b_h = leer_estado('base_calib_homo')
            p_r_h = leer_estado('rover_calib_homo')

            if not nav_path or not p_b_h or not p_r_h: 
                yield "> [ERROR FATAL] Faltan archivos. Ve a la Pestaña 2.\n"; return

            obs_b_raw = parse_rinex_obs_completo(p_b_h)
            obs_r_raw = parse_rinex_obs_completo(p_r_h)
            nav = parse_rinex_nav_real(nav_path)
            sp3 = parse_sp3_preciso(sp3_path) if sp3_path else {}
            
            modo_str, ratio, msg = analizar_calidad_y_senales_rinex(obs_b_raw, obs_r_raw, max_gap_tolerado=p_max_gap)
            yield f"> [ENRUTADOR] Análisis completado: {msg}\n"
            
            if modo_str == "MODO_C_SPP":
                yield "\n> [ERROR] Operación Abortada: El Enrutador detectó MODO C (SPP). Requiere implementación geométrica dedicada.\n"
                return
            
            lat_b, lon_b, _ = utm_a_geodesicas(utm_e, utm_n, utm_h, utm_hem)
            X_b, Y_b, Z_b = geodesicas_a_ecef(lat_b, lon_b, utm_c + h_b)
            X_bg, Y_bg, Z_bg = geodesicas_a_ecef(lat_b, lon_b, utm_c)

            if modo_str in ["MODO_A_CODIGO", "MODO_C_PPK"]:
                yield f"> [SISTEMA] Iniciando Búsqueda Determinista | {modo_str}...\n"
                
                if modo_str == "MODO_A_CODIGO":
                    sd_suavizada = aislar_diferencias_simples_ppk(obs_b_raw, obs_r_raw)
                else:
                    sd_suavizada = aislar_diferencias_MODO_C(obs_b_raw, obs_r_raw)
                    
                if not sd_suavizada: yield "> [ERROR] No hay épocas sincronizadas válidas.\n"; return
                t_sample = list(sd_suavizada.keys())
                
                yield "[PROGRESO] Fase 1: Extracción de Límites (Pre-Scan EKF)...\n"
                P_init = matid(3)
                for i in range(3): P_init[i][i] = 100.0
                kf_estado_raw = {'X': [[X_bg], [Y_bg], [Z_bg]], 'P': P_init, 'X_base': (X_b, Y_b, Z_b), 'fix_flags': 0, 'h_r': h_r}
                coords_raw = []
                
                for t in t_sample:
                    if modo_str == "MODO_A_CODIGO":
                        sem, status, kf_estado_raw, _ = procesar_ekF_lambda(sd_suavizada[t], nav, sp3, kf_estado_raw, t, 10.0, p_snr)
                    else:
                        sem, status, kf_estado_raw, _ = procesar_ekF_PPK_L1_L5(sd_suavizada[t], nav, sp3, kf_estado_raw, t, 10.0, p_snr)
                        
                    if sem:
                        la, lo, al = ecef_a_geodesicas(sem[0], sem[1], sem[2])
                        nt, et = geodesicas_a_utm(la, lo, utm_h)
                        coords_raw.append((nt, et, al, status))
                
                if not coords_raw: yield "> [ERROR] Filtro de Kalman colapsado.\n"; return
                
                deltas_h = sorted([math.hypot(c[0] - utm_n_r, c[1] - utm_e_r) for c in coords_raw])
                deltas_v = sorted([abs(c[2] - utm_c_r) for c in coords_raw])
                idx_optimo = max(1, len(deltas_h) // 10)
                best_eh = max(0.01, float(deltas_h[idx_optimo]) * 1.5)
                best_ev = max(0.01, float(deltas_v[idx_optimo]) * 1.5)
                
                yield f"  [*] Límite Horizontal EKF Inyectado: {f_14(best_eh)} m\n"
                yield f"  [*] Límite Vertical EKF Inyectado: {f_14(best_ev)} m\n\n"
                
                yield f"[PROGRESO] Fase 2: Malla Pentadimensional EKF (Iteraciones: {p_iter})...\n"
                global_best_score = float('inf')
                best_rmse = float('inf')
                best_params = {}
                
                m_center, m_span = 10.0, 5.0
                cp_center, cp_span = 2.0, 1.5
                ca_center, ca_span = 2.0, 1.5
                snr_center, snr_span = p_snr, 5.0
                gap_center, gap_span = p_max_gap, 0.02
                
                p_b_raw = leer_estado('base_raw')
                p_r_raw = os.path.join(UPLOAD_FOLDER, 'rover_calibracion_raw.obs')
                obs_b_full = parse_rinex_obs_completo(p_b_raw) if p_b_raw and os.path.exists(p_b_raw) else obs_b_raw
                obs_r_full = parse_rinex_obs_completo(p_r_raw) if os.path.exists(p_r_raw) else obs_r_raw
                rover_tows_full = sorted(list(obs_r_full.keys()))
                base_tows_full = sorted(list(obs_b_full.keys()))
                
                for nivel in range(p_iter):
                    yield f"  [+] Refinando espacio de búsqueda (Zoom {nivel+1}/{p_iter})...\n"
                    m_grid = [max(1.0, min(25.0, x)) for x in [m_center - m_span, m_center, m_center + m_span]]
                    cp_grid = [max(0.1, min(5.0, x)) for x in [cp_center - cp_span, cp_center, cp_center + cp_span]]
                    ca_grid = [max(0.1, min(5.0, x)) for x in [ca_center - ca_span, ca_center, ca_center + ca_span]]
                    snr_grid = [max(25.0, min(45.0, x)) for x in [snr_center - snr_span, snr_center, snr_center + snr_span]]
                    gap_grid = [max(0.01, min(0.05, x)) for x in [gap_center - gap_span, gap_center, gap_center + gap_span]]
                    
                    nivel_best_rmse = float('inf')
                    nivel_best_params = {}
                    
                    for gap in set(gap_grid):
                        obs_b_sync = {}
                        for tr in rover_tows_full:
                            if not base_tows_full: continue
                            idx = min(range(len(base_tows_full)), key=lambda i: abs(base_tows_full[i] - tr))
                            if abs(base_tows_full[idx] - tr) <= gap:
                                obs_b_sync[tr] = obs_b_full[base_tows_full[idx]].copy()
                                obs_b_sync[tr]['_meta'] = obs_r_full[tr]['_meta']
                                
                        if modo_str == "MODO_A_CODIGO":
                            sd_suav = aislar_diferencias_simples_ppk(obs_b_sync, obs_r_full)
                        else:
                            sd_suav = aislar_diferencias_MODO_C(obs_b_sync, obs_r_full)
                            
                        t_samp = list(sd_suav.keys())
                        if not sd_suav: continue
                        
                        for m in set(m_grid):
                            for snr in set(snr_grid):
                                kf_est = {'X': [[X_bg], [Y_bg], [Z_bg]], 'P': P_init, 'X_base': (X_b, Y_b, Z_b), 'fix_flags': 0, 'h_r': h_r}
                                coords = []
                                for t in t_samp:
                                    if modo_str == "MODO_A_CODIGO":
                                        sem, status, kf_est, _ = procesar_ekF_lambda(sd_suav[t], nav, sp3, kf_est, t, m, snr)
                                    else:
                                        sem, status, kf_est, _ = procesar_ekF_PPK_L1_L5(sd_suav[t], nav, sp3, kf_est, t, m, snr)
                                        
                                    if sem:
                                        la, lo, al = ecef_a_geodesicas(sem[0], sem[1], sem[2])
                                        nt, et = geodesicas_a_utm(la, lo, utm_h)
                                        coords.append((nt, et, al, status))
                                
                                if not coords: continue
                                
                                for cp in set(cp_grid):
                                    for ca in set(ca_grid):
                                        res = estadistica_desacoplada(coords, cp, ca, best_eh, best_ev)
                                        if res[0] is None: continue
                                        nf, ef, zf, std_n, std_e, std_z, ret, fix_ratio = res
                                        
                                        rmse_3d = math.sqrt((nf - utm_n_r)**2 + (ef - utm_e_r)**2 + (zf - utm_c_r)**2)
                                        score = (rmse_3d ** 3) * (1.0 + gap * 0.05) * (1.0 + (1.0 - (fix_ratio/100.0)) * 2.0)
                                        
                                        if score < nivel_best_rmse:
                                            nivel_best_rmse = score
                                            nivel_best_params = {'m': m, 'snr': snr, 'gap': gap, 'cp': cp, 'ca': ca, 'rmse': rmse_3d}
                                            if score < global_best_score:
                                                global_best_score = score
                                                best_rmse = rmse_3d
                                                best_params = {'mask': m, 'cp': cp, 'ca': ca, 'eh': best_eh, 'ev': best_ev, 'max_gap': gap, 'snr': snr, 'rmse': rmse_3d, 'ret': ret, 'dn': nf - utm_n_r, 'de': ef - utm_e_r, 'dz': zf - utm_c_r}
                    
                    if nivel_best_rmse != float('inf'):
                        yield f"  [*] Fin Iteración {nivel+1} | Mejor RMSE Local: {f_14(nivel_best_params['rmse'])} m\n"
                    
                    if global_best_score != float('inf'):
                        m_center, m_span = best_params['mask'], m_span / 2.0
                        cp_center, cp_span = best_params['cp'], cp_span / 2.0
                        ca_center, ca_span = best_params['ca'], ca_span / 2.0
                        snr_center, snr_span = best_params['snr'], snr_span / 2.0
                        gap_center, gap_span = best_params['max_gap'], gap_span / 2.0
                    else:
                        m_span /= 2.0; cp_span /= 2.0; ca_span /= 2.0; snr_span /= 2.0; gap_span /= 2.0

            else:
                yield f"> [SISTEMA] Iniciando Búsqueda Determinista | MÓDULO B (IRLS Clásico Termux)...\n"
                p_b_raw = leer_estado('base_raw')
                p_r_raw = os.path.join(UPLOAD_FOLDER, 'rover_calibracion_raw.obs')
                obs_b_full = parse_rinex_obs_completo(p_b_raw) if p_b_raw and os.path.exists(p_b_raw) else obs_b_raw
                obs_r_full = parse_rinex_obs_completo(p_r_raw) if os.path.exists(p_r_raw) else obs_r_raw

                rover_tows = sorted(list(obs_r_full.keys()))
                base_tows = sorted(list(obs_b_full.keys()))
                obs_b_sync = {}
                for tr in rover_tows:
                    if not base_tows: continue
                    idx = min(range(len(base_tows)), key=lambda i: abs(base_tows[i] - tr))
                    if abs(base_tows[idx] - tr) <= p_max_gap:
                        obs_b_sync[tr] = obs_b_full[base_tows[idx]].copy()
                        obs_b_sync[tr]['_meta'] = obs_r_full[tr]['_meta']

                sd_suavizada = aislar_diferencias_MODO_B(obs_b_sync, obs_r_full)
                if not sd_suavizada: yield "> [ERROR] No hay épocas sincronizadas válidas.\n"; return
                t_sample = list(sd_suavizada.keys())
                
                yield "[PROGRESO] Fase 1: Extracción de Límites (Pre-Scan Clásico IRLS)...\n"
                coords_raw = []
                for t in t_sample:
                    sem, status = calcular_IRLS_MODO_B(sd_suavizada[t], nav, X_b, Y_b, Z_b, t, 10.0)
                    if sem:
                        la, lo, al = ecef_a_geodesicas(sem[0], sem[1], sem[2])
                        nt, et = geodesicas_a_utm(la, lo, utm_h)
                        coords_raw.append((nt, et, al, status))
                
                if not coords_raw: yield "> [ERROR] Nube de puntos bruta colapsada.\n"; return
                
                deltas_h = sorted([math.hypot(c[0] - utm_n_r, c[1] - utm_e_r) for c in coords_raw])
                deltas_v = sorted([abs(c[2] - utm_c_r) for c in coords_raw])
                idx_optimo = max(1, len(deltas_h) // 10)
                best_eh = max(0.01, float(deltas_h[idx_optimo]) * 1.5)
                best_ev = max(0.01, float(deltas_v[idx_optimo]) * 1.5)
                
                yield f"  [*] Límite Horizontal IRLS Inyectado: {f_14(best_eh)} m\n"
                yield f"  [*] Límite Vertical IRLS Inyectado: {f_14(best_ev)} m\n\n"
                
                yield f"[PROGRESO] Fase 2: Malla Tridimensional Clásica (Iteraciones: {p_iter})...\n"
                global_best_score = float('inf')
                best_rmse = float('inf')
                best_params = {}
                
                m_center, m_span = 10.0, 5.0
                cp_center, cp_span = 2.0, 1.5
                ca_center, ca_span = 2.0, 1.5
                
                for nivel in range(p_iter):
                    yield f"  [+] Refinando espacio de búsqueda (Zoom {nivel+1}/{p_iter})...\n"
                    m_grid = [max(5.0, min(15.0, x)) for x in [m_center - m_span, m_center, m_center + m_span]]
                    cp_grid = [max(0.1, min(5.0, x)) for x in [cp_center - cp_span, cp_center, cp_center + cp_span]]
                    ca_grid = [max(0.1, min(5.0, x)) for x in [ca_center - ca_span, ca_center, ca_center + ca_span]]
                    
                    nivel_best_rmse = float('inf')
                    nivel_best_params = {}
                    
                    for m in set(m_grid):
                        coords = []
                        for t in t_sample:
                            sem, status = calcular_IRLS_MODO_B(sd_suavizada[t], nav, X_b, Y_b, Z_b, t, m)
                            if sem:
                                la, lo, al = ecef_a_geodesicas(sem[0], sem[1], sem[2])
                                nt, et = geodesicas_a_utm(la, lo, utm_h)
                                coords.append((nt, et, al, status))
                        
                        if not coords: continue
                        
                        for cp in set(cp_grid):
                            for ca in set(ca_grid):
                                res = estadistica_desacoplada(coords, cp, ca, best_eh, best_ev)
                                if res[0] is None: continue
                                nf, ef, zf, std_n, std_e, std_z, ret, fix_ratio = res
                                rmse_3d = math.sqrt((nf - utm_n_r)**2 + (ef - utm_e_r)**2 + (zf - utm_c_r)**2)
                                
                                if rmse_3d < nivel_best_rmse:
                                    nivel_best_rmse = rmse_3d
                                    nivel_best_params = {'m': m, 'cp': cp, 'ca': ca, 'rmse': rmse_3d}
                                    if rmse_3d < global_best_score:
                                        global_best_score = rmse_3d
                                        best_rmse = rmse_3d
                                        best_params = {'mask': m, 'cp': cp, 'ca': ca, 'eh': best_eh, 'ev': best_ev, 'max_gap': p_max_gap, 'snr': p_snr, 'rmse': rmse_3d, 'ret': ret, 'dn': nf - utm_n_r, 'de': ef - utm_e_r, 'dz': zf - utm_c_r}
                    
                    if nivel_best_rmse != float('inf'):
                        yield f"  [*] Fin Iteración {nivel+1} | Mejor RMSE Local: {f_14(nivel_best_params['rmse'])} m\n"
                        
                    if global_best_score != float('inf'):
                        m_center, m_span = best_params['mask'], m_span / 2.0
                        cp_center, cp_span = best_params['cp'], cp_span / 2.0
                        ca_center, ca_span = best_params['ca'], ca_span / 2.0
                    else:
                        m_span /= 2.0; cp_span /= 2.0; ca_span /= 2.0
            
            if best_rmse != float('inf'):
                guardar_estado('opt_mask', best_params['mask'])
                guardar_estado('opt_cp', best_params['cp'])
                guardar_estado('opt_ca', best_params['ca'])
                guardar_estado('opt_max_gap', best_params.get('max_gap', p_max_gap))
                guardar_estado('opt_snr', best_params.get('snr', p_snr))
                guardar_estado('opt_eh', best_params['eh'])
                guardar_estado('opt_ev', best_params['ev'])
                
                guardar_estado('opt_bias_n', best_params['dn'])
                guardar_estado('opt_bias_e', best_params['de'])
                guardar_estado('opt_bias_z', best_params['dz'])
                
                guardar_estado('estrategia_activa', modo_str)
                
                fecha_calculo = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                yield "\n========================================================\n"
                yield f"      [INFORME] PARÁMETROS ÓPTIMOS ({modo_str})\n"
                yield "========================================================\n"
                yield f"  [-] Fecha y Hora de Cálculo: {fecha_calculo}\n"
                yield f"  [-] Coord. Base Fija (N,E,Z): {f_14(utm_n)}, {f_14(utm_e)}, {f_14(utm_c)}\n"
                yield f"  [-] Coord. Rover Cal (N,E,Z): {f_14(utm_n_r)}, {f_14(utm_e_r)}, {f_14(utm_c_r)}\n"
                yield "--------------------------------------------------------\n"
                yield f"  [-] Tolerancia Sync (max_gap): {f_14(best_params.get('max_gap', p_max_gap))}\n"
                if modo_str in ["MODO_A_CODIGO", "MODO_C_PPK"]: 
                    yield f"  [-] Máscara SNR (dBHz): {f_14(best_params.get('snr', p_snr))}\n"
                yield f"  [-] Máscara Elevación (°): {f_14(best_params['mask'])}\n"
                yield f"  [-] Filtro Sigma Plan (cp): {f_14(best_params['cp'])}\n"
                yield f"  [-] Filtro Sigma Alt (ca): {f_14(best_params['ca'])}\n"
                yield f"  [-] Error Permitido Horizontal (m): {f_14(best_params['eh'])}\n"
                yield f"  [-] Error Permitido Vertical (m): {f_14(best_params['ev'])}\n"
                yield "--------------------------------------------------------\n"
                yield f"  [*] Menor Distancia 3D al Punto: {f_14(best_params['rmse'])} m\n"
                yield f"  [*] Deltas Residuales -> N: {f_14(best_params['dn'])}m, E: {f_14(best_params['de'])}m, Z: {f_14(best_params['dz'])}m\n"
                yield f"  [*] Épocas Retenidas: {best_params['ret']}\n"
                yield "========================================================\n"
                yield "\n[SUCCESS]"
            else:
                yield "\n> [ERROR] El modelo no convergió. Filtros demasiado agresivos.\n"
        except Exception as e: yield f"\n> [ERROR FATAL] {str(e)}"
    return Response(procesar(), mimetype='text/plain')

@app.route('/API/tab4_procesar', methods=['POST'])
def tab4_procesar():
    utm_n = leer_estado('utm_norte')
    utm_e = leer_estado('utm_este')
    utm_c = leer_estado('utm_cota')
    utm_h = leer_estado('utm_huso')
    utm_hem = leer_estado('utm_hemisferio')
    h_b = leer_estado('altura_base')
    h_r = leer_estado('altura_rover')
    
    p_mask = leer_estado('opt_mask')
    p_cp = leer_estado('opt_cp')
    p_ca = leer_estado('opt_ca')
    err_hor_max = leer_estado('opt_eh')
    err_ver_max = leer_estado('opt_ev')
    p_max_gap = leer_estado('opt_max_gap')
    p_snr = leer_estado('opt_snr')
    estrategia = leer_estado('estrategia_activa')

    url_rover_nuevo = request.form.get('url_rover_nuevo')
    
    if p_mask is None or utm_n is None:
        return Response("> [ERROR FATAL] Parámetros o coordenadas no encontrados. Ejecute la Pestaña 3 primero.\n", mimetype='text/plain')

    if not url_rover_nuevo or url_rover_nuevo.strip() == '': 
        return Response("> [ERROR] Falta el enlace de Drive del nuevo archivo RINEX Rover.\n", mimetype='text/plain')

    p_r_nuevo = os.path.join(UPLOAD_FOLDER, 'rover_nuevo_raw.obs')

    def procesar():
        try:
            yield "> [RED] Descargando Nuevo RINEX Rover desde Google Drive...\n"
            descargar_desde_gdrive(url_rover_nuevo, p_r_nuevo)
            rf_nuevo_filename = "Drive_Nuevo_Rover.obs"
            
            nav_path = leer_estado('nav_path')
            sp3_path = leer_estado('sp3_path')
            p_b_raw = leer_estado('base_raw') 

            if not nav_path or not p_b_raw or not os.path.exists(p_b_raw): 
                yield "> [ERROR FATAL] Falta archivo RINEX Base original o Efemérides.\n"; return

            obs_b_raw = parse_rinex_obs_completo(p_b_raw)
            obs_r_raw = parse_rinex_obs_completo(p_r_nuevo) 
            nav = parse_rinex_nav_real(nav_path)
            sp3 = parse_sp3_preciso(sp3_path) if sp3_path else {}
            
            modo_str, ratio, msg = analizar_calidad_y_senales_rinex(obs_b_raw, obs_r_raw, max_gap_tolerado=p_max_gap)
            yield f"> [ENRUTADOR] Análisis completado: {msg}\n"
            
            if modo_str == "MODO_C_SPP":
                yield "\n> [ERROR] Operación Abortada: El Enrutador detectó MODO C (SPP). Requiere implementación geométrica dedicada.\n"
                return
            
            lat_b, lon_b, _ = utm_a_geodesicas(utm_e, utm_n, utm_h, utm_hem)
            X_b, Y_b, Z_b = geodesicas_a_ecef(lat_b, lon_b, utm_c + h_b)
            X_bg, Y_bg, Z_bg = geodesicas_a_ecef(lat_b, lon_b, utm_c)

            if modo_str in ["MODO_A_CODIGO", "MODO_C_PPK"]:
                yield f"\n> [SISTEMA] Iniciando Procesamiento DGPS | {modo_str}...\n"
                if sp3: yield "[PROGRESO] Órbitas Precisas SP3 acopladas con éxito...\n"
                
                rover_tows = sorted(list(obs_r_raw.keys()))
                base_tows = sorted(list(obs_b_raw.keys()))
                obs_b_sync = {}
                for tr in rover_tows:
                    if not base_tows: continue
                    idx = min(range(len(base_tows)), key=lambda i: abs(base_tows[i] - tr))
                    if abs(base_tows[idx] - tr) <= p_max_gap:
                        obs_b_sync[tr] = obs_b_raw[base_tows[idx]].copy()
                        obs_b_sync[tr]['_meta'] = obs_r_raw[tr]['_meta']
                
                yield "[PROGRESO] Extrayendo Observables PPK...\n"
                if modo_str == "MODO_A_CODIGO":
                    sd_suavizada = aislar_diferencias_simples_ppk(obs_b_sync, obs_r_raw)
                else:
                    sd_suavizada = aislar_diferencias_MODO_C(obs_b_sync, obs_r_raw)
                    
                if not sd_suavizada: yield "\n> [ERROR] No hay épocas sincronizadas válidas.\n"; return
                
                yield "[PROGRESO] Fase 1: Pasada Forward EKF + Mareas Sólidas...\n"
                P_init = matid(3)
                for i in range(3): P_init[i][i] = 100.0
                kf_est = {'X': [[X_bg], [Y_bg], [Z_bg]], 'P': P_init, 'X_base': (X_b, Y_b, Z_b), 'fix_flags': 0, 'h_r': h_r}
                fwd_states = []
                t_eps = len(sd_suavizada); c = 0
                
                for t in sd_suavizada:
                    c += 1
                    if t_eps > 0 and c % max(1, t_eps // 10) == 0: 
                        yield f"[PROGRESO] Propagando Matriz Covarianza... {int((c / t_eps) * 100)}%\n"
                        
                    if modo_str == "MODO_A_CODIGO":
                        sem, status, kf_est, st_dict = procesar_ekF_lambda(sd_suavizada[t], nav, sp3, kf_est, t, p_mask, p_snr)
                    else:
                        sem, status, kf_est, st_dict = procesar_ekF_PPK_L1_L5(sd_suavizada[t], nav, sp3, kf_est, t, p_mask, p_snr)
                        
                    if sem and st_dict:
                        st_dict['status'] = status
                        fwd_states.append(st_dict)

                if not fwd_states: yield "\n> [ERROR] Colapso total del Filtro Kalman.\n"; return
                
                yield "[PROGRESO] Fase 2: Aplicando Suavizador RTS Bidireccional...\n"
                sm_states = suavizador_rts_backward(fwd_states)
                
                coords = []
                for i in range(len(sm_states)):
                    dx_t, dy_t, dz_t = fwd_states[i]['tide']
                    x_crustal = sm_states[i][0][0] - dx_t
                    y_crustal = sm_states[i][1][0] - dy_t
                    z_crustal = sm_states[i][2][0] - dz_t
                    la, lo, al = ecef_a_geodesicas(x_crustal, y_crustal, z_crustal)
                    nt, et = geodesicas_a_utm(la, lo, utm_h)
                    coords.append((nt, et, al, fwd_states[i]['status']))

            else:
                yield f"\n> [SISTEMA] Iniciando Procesamiento DGPS | MÓDULO B (IRLS Clásico Termux)...\n"
                yield "[PROGRESO] Extrayendo Observables Asincrónicas...\n"
                
                rover_tows = sorted(list(obs_r_raw.keys()))
                base_tows = sorted(list(obs_b_raw.keys()))
                obs_b_sync = {}
                for tr in rover_tows:
                    if not base_tows: continue
                    idx = min(range(len(base_tows)), key=lambda i: abs(base_tows[i] - tr))
                    if abs(base_tows[idx] - tr) <= p_max_gap:
                        obs_b_sync[tr] = obs_b_raw[base_tows[idx]].copy()
                        obs_b_sync[tr]['_meta'] = obs_r_raw[tr]['_meta']

                sd_suavizada = aislar_diferencias_MODO_B(obs_b_sync, obs_r_raw)
                if not sd_suavizada: yield "\n> [ERROR] No hay épocas sincronizadas válidas.\n"; return
                
                coords = []
                t_eps = len(sd_suavizada); c = 0
                for t in sd_suavizada:
                    c += 1
                    if c % max(1, t_eps // 10) == 0: yield f"[PROGRESO] Resolviendo Matrices IRLS DGPS... {int((c / t_eps) * 100)}%\n"
                    
                    sem, status = calcular_IRLS_MODO_B(sd_suavizada[t], nav, X_b, Y_b, Z_b, t, p_mask)
                    if sem:
                        la, lo, al = ecef_a_geodesicas(sem[0], sem[1], sem[2])
                        nt, et = geodesicas_a_utm(la, lo, utm_h)
                        coords.append((nt, et, al, status))
                        
                if not coords: yield "\n> [ERROR] Fracaso algorítmico total en Inversión NxN.\n"; return

            res_estadistica = estadistica_desacoplada(coords, p_cp, p_ca, err_hor_max, err_ver_max)
            if res_estadistica[0] is None:
                yield "\n> [ERROR] Operación Abortada: El 100% de las épocas superan el Error Máximo configurado.\n"; return
                
            nf, ef, zf, std_n, std_e, std_z, ret, fix_ratio = res_estadistica
            
            bias_n = leer_estado('opt_bias_n') or 0.0
            bias_e = leer_estado('opt_bias_e') or 0.0
            bias_z = leer_estado('opt_bias_z') or 0.0
            
            if modo_str in ["MODO_B_ASINCRONO", "MODO_B_FASE"]:
                nf_final = nf - bias_n
                ef_final = ef - bias_e
                zf_final_ground = zf - bias_z 
                shift_applied = True
            else:
                nf_final = nf
                ef_final = ef
                zf_final_ground = zf - h_r     
                shift_applied = False
            
            p_dict = {
                'mask': p_mask, 'cp': p_cp, 'ca': p_ca,
                'max_gap': p_max_gap, 'snr': p_snr if modo_str in ["MODO_A_CODIGO", "MODO_C_PPK"] else 0.0,
                'err_h': err_hor_max, 'err_v': err_ver_max,
                'nf': nf_final, 'ef': ef_final, 'zf': zf_final_ground, 
                'ret': ret, 'total': len(coords), 'std_n': std_n, 'std_e': std_e, 'std_z': std_z,
                'ez': std_z, 'fix_r': fix_ratio,
                'base_file': leer_estado('name_base_raw') or "Drive_Base.obs",
                'rover_file': rf_nuevo_filename,
                'nav_file': leer_estado('name_nav_file') or "auto_nav.nav",
                'sp3_file': leer_estado('name_sp3_file') if modo_str in ["MODO_A_CODIGO", "MODO_C_PPK"] else None,
                'b_n': utm_n, 'b_e': utm_e, 'b_z': utm_c,
                'r_n_calc': nf_final, 'r_e_calc': ef_final, 'r_z_calc': zf_final_ground,
                'estrategia': estrategia,
                'shift_applied': shift_applied,
                'bias_n': bias_n, 'bias_e': bias_e, 'bias_z': bias_z
            }
            
            yield "[PROGRESO] Ajuste Finalizado.\n"
            yield generar_informe_ascii("MEDICION", p_dict)
            yield "\n[SUCCESS]"
        except Exception as e: yield f"\n> [ERROR FATAL] {str(e)}"
    return Response(procesar(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True)

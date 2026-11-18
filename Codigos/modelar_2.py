"""
MODELAR UN SOBREVUELO -- CODIGO UNICO Y AUTOCONTENIDO CON SELECTOR DE MODELO
===========================================================================

Un solo archivo. No depende de ningun otro. Cambia UNA linea (MODELO, abajo)
para elegir que calcular:

    MODELO = "conicas"      -> solo Modelo 1 (conicas empalmadas, analitico)
    MODELO = "integracion"  -> solo Modelo 2 (integracion numerica de Newton)
    MODELO = "ambos"        -> los dos y los compara con los datos reales

Usa datos reales de Voyager 2 por Jupiter (julio 1979) de JPL Horizons.
Requisitos: python3 con numpy, scipy, matplotlib, requests + internet la 1a vez
(despues usa la cache en la carpeta ./cache).

Correlo con:   python modelar.py
"""
import os
import numpy as np
import requests
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp

# =====================  <<<   AQUI ELIGES   >>>  =====================
MODELO = "ambos"            # "conicas" | "integracion" | "ambos"
HACER_FIGURAS = True      # True = tambien guarda las graficas PNG
CUERPOS = ("sun", "jup", "sat")   # cuerpos del Modelo 2 (quita "sat" si quieres)
# =====================================================================

# ---- Constantes fisicas (km, s) ----
GM_SUN     = 1.32712440018e11
GM_JUPITER = 1.26686534e8
GM_SATURN  = 3.7931187e7
AU         = 1.495978707e8
A_JUPITER  = 5.2038 * AU

WIN_START = "1979-01-01"
WIN_STOP  = "1980-02-01"
DAY = 86400.0
OUT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(OUT, "cache"); os.makedirs(CACHE, exist_ok=True)


# ==========================================================================
# 1. DESCARGA DE EFEMERIDES (JPL Horizons) con cache en disco
# ==========================================================================
def get_states(command, start, stop, step, center="500@0", label=None):
    """Devuelve (jd, r[km], v[km/s]) del cuerpo 'command' visto desde 'center'."""
    label = label or str(command)
    key = f"{label}_{center}_{start}_{stop}_{step}".replace(" ", "").replace(":", "").replace("@", "at")
    fn = os.path.join(CACHE, key + ".npz")
    if os.path.exists(fn):
        d = np.load(fn); return d["jd"], d["r"], d["v"]
    params = {
        "format": "text", "COMMAND": f"'{command}'", "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES", "EPHEM_TYPE": "VECTORS", "CENTER": f"'{center}'",
        "START_TIME": f"'{start}'", "STOP_TIME": f"'{stop}'", "STEP_SIZE": f"'{step}'",
        "VEC_TABLE": "2", "REF_PLANE": "ECLIPTIC", "REF_SYSTEM": "J2000",
        "OUT_UNITS": "KM-S", "CSV_FORMAT": "YES",
    }
    txt = requests.get("https://ssd.jpl.nasa.gov/api/horizons.api", params=params, timeout=120).text
    lines = txt.splitlines()
    i0 = next(i for i, l in enumerate(lines) if l.strip() == "$$SOE")
    i1 = next(i for i, l in enumerate(lines) if l.strip() == "$$EOE")
    jd, R, V = [], [], []
    for l in lines[i0+1:i1]:
        p = [c.strip() for c in l.split(",")]
        if len(p) < 8:
            continue
        jd.append(float(p[0]))
        R.append([float(p[2]), float(p[3]), float(p[4])])
        V.append([float(p[5]), float(p[6]), float(p[7])])
    jd = np.array(jd); R = np.array(R); V = np.array(V)
    np.savez(fn, jd=jd, r=R, v=V)
    return jd, R, V


# ==========================================================================
# 2. GEOMETRIA DE LA HIPERBOLA (lo propio del Modelo 1)
# ==========================================================================
def hyperbola_from_vinf_rp(v_inf, r_p, mu):
    """(e, delta, b): excentricidad, angulo de deflexion [rad] y par. de impacto."""
    e = 1.0 + r_p * v_inf * v_inf / mu
    delta = 2.0 * np.arcsin(1.0 / e)
    a_mag = mu / (v_inf * v_inf)
    b = a_mag * np.sqrt(e * e - 1.0)
    return e, delta, b


def rotar(v, eje, ang):
    """Rota v un angulo 'ang' alrededor de 'eje' (formula de Rodrigues)."""
    k = eje / np.linalg.norm(eje)
    return v*np.cos(ang) + np.cross(k, v)*np.sin(ang) + k*np.dot(k, v)*(1-np.cos(ang))


# ==========================================================================
# 3. PREPARATIVOS COMUNES (datos, interpoladores, geometria del encuentro)
# ==========================================================================
def preparar():
    print("Descargando efemerides de JPL Horizons (con cache)...")
    jd_sun, r_sun, v_sun = get_states("10",  WIN_START, WIN_STOP, "12 h", label="sun")
    jd_jup, r_jup, v_jup = get_states("599", WIN_START, WIN_STOP, "12 h", label="jupiter")
    jd_sat, r_sat, v_sat = get_states("6",   WIN_START, WIN_STOP, "12 h", label="saturn")
    jd_pr,  r_pr,  v_pr  = get_states("-32", WIN_START, WIN_STOP, "6 h",  label="voyager2")
    jd_rel, r_rel_e, _   = get_states("-32", WIN_START, WIN_STOP, "6 h",
                                      center="500@599", label="voy2_jovi")
    jd_relf, r_relf, _   = get_states("-32", "1979-07-08", "1979-07-11", "5 m",
                                      center="500@599", label="voy2_jovi_fine")

    JD0 = jd_pr[0]
    def secs(jd):
        return (jd - JD0) * DAY

    cs = {
        "sun":  CubicSpline(secs(jd_sun), r_sun), "sunv": CubicSpline(secs(jd_sun), v_sun),
        "jup":  CubicSpline(secs(jd_jup), r_jup), "jupv": CubicSpline(secs(jd_jup), v_jup),
        "sat":  CubicSpline(secs(jd_sat), r_sat),
        "pr":   CubicSpline(secs(jd_pr), r_pr),   "prv":  CubicSpline(secs(jd_pr), v_pr),
    }
    R_SOI = A_JUPITER * (GM_JUPITER / GM_SUN) ** (2.0 / 5.0)
    d_relf = np.linalg.norm(r_relf, axis=1); ip = int(np.argmin(d_relf))
    r_perijove = d_relf[ip]; jd_perijove = jd_relf[ip]
    d_rel = np.linalg.norm(r_rel_e, axis=1); inside = d_rel < R_SOI
    idx_in = int(np.argmax(inside)); idx_out = len(inside)-1-int(np.argmax(inside[::-1]))
    return dict(secs=secs, cs=cs, R_SOI=R_SOI, r_perijove=r_perijove,
                jd_perijove=jd_perijove, jd_soi_in=jd_rel[idx_in], jd_soi_out=jd_rel[idx_out],
                jd_pr=jd_pr, r_pr=r_pr, v_pr=v_pr, jd_jup=jd_jup, r_jup=r_jup,
                r_sun=r_sun, r_sat=r_sat, r_relf=r_relf, ip=ip)


# ==========================================================================
# 4. MODELO 1 -- CONICAS EMPALMADAS
# ==========================================================================
def modelo_conicas(D):
    secs = D["secs"]; cs = D["cs"]
    t_in = secs(D["jd_soi_in"]); t_out = secs(D["jd_soi_out"])
    r_in = cs["pr"](t_in);  v_in = cs["prv"](t_in)
    v_out = cs["prv"](t_out)
    rj_in = cs["jup"](t_in); vj_in = cs["jupv"](t_in); vj_out = cs["jupv"](t_out)
    vsun_out = cs["sunv"](t_out)
    v_inf_in = v_in - vj_in                 # velocidad relativa a Jupiter (entrada)
    v_inf_out_real = v_out - vj_out         # (salida, dato real, para elegir el giro)
    vinf = np.linalg.norm(v_inf_in)
    e, delta, b = hyperbola_from_vinf_rp(vinf, D["r_perijove"], GM_JUPITER)
    h_hat = np.cross(r_in - rj_in, v_inf_in); h_hat /= np.linalg.norm(h_hat)
    cand = {s: rotar(v_inf_in, h_hat, s*delta) for s in (+1.0, -1.0)}
    v_inf_out = min(cand.values(), key=lambda w: np.linalg.norm(w - v_inf_out_real))
    v_helio_out = (v_inf_out + vj_out) - vsun_out
    v_helio_out_real = v_out - vsun_out
    return dict(vinf=vinf, e=e, delta=delta, b=b,
                v_helio_out=v_helio_out, v_helio_out_real=v_helio_out_real,
                err=np.linalg.norm(v_helio_out - v_helio_out_real))


# ==========================================================================
# 5. MODELO 2 -- INTEGRACION NUMERICA DE NEWTON
# ==========================================================================
def modelo_integracion(D):
    secs = D["secs"]; cs = D["cs"]
    mu = {"sun": GM_SUN, "jup": GM_JUPITER, "sat": GM_SATURN}
    def rhs(t, y):
        r = y[:3]; a = np.zeros(3)
        for k in CUERPOS:
            d = r - cs[k](t); a += -mu[k]*d/np.linalg.norm(d)**3
        return np.concatenate([y[3:], a])
    t_eval = secs(D["jd_pr"]); y0 = np.concatenate([D["r_pr"][0], D["v_pr"][0]])
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval,
                    method="DOP853", rtol=1e-11, atol=1e-6, max_step=DAY)
    r_num = sol.y[:3].T; v_num = sol.y[3:].T
    pos_err = np.linalg.norm(r_num - D["r_pr"], axis=1)
    ii = int(np.argmin(np.abs(t_eval - secs(D["jd_soi_out"]))))
    v_helio_out = v_num[ii] - cs["sunv"](secs(D["jd_soi_out"]))
    v_helio_out_real = D["v_pr"][ii] - cs["sunv"](secs(D["jd_soi_out"]))
    return dict(pos_err=pos_err, v_helio_out=v_helio_out,
                v_helio_out_real=v_helio_out_real,
                err=np.linalg.norm(v_helio_out - v_helio_out_real))


# ==========================================================================
# 6. FIGURAS (opcional)
# ==========================================================================
def figura_conicas(D, res1):
    """Figura 3: la hiperbola del Modelo 1 contra la trayectoria REAL de Voyager 2
    vista desde Jupiter, en el plano de la orbita, cerca del perijove."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = os.path.join(OUT, "figs"); os.makedirs(figdir, exist_ok=True)
    R_J = 71492.0
    r = D["r_relf"]; ip = D["ip"]
    # Ejes del plano orbital: p_hat apunta al perijove, h_hat es la normal al plano
    p_hat = r[ip] / np.linalg.norm(r[ip])
    h_hat = np.cross(r[max(ip-1, 0)], r[min(ip+1, len(r)-1)]); h_hat /= np.linalg.norm(h_hat)
    q_hat = np.cross(h_hat, p_hat)
    # Trayectoria real proyectada en ese plano
    xr = r @ p_hat / R_J; yr = r @ q_hat / R_J
    # Hiperbola del Modelo 1: r = p / (1 + e cos(theta)), con el mismo perijove
    e = res1["e"]; rp = D["r_perijove"]; p = rp * (1 + e)
    th_inf = np.arccos(-1.0 / e)
    th = np.linspace(-th_inf + 1e-3, th_inf - 1e-3, 4000)
    rr = p / (1 + e*np.cos(th))
    keep = rr < 1.2 * np.max(np.linalg.norm(r, axis=1))
    xh = rr[keep]*np.cos(th[keep]) / R_J; yh = rr[keep]*np.sin(th[keep]) / R_J
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(xr, yr, 'b-', lw=2, label="Voyager 2 real (JPL Horizons)")
    ax.plot(xh, yh, '--', color="tab:orange", lw=2, label="Hiperbola del Modelo 1 (conicas)")
    ax.add_patch(plt.Circle((0, 0), 1.0, color="peru", label="Jupiter (a escala)"))
    ax.plot([xr[ip]], [yr[ip]], 'ko', ms=5, label="perijove")
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(loc="best")
    ax.set_xlabel("x [radios de Jupiter]  (eje hacia el perijove)")
    ax.set_ylabel("y [radios de Jupiter]")
    ax.set_title(f"Marco jovicentrico: e = {e:.3f},  delta = {np.degrees(res1['delta']):.1f} grados")
    fig.savefig(os.path.join(figdir, "modelar_hiperbola.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def figuras(D, res2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = os.path.join(OUT, "figs"); os.makedirs(figdir, exist_ok=True)
    secs = D["secs"]; cs = D["cs"]; t_eval = secs(D["jd_pr"])
    r_sun_t = np.array([cs["sun"](t) for t in t_eval])
    r_pr_h = D["r_pr"] - r_sun_t
    r_jup_h = D["r_jup"] - D["r_sun"]; r_sat_h = D["r_sat"] - D["r_sun"]
    plt.figure(figsize=(7, 7))
    plt.plot(r_pr_h[:, 0]/AU, r_pr_h[:, 1]/AU, 'b-', lw=1.5, label="Voyager 2")
    plt.plot(r_jup_h[:, 0]/AU, r_jup_h[:, 1]/AU, color="orange", lw=2, label="Jupiter")
    plt.plot(r_sat_h[:, 0]/AU, r_sat_h[:, 1]/AU, color="gold", lw=2, label="Saturno")
    plt.plot(0, 0, 'y*', ms=16, label="Sol")
    plt.axis("equal"); plt.grid(alpha=.3); plt.legend()
    plt.xlabel("x [AU]"); plt.ylabel("y [AU]"); plt.title("Trayectoria heliocentrica")
    plt.savefig(os.path.join(figdir, "modelar_heliocentrica.png"), dpi=130, bbox_inches="tight")
    plt.close()
    if res2 is not None:
        days = D["jd_pr"] - D["jd_perijove"]
        plt.figure(figsize=(9, 4))
        plt.semilogy(days, res2["pos_err"], 'm-')
        plt.axvline(0, color="k", ls=":", alpha=.5)
        plt.xlabel("dias desde el perijove"); plt.ylabel("error de posicion [km]")
        plt.title("Modelo 2: error de la integracion vs datos reales")
        plt.grid(alpha=.3, which="both")
        plt.savefig(os.path.join(figdir, "modelar_error.png"), dpi=130, bbox_inches="tight")
        plt.close()
    print("Figuras guardadas en:", figdir)


# ==========================================================================
# 7. PROGRAMA PRINCIPAL -- aqui actua el selector MODELO
# ==========================================================================
def main():
    assert MODELO in ("conicas", "integracion", "ambos"), "MODELO invalido"
    D = preparar()
    print(f"\nEsfera de influencia R_SOI = {D['R_SOI']:,.0f} km = {D['R_SOI']/AU:.3f} AU")
    print(f"Perijove real              = {D['r_perijove']:,.0f} km")
    print(f"Tiempo dentro de la SOI    = {D['jd_soi_out']-D['jd_soi_in']:.1f} dias\n")

    res1 = res2 = None
    if MODELO in ("conicas", "ambos"):
        res1 = modelo_conicas(D)
        print("="*58 + "\nMODELO 1 -- CONICAS EMPALMADAS\n" + "="*58)
        print(f"v_infinito            = {res1['vinf']:.4f} km/s")
        print(f"excentricidad hiperb. = {res1['e']:.4f}")
        print(f"angulo de deflexion   = {np.degrees(res1['delta']):.3f} deg")
        print(f"parametro de impacto  = {res1['b']:,.0f} km")
        print(f"|v_helio| de salida   = {np.linalg.norm(res1['v_helio_out']):.4f} km/s")
        print(f"  (dato real)         = {np.linalg.norm(res1['v_helio_out_real']):.4f} km/s")
        print(f"error vs dato real    = {res1['err']:.4f} km/s\n")

    if MODELO in ("integracion", "ambos"):
        res2 = modelo_integracion(D)
        print("="*58 + "\nMODELO 2 -- INTEGRACION NUMERICA\n" + "="*58)
        print(f"cuerpos incluidos     = {CUERPOS}")
        print(f"|v_helio| de salida   = {np.linalg.norm(res2['v_helio_out']):.4f} km/s")
        print(f"  (dato real)         = {np.linalg.norm(res2['v_helio_out_real']):.4f} km/s")
        print(f"error vs dato real    = {res2['err']:.4f} km/s")
        print(f"error de posicion max = {res2['pos_err'].max():,.0f} km\n")

    if MODELO == "ambos":
        print("="*58 + "\nCOMPARACION\n" + "="*58)
        print(f"Error Modelo 1 (conicas)     = {res1['err']:.4f} km/s")
        print(f"Error Modelo 2 (integracion) = {res2['err']:.4f} km/s")
        print("-> la diferencia es el error que introduce la aproximacion de")
        print("   conicas empalmadas (ignora al Sol dentro de la SOI).\n")

    if HACER_FIGURAS:
        figuras(D, res2)
        if res1 is not None:
            figura_conicas(D, res1)


if __name__ == "__main__":
    main()

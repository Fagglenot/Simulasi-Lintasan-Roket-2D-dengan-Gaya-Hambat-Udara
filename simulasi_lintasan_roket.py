import numpy as np
import matplotlib.pyplot as grafik
from scipy.integrate import solve_ivp
g0 = 9.8    # Konstanta gravitasi standar
R_bumi = 6371000.0

#ni nanti input, isi dulu karena vscode marah marah karena variabel ini gaada
M_PROP = 300.0
M_DRY = 200.0
sudut_peluncuran = 85
THRUST = 150000
ISP = 250.0 # Impuls spesifik (s)
diameter = 0.3 
# Kumpulan rumus ig
area = np.pi * (diameter/2)**2
MDOT = THRUST / (ISP * g0)   # Literally dm/dt yang nama lainnya M dot 
T_pembakaran = M_PROP/MDOT
M0 = M_PROP + M_DRY

def atmosfer(h):
    h = max(h,0.0)
    R_udara =  287.053  # R/Massa molar udara
    gamma = 1.4
    if h < 11000:    # Lapisan troposfer
        T = 288.15 - 0.0065*h   # T di sini adalah temperatur
        p = 101325 * (T/288.15)**5.25588
    elif h < 20000:   # Lapisan stratosfer
        T = 216.65
        p = 22632.1*np.exp(-g0*(h-11000)/(R_udara*T))
    elif h < 32000:
        T = 216.65 + 0.001 * (h-20000)
        p = 5474.89*(T/216.65)**(-34.1632)
    else:
        T = 228.65
        p = 868.02*np.exp(-g0*(h-32000)/(R_udara*T))

    rho = p/(R_udara*T)
    a = np.sqrt(gamma* R_udara * T)
    return rho,a

# fungsi koefisien drag ini make interpolasi linear, karena hubungan antara mereka itu rumit
# jadi dia nyimpen titik referensi untuk setiap mach berapa koef dragnya 
# np.interp menarik garis lurus diantara garis-garis tersebut. IFYKYK

def koefisien_drag(mach):  
    return float(np.interp(mach, [0.0, 0.8,1.0,1.2,2.0,3.0,5.0],
                           [0.30, 0.32,0.55,0.50,0.38,0.30,0.35]))

def dinamika(t:int,z:list,powered:bool):
    x,y,vx,vy,m = z
    v = np.sqrt(vx**2 + vy**2)
    sudut =  np.radians(sudut_peluncuran)
    thrust = THRUST if powered else 0.0
    Fx, Fy = thrust*np.cos(sudut), thrust*np.sin(sudut) # Gaya sumbu x dan y

    if v > 1e-6:
        rho,a = atmosfer(y)
        Fd = 0.5 * rho * v**2 * koefisien_drag(v/a) * area
        Fx -= Fd * vx/v
        Fy -= Fd* vy/v   # Gaya sumbu x dan y dikurangi gaya drag
    g = g0 * (R_bumi/(R_bumi+y))**2
    ax = Fx/m
    ay = Fy/m - g
    dm = -MDOT if powered else 0.0
    return [vx,vy, ax, ay, dm]

# PENTING: INGAT SEBELUMNYA BAHWA x,y,vx,vy, m = y
# Maka y adalah list yang berisi [x,y,vx,vy,m]

def cek_ground(t,y,powered):    # Ini fungsi yang pengen cek apakah roket mencapai tanah
    return y[1]                 # Karena y[1] adalah y, jika y = 0 maka roket mencapai tanah
cek_ground.terminal = True
cek_ground.direction = -1
def cek_apogee(t,y,powered):  # Apogee itu titik puncak, kali ini fungsi cek kecepatan vertikal
    return y[3]               # Kalau kecepatan vertikal = 0, roket berada di apogee
cek_apogee.terminal = False
cek_apogee.direction = -1

def simulasi():
    sudut = np.radians(sudut_peluncuran)
    y0 = [0.0, 0.0, 1e-3 * np.cos(sudut), 1e-3 * np.sin(sudut), M0]
    simulasi1 = solve_ivp(dinamika, (0,T_pembakaran), y0, args =(True,), method ="RK45", rtol =1e-8, atol= 1e-8, max_step = 1.0, events = ([cek_apogee, cek_ground]))
    # Perhatikan bahwa fungsi dinamika itu parameternya (t,y,bool), itu sudah ditulis di 0,T_pembakaran, y0, True
    # Ini setelah T pembakaran, Mprop habis
    simulasi2 = solve_ivp(dinamika, (T_pembakaran, T_pembakaran+2000), simulasi1.y[:, -1], args =(False,), method = "RK45", rtol = 1e-8, atol = 1e-8, max_step = 1.0, events = ([cek_apogee, cek_ground]))

    t = np.concatenate([simulasi1.t, simulasi2.t[1:]])
    z = np.hstack([simulasi1.y, simulasi2.y[:, 1:]])

    apogee_events =list(simulasi1.y_events[0])+list(simulasi2.y_events[0])
    apogee_times = list(simulasi1.t_events[0])+list(simulasi2.t_events[0])
    t_apo, y_apo = apogee_times[0], apogee_events[0]

    return t,z, t_apo,y_apo

def main():
    t,z,t_apo,y_apo = simulasi()
    x,y,vx,vy, m = z
    v = np.sqrt(vx**2+vy**2)
    i_burn = np.argmin(np.abs(t-T_pembakaran))


    fig, ax = grafik.subplots()
    ax.plot(x/1000, y/1000, color="tab:blue", lw = 2, label = "Lintasan")
    ax.plot(x[i_burn]/1000, y[i_burn]/1000, "o", color="tab:orange", label = f"Burnout ({T_pembakaran:.0f}s)")
    ax.plot(y_apo[0]/1000, y_apo[1]/1000, "r*", ms = 15, label = f"Apogee {y_apo[1]/1000:.1f} km")
    ax.set(xlabel = "Jarak Horizontal[km]", ylabel = "Ketinggian[km]")
    ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("Simulasi Roket 2D dengan Gaya Hambat Udara dan Massa Berubah", fontsize = 14)
    fig.tight_layout()
    fig.savefig("hasil_simulasi_roket.png", dpi = 150)
    grafik.show()

main()

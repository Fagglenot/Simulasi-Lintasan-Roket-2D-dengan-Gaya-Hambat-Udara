import numpy as np
import matplotlib.pyplot as grafik
from scipy.integrate import solve_ivp
import sys
g0 = 9.8    # Konstanta gravitasi standar
R_bumi = 6371000.0

print("Input Parameter Roket")
M_PROP = float(input("Massa propelan (kg): "))
M_DRY = float(input("Massa Kering (kg): "))
THRUST = float(input("Gaya dorong (N): "))
ISP = float(input("Impuls Spesifik Isp (s): "))
diameter = float(input("Diameter roket (m): "))
sudut_peluncuran = float(input("Sudut elevasi peluncuran (derajat, max = 90): "))
# Kumpulan rumus ig
area = np.pi * (diameter/2)**2
MDOT = THRUST / (ISP * g0)   # Literally dm/dt yang nama lainnya M dot 
T_pembakaran = M_PROP/MDOT
M0 = M_PROP + M_DRY
berat_awal = M0 * g0

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
    if v>30.0:
        ux, uy = vx/v,vy/v
    else:
        sudut =  np.radians(sudut_peluncuran)
        ux, uy = np.cos(sudut), np.sin(sudut)
    if THRUST <= berat_awal:
        print("Dorong lebih kecil dari berat roket, roket tidak akan lepas landas.")
        print("Naikkan gaya dorong atau kurangi massa, lalu jalankan ulang.")
        sys.exit()
    thrust = THRUST if powered else 0.0
    Fx, Fy = thrust*ux, thrust*uy # Gaya sumbu x dan y

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
# Maka z adalah list yang berisi [x,y,vx,vy,m]

def cek_ground(t,z,powered):    # Ini fungsi yang pengen cek apakah roket mencapai tanah
    return z[1]                 # Karena z[1] adalah y, jika y = 0 maka roket mencapai tanah
cek_ground.terminal = True
cek_ground.direction = -1
def cek_apogee(t,z,powered):  # Apogee itu titik puncak, kali ini fungsi cek kecepatan vertikal
    return z[3]               # Kalau kecepatan vertikal = 0, roket berada di apogee
cek_apogee.terminal = False
cek_apogee.direction = -1

def simulasi():
    sudut = np.radians(sudut_peluncuran)
    y0 = [0.0, 0.0, 1e-3 * np.cos(sudut), 1e-3 * np.sin(sudut), M0]
    simulasi1 = solve_ivp(dinamika, (0,T_pembakaran), y0, args =(True,), 
                          method ="RK45", rtol =1e-8, atol= 1e-8, max_step = 0.5, 
                          events = ([cek_apogee, cek_ground]))
    # solve_ivp butuh parameter fungsi dinamika, range(Tawal,Takhir), kondisi awal
    # ada bool True karena parameter powered di dinamika adalah True
    # Metode dipakai RK45 karena mudah dan sudah default, rtol dan atol adalah toleransi galat
    # event untuk mengecek apakah apogee atau ground, kalo apogee simulasi lanjut, kalau ground berhenti
    # Ini setelah T pembakaran, Mprop habis
    simulasi2 = solve_ivp(dinamika, (T_pembakaran, T_pembakaran+2000), simulasi1.y[:, -1], args =(False,), 
                          method = "RK45", rtol = 1e-8, atol = 1e-8, max_step = 0.5, 
                          events = ([cek_apogee, cek_ground]))

    t = np.concatenate([simulasi1.t, simulasi2.t[1:]])
    z = np.concatenate([simulasi1.y, simulasi2.y[:, 1:]], axis=1)

    apogee_events =list(simulasi1.y_events[0])+list(simulasi2.y_events[0])
    apogee_times = list(simulasi1.t_events[0])+list(simulasi2.t_events[0])
    t_apo, y_apo = apogee_times[0], apogee_events[0]

    return t,z, t_apo,y_apo

def main():
    t,z,t_apo,y_apo = simulasi()
    x,y,vx,vy, m = z
    v = np.hypot(vx,vy)
    i_burn = np.argmin(np.abs(t-T_pembakaran))


    fig, ax = grafik.subplots(1,2, figsize = (17,5))
    ax[0].plot(x/1000, y/1000, color="tab:blue", lw = 2, label = "Lintasan")
    ax[0].plot(x[i_burn]/1000, y[i_burn]/1000, "o", color="tab:orange", label = f"Burnout ({T_pembakaran:.0f}s)")
    ax[0].plot(y_apo[0]/1000, y_apo[1]/1000, "r*", ms = 15, label = f"Apogee {y_apo[1]/1000:.1f} km")
    ax[0].set(xlabel = "Jarak Horizontal[km]", ylabel = "Ketinggian[km]")
    ax[0].grid(alpha=0.3)
    ax[0].legend()
    # Grafik Kecepatan vs waktu
    ax[1].plot(t, v, label="Kecepatan total", lw=2)
    ax[1].plot(t, vy, "--", label="Kecepatan vertikal")
    ax[1].plot(t, vx, ":", label="Kecepatan horizontal")
    ax[1].axvline(T_pembakaran, color="y", ls="--",  label="Burnout")
    ax[1].axvline(t_apo, color="r", ls="--",  label="Apogee")
    ax[1].set(xlabel="Waktu [s]", ylabel="Kecepatan [m/s]", title="Grafik Kecepatan terhadap Waktu")
    ax[1].grid(alpha=0.3)
    ax[1].legend()
    fig.suptitle("Simulasi Roket 2D dengan Gaya Hambat Udara dan Massa Berubah", fontsize = 14)
    fig.tight_layout()
    fig.savefig("hasil_simulasi_roket.png", dpi = 150)
    grafik.show()

main()

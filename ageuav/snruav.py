import numpy as np
import matplotlib.pyplot as plt
from deep import db_to_linear
from deep import evaluate_classifier_with_snr

alpha = np.array([0.1, 0.3, 0.5, 0.5])
beta = np.array([750, 500, 300, 300])
gamma = np.array([8, 15, 20, 47])
neta_L = np.array([0.1, 1, 1.6, 2.3])
neta_NL = np.array([21, 20, 23, 34])
symbol = ['b--*', 'r:', 'g--+', 'c--o']
D = 500
D1 = 250
D2 = 250
H = np.concatenate((np.arange(10, 101, 8), np.arange(100, 3 * D + 1, 25)))

C_a = np.array([
    [9.34e-1, 2.30e-1, -2.25e-3, 1.86e-5],
    [1.97e-2, 2.440e-3, 6.58e-6, 0],
    [-1.24e-4, -3.34e-6, 0, 0],
    [2.73e-7, 0, 0, 0]
])

C_b = np.array([
    [1.17e+0, -7.56e-2, 1.98e-3, -1.78e-5],
    [-5.79e-3, 1.81e-4, -1.65e-6, 0],
    [1.73e-5, -2.02e-7, 0, 0],
    [-2.00e-8, 0, 0, 0]
])

cof_a = np.zeros(4)
cof_b = np.zeros(4)

a_mat = np.zeros((4, 10))
b_mat = np.zeros((4, 10))

for f in range(4):
    k = 0
    for i in range(4):
        for j in range(4 - i):
            k += 1
            a_mat[f, k - 1] = C_a[i, j] * ((alpha[f] * beta[f]) ** i * (gamma[f] ** j))
            b_mat[f, k - 1] = C_b[i, j] * ((alpha[f] * beta[f]) ** i * (gamma[f] ** j))

for f in range(4):
    for i in range(a_mat.shape[1]):
        cof_a[f] += a_mat[f, i]

    for i in range(b_mat.shape[1]):
        cof_b[f] += b_mat[f, i]

Approx1 = np.zeros((4, H.shape[0]))
fr = 6e9
Cs = 3e8
No = 1e-13
Pw = 5e-3

log_alpha_NL = np.zeros(len(H))
log_alpha_L = np.zeros(len(H))
t = np.zeros(len(H))
alpha_L = np.zeros(len(H))
alpha_NL = np.zeros(len(H))
alpha_1 = np.zeros(len(H))
Approx1 = np.zeros((4, len(H)))
for f in range(4):
    for j in range(H.shape[0]):
        theta1 = np.rad2deg(np.arctan(H[j] / D1))
        t = 1 / (1 + (cof_a[f] * np.exp(-cof_b[f] * (theta1 - cof_a[f]))))
        theta1 = np.degrees(np.arctan(H[j] / D1))
        d1 = np.sqrt(H[j]**2 + D1**2)
        log_alpha_NL[j] = 20 * np.log10(d1) + 20 * np.log10(4 * fr * np.pi / Cs) + neta_NL[f]
        log_alpha_L[j] = 20 * np.log10(d1) + 20 * np.log10(4 * fr * np.pi / Cs) + neta_L[f]
        alpha_L[j] = 1 / 10**(log_alpha_L[j] / 10)
        alpha_NL[j] = 1 / 10**(log_alpha_NL[j] / 10)
        alpha_1[j] = alpha_L[j] * t + alpha_NL[j] * (1 - t)
        l_gain = alpha_1[j] * Pw / No
        snr_db = 10 * np.log10(l_gain)
        acuur = evaluate_classifier_with_snr(snr_db)
        Approx1[f, j] = l_gain

plt.plot(H, Approx1[0, :], 'g--o', label='Suburban')
plt.plot(H, Approx1[1, :], 'r--*', label='Urban')
plt.plot(H, Approx1[2, :], 'b--p', label='Dense Urban')
plt.plot(H, Approx1[3, :], 'm--+', label='Highrise Urban')
plt.xlabel('H')
plt.ylabel('SNR')
plt.legend()
plt.show()

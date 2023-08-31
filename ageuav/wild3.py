import numpy as np
import matplotlib.pyplot as plt
from deep_fading_work import  test_accurcy, get_data, train
from age import calculate_age, calculate_age_theory

alpha = np.array([0.1, 0.3, 0.5, 0.5])
beta = np.array([750, 500, 300, 300])
gamma = np.array([8, 15, 20, 47])
neta_L = np.array([0.1, 1, 1.6, 2.3])
neta_NL = np.array([21, 20, 23, 34])
symbol = ['b--*', 'r:', 'g--+', 'c--o']
D = 1500
#H = np.concatenate((np.arange(10, 101, 100), np.arange(100, 3 * D + 1, 1000)))
#H = np.concatenate(( np.arange(10, 20, 5),np.arange(21, 200, 10), np.arange(201, 5000, 200)))
H = np.concatenate((np.linspace(10, 500, num=10), np.linspace(501, 5000, num=10)))
#H = np.arange(50, 151 , 50)
#H = np.linspace(10, 2000, num=30)
#H=  np.array([10, 50, 250, 500, 800, 1000])
# Load the dataset
train_folder = "/home/chathuranga_basnayaka/Desktop/my/semantic/wild/deepJSCC-feedback/wilddata/forest_fire/Training and Validation"
#test_folder = "/home/chathuranga_basnayaka/Desktop/my/semantic/wild/deepJSCC-feedback/wilddata/forest_fire/Testing"
test_folder = "/home/chathuranga_basnayaka/Desktop/my/semantic/wild/deepJSCC-feedback/wilddata/forest_fire/Training and Validation"


x_test, y_test = get_data(test_folder)

training = False
train_snrdb = 20
block_size = 8
train_K=0.5
if training is True:
   x_train, y_train = get_data(train_folder)
   train_accuarcy= train(train_snrdb, x_train, y_train, x_test, y_test,block_size)
   print(train_accuarcy)

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

Approx1 = np.zeros((1, H.shape[0]))
Approx2 = np.zeros((1, H.shape[0]))
Approx3 = np.zeros((1, H.shape[0]))
Approx4 = np.zeros((1, H.shape[0]))
fr = 900e6
Cs = 3e8
No = 1e-10
Pw = 1
Pw2 = 0.5

log_alpha_NL = np.zeros(len(H))
log_alpha_L = np.zeros(len(H))
t = np.zeros(len(H))
alpha_L = np.zeros(len(H))
alpha_NL = np.zeros(len(H))
alpha_1 = np.zeros(len(H))

log_alpha_NL_d = np.zeros(len(H))
log_alpha_L_d = np.zeros(len(H))
t_d = np.zeros(len(H))
alpha_L_d= np.zeros(len(H))
alpha_NL_d = np.zeros(len(H))
alpha_1_d= np.zeros(len(H))

for f in range(1):
    for j in range(H.shape[0]):
        theta1 = np.rad2deg(np.arctan(H[j] / D))
        t = 1 / (1 + (cof_a[f] * np.exp(-cof_b[f] * (theta1 - cof_a[f]))))
        K=1 / ((cof_a[f] * np.exp(-cof_b[f] * (theta1 - cof_a[f]))))
        theta1 = np.degrees(np.arctan(H[j] / D))
        d1 = np.sqrt(H[j]**2 + D**2)
        log_alpha_NL[j] = 20 * np.log10(d1) + 20 * np.log10(4 * fr * np.pi / Cs) + neta_NL[f]
        log_alpha_L[j] = 20 * np.log10(d1) + 20 * np.log10(4 * fr * np.pi / Cs) + neta_L[f]
        alpha_L[j] = 1 / 10**(log_alpha_L[j] / 10)
        alpha_NL[j] = 1 / 10**(log_alpha_NL[j] / 10)
        alpha_1[j] = alpha_L[j] * t + alpha_NL[j] * (1 - t)
        l_gain = alpha_1[j] * Pw / No
        snr_value_db = 10 * np.log10(l_gain)
        l_gain_2= alpha_1[j] * Pw2 / No
        snr_value_db_2 = 10 * np.log10(l_gain_2)
        acuuracy_1 = test_accurcy(snr_value_db, x_test, y_test,block_size)
        acuuracy_2 = test_accurcy(snr_value_db_2, x_test, y_test,block_size)
        
        D1=700
        fr_d = fr
        theta1_d = np.rad2deg(np.arctan(H[j] / D1))
        t_d = 1 / (1 + (cof_a[f] * np.exp(-cof_b[f] * (theta1_d - cof_a[f]))))
        K_d=1 / ((cof_a[f] * np.exp(-cof_b[f] * (theta1_d - cof_a[f]))))
        theta1_d = np.degrees(np.arctan(H[j] / D1))
        d1_d = np.sqrt(H[j]**2 + D1**2)
        log_alpha_NL_d[j] = 20 * np.log10(d1_d) + 20 * np.log10(4 * fr_d * np.pi / Cs) + neta_NL[f]
        log_alpha_L_d[j] = 20 * np.log10(d1_d) + 20 * np.log10(4 * fr_d * np.pi / Cs) + neta_L[f]
        alpha_L_d[j] = 1 / 10**(log_alpha_L_d[j] / 10)
        alpha_NL_d[j] = 1 / 10**(log_alpha_NL_d[j] / 10)
        alpha_1_d[j] = alpha_L_d[j] * t + alpha_NL_d[j] * (1 - t_d)
        l_gain_d = alpha_1_d[j] * Pw / No
        snr_value_db_d = 10 * np.log10(l_gain_d)
        l_gain_2= alpha_1_d[j] * Pw2 / No
        snr_value_db_2_d = 10 * np.log10(l_gain_2)
        acuuracy_3 = test_accurcy(snr_value_db_d, x_test, y_test,block_size)
        acuuracy_4 = test_accurcy(snr_value_db_2_d, x_test, y_test,block_size)
      
        #acuuracy = test_accurcy(snr_value_db, x_test, y_test,block_size,K)
        #mis_err=1-acuuracy
        #symbol_time=1
        #serv=symbol_time*block_size
        #capture_time=1
        #age_theory, age_sim= calculate_age (mis_err,serv,capture_time) 
        #age_theory= calculate_age_theory (mis_err,serv,capture_time)    
        #Approx1[f, j] = age_theory
        #Approx2[f, j] = age_sim
        Approx1[f, j] = acuuracy_1
        Approx2[f, j] = acuuracy_2
        Approx3[f, j] = acuuracy_3
        Approx4[f, j] = acuuracy_4
        #Approx1[f, j] = snr_value_db
        #Approx2[f, j] = snr_value_db
#plt.plot(H, Approx1[0, :], 'g--o', label='Theory')
plt.plot(H, Approx1[0, :], color='g', linestyle='-', marker='o', label='$P_w=30dBm, d_{G,U}=1500m$')
plt.plot(H, Approx2[0, :], color='b', linestyle='-', marker='p', label='$P_w=27dBm,d_{G,U}=1500m$')
plt.plot(H, Approx3[0, :], color='r', linestyle='--', marker='o', label='$P_w=30dBm,d_{G,U}=750m$')
plt.plot(H, Approx4[0, :], color='#FFA500', linestyle='--', marker='o', label='$P_w=27dBm,d_{G,U}=750m$')

font_family = "Times New Roman"
plt.xlabel(' UAV Height (H) [m]',fontname=font_family,fontsize=14)
plt.ylabel('Classfication accuracy',fontname=font_family,fontsize=14)
plt.legend()
plt.grid(True)
#plt.xscale('log')
plt.show()


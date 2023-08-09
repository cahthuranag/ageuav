import numpy as np

serv = 1

mu = 1 / serv
lambda1 = mu
num_events = 10000
inter_arrival_times = -np.log(np.random.rand(num_events)) / lambda1

arrival_timestamps = np.insert(np.cumsum(inter_arrival_times[:-1]), 0, 0)
inter_service_times = -np.log(np.random.rand(num_events)) / mu
server_timestamps_1 = np.copy(arrival_timestamps)
departure_timestamps_1 = np.zeros(num_events)
departure_timestamps_s = np.zeros(num_events)

erov_s = 0.99
er_indi = np.random.rand(1) > erov_s

if er_indi == 0:
    departure_timestamps_s[0] = 0
    server_timestamps_1[0] = 0
    departure_timestamps_1[:] = 0
else:
    departure_timestamps_s[0] = server_timestamps_1[0] + inter_service_times[0]
    departure_timestamps_1[0] = server_timestamps_1[0] + inter_service_times[0]

for i in range(1, num_events):
    if arrival_timestamps[i] < departure_timestamps_1[i - 1]:
        departure_timestamps_1[i] = departure_timestamps_1[i - 1]
        departure_timestamps_s[i] = 0
        server_timestamps_1[i] = 0
    else:
        er_indi = np.random.rand(1) > erov_s
        if er_indi == 0:
            departure_timestamps_s[i] = 0
            server_timestamps_1[i] = 0
            departure_timestamps_1[i] = departure_timestamps_1[i - 1]
        else:
            departure_timestamps_1[i] = arrival_timestamps[i] + inter_service_times[i]
            departure_timestamps_s[i] = departure_timestamps_1[i]

dep = departure_timestamps_s[departure_timestamps_s != 0]
sermat = server_timestamps_1[server_timestamps_1 != 0]
depcop = np.copy(dep)

if server_timestamps_1[-1] == 0:
    if depcop.size > 0:
        depcop = depcop[:-1]
        maxt = max(arrival_timestamps[-1], dep[-1])
    else:
        maxt = arrival_timestamps[-1]
    v1 = np.append(depcop, maxt)
else:
    v1 = dep

if departure_timestamps_s[0] == 0:
    if sermat.size > 0:
        t1 = sermat
    else:
        t1 = np.array([0])
else:
    t1 = np.insert(sermat, 0, 0)

def av_age_func(v, T, lambha):
    p = lambha * 0.001
    times = np.arange(0, v[-1] + p, p)
    ii = 0
    offset = 0
    age = np.copy(times)
    for i in range(len(times)):
        if ii < len(v) and times[i] >= v[ii]:
            offset = T[ii]
            ii += 1
        age[i] -= offset
    av_age = np.trapz(age, times) / max(times)
    return av_age

age_simulation = av_age_func(v1, t1, serv)

erov = erov_s
#age_theory = (1 / (lambda1)) + (1 / ((1 - erov) * mu)) + ((lambda1) / ((mu + lambda1) * mu))
age_theory = 1 / (lambda1) + 1 / ((1 - erov) * mu) + lambda1 / ((mu * (1 - erov) )* (lambda1+(mu * (1 - erov))))

print(age_theory)
print(age_simulation)


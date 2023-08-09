import numpy as np

serv = 1
mu = 1 / serv
lambda1 = mu
num_events = 10000

inter_arrival_times = 1 / lambda1 * np.log(1 / np.random.rand(num_events))
arrival_timestamps = np.insert(np.cumsum(inter_arrival_times[:-1]), 0, 0)
inter_service_times = 1 / mu * np.log(1 / np.random.rand(num_events))
departure_timestamps = arrival_timestamps + inter_service_times
departure_timestamps_new = []
arrival_timestamps_new = []

erov_s = 0.6

for i in range(1, num_events):
    er_indi = np.random.rand(1) > erov_s
    if arrival_timestamps[i] < departure_timestamps[i - 1] or not er_indi:
        departure_timestamps_new.append(np.nan)
        arrival_timestamps_new.append(np.nan)
    else:
        departure_timestamps_new.append(departure_timestamps[i])
        arrival_timestamps_new.append(arrival_timestamps[i])

departure_timestamps_valid = np.array(departure_timestamps_new)[~np.isnan(departure_timestamps_new)]
arrival_timestamps_valid = np.array(arrival_timestamps_new)[~np.isnan(arrival_timestamps_new)]


def av_age_func(v, T, lambha):
    p = lambha * 0.01
    times = np.arange(0, v[-1], p)
    for i in range(1, len(v)):
        dummy = np.arange(v[i - 1], v[i], p)
        times = np.concatenate((times, dummy))
    ii = 1
    offset = 0
    age = times.copy()
    for i in range(len(times)):
        if ii <= len(v) and times[i] >= v[ii]:
            offset = T[ii]
            ii = ii + 1
        age[i] = age[i] - offset
    av_age = np.trapz(age, times) / max(times)
    return av_age


age_simulation = av_age_func(departure_timestamps_valid, arrival_timestamps_valid, serv)
erov = erov_s
age_theory = (1 / lambda1) + (1 / ((1 - erov) * mu)) + ((lambda1) / (((mu * (1 - erov)) + lambda1) * (mu * (1 - erov))))

print(age_simulation)
print(age_theory)

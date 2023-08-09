import numpy as np

def av_age_func(v, T, lambha):
    p = lambha * 0.001
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

def calculate_age(erov_s):
    serv = 1
    mu = 1 / serv
    lambda1 = mu
    num_events = 2000

    inter_arrival_times = 1 / lambda1 * np.log(1 / np.random.rand(num_events))
    arrival_timestamps = np.insert(np.cumsum(inter_arrival_times[:-1]), 0, 0)
    inter_service_times = 1 / mu * np.log(1 / np.random.rand(num_events))
    departure_timestamps = arrival_timestamps + inter_service_times
    departure_timestamps_new = []
    arrival_timestamps_new = []
    

    er_indi = np.random.rand(1) > erov_s

    if  not er_indi:
        departure_timestamps_new.append(departure_timestamps[1])
        arrival_timestamps_new.append(arrival_timestamps[0])
    else:
        departure_timestamps_new.append(departure_timestamps[0])
        arrival_timestamps_new.append(arrival_timestamps[0])


    for i in range(1, num_events-2):
        er_indi = np.random.rand(1) > erov_s
        if arrival_timestamps[i] < departure_timestamps[i - 1] or not er_indi:
            departure_timestamps_new.append(np.nan)
            arrival_timestamps_new.append(np.nan)
        else:
            departure_timestamps_new.append(departure_timestamps[i])
            arrival_timestamps_new.append(arrival_timestamps[i])


    er_indi = np.random.rand(1) > erov_s
    if arrival_timestamps[num_events-1] < departure_timestamps[num_events - 2] or not er_indi:
        departure_timestamps_new.append(departure_timestamps[num_events-1])
        arrival_timestamps_new.append(arrival_timestamps[num_events-2])
    else:
        departure_timestamps_new.append(departure_timestamps[num_events-1])
        arrival_timestamps_new.append(arrival_timestamps[num_events-1])

    departure_timestamps_valid = np.array(departure_timestamps_new)[~np.isnan(departure_timestamps_new)]
    arrival_timestamps_valid = np.array(arrival_timestamps_new)[~np.isnan(arrival_timestamps_new)]
    print(departure_timestamps_valid, arrival_timestamps_valid)

    age_simulation = av_age_func(departure_timestamps_valid, arrival_timestamps_valid, serv)
    erov = erov_s
    age_theory = (1 / (lambda1 * (1 - erov))) + (1 / ((1 - erov) * mu)) + ((lambda1) / (((mu) + lambda1) * (mu)))

    return age_theory, age_simulation

# Example usage:
erov_s = 0.9999
age_theory, age_simulation = calculate_age(erov_s)
print("Age Theory:", age_theory)
print("Age Simulation:", age_simulation)

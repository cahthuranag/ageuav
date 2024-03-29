import numpy as np

# Reproducibility of results
np.random.seed(42)

def av_age_func(v, T, lambha):
    if len(v) == 0 and len(T) == 0:
        print('input is empty for age calculation')
    if len(v) == 1 and len(T) == 1:
        av_age = (v[0] + T[0]) / 2
    else:
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

def calculate_age(erov_s,serv,capture_time):
    
    mu = 1 / serv
    lambda1 = 1/capture_time
    num_events = 50000

    inter_arrival_times = 1 / lambda1 * np.log(1 / np.random.rand(num_events))
    arrival_timestamps = np.insert(np.cumsum(inter_arrival_times[:-1]), 0, 0)
    inter_service_times = 1 / mu * np.log(1 / np.random.rand(num_events))
    departure_timestamps = arrival_timestamps + inter_service_times

    departure_timestamps_new = []
    arrival_timestamps_new = []

    for i in range(1, num_events-1):
        er_indi = np.random.rand(1) > erov_s
        if arrival_timestamps[i] >= departure_timestamps[i - 1] and er_indi:
            departure_timestamps_new.append(departure_timestamps[i])
            arrival_timestamps_new.append(arrival_timestamps[i])

    if np.random.rand(1) > erov_s:
        if len(departure_timestamps_new) > 0:
            arrival_timestamps_new.insert(0, arrival_timestamps[0])
            departure_timestamps_new.insert(0, departure_timestamps_new[-1])
    else:
        departure_timestamps_new.insert(0, departure_timestamps[0])
        arrival_timestamps_new.insert(0, arrival_timestamps[0])

    if arrival_timestamps[-1] >= departure_timestamps[- 2] and np.random.rand(1) > erov_s:
        if len(arrival_timestamps_new) > 0:
            departure_timestamps_new.append(departure_timestamps[-1])
            arrival_timestamps_new.append(arrival_timestamps_new[-1])
    else:
        departure_timestamps_new.append(departure_timestamps[-1])
        arrival_timestamps_new.append(arrival_timestamps[-1])

    if len(arrival_timestamps_new) == 0 and len(departure_timestamps_new) == 0:
        arrival_timestamps_new.append(arrival_timestamps[0])
        departure_timestamps_new.append(departure_timestamps[-1])

    age_simulation = av_age_func(departure_timestamps_new, arrival_timestamps_new, serv)
    age_theory = (1 / (lambda1 * (1 - erov_s))) + (1 / ((1 - erov_s) * mu)) + ((lambda1) / (((mu) + lambda1) * (mu)))

    return age_theory, age_simulation

def calculate_age_theory(erov_s,serv,capture_time):
    mu = 1 / serv
    lambda1 = 1/capture_time
    age_theory = (1 / (lambda1 * (1 - erov_s))) + (1 / ((1 - erov_s) * mu)) + ((lambda1) / (((mu) + lambda1) * (mu)))
    return age_theory

def main():
   erov_s = 0.2
   serv = 8
   capture_time=1
   age_theory, age_simulation = calculate_age(erov_s,serv,capture_time)
   print("Age Theory:", age_theory)
   print("Age Simulation:", age_simulation)

if __name__ == "__main__":
    main()
    
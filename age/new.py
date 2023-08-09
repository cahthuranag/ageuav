import numpy as np

np.random.seed(42)  # Setting a random seed for reproducibility

serv = 1
mu = 1 / serv
lambda1 = mu
num_events = 100

# Generate inter-arrival times and arrival timestamps
inter_arrival_times = (1 / lambda1) * np.log(1 / np.random.rand(num_events))
arrival_timestamps = np.concatenate(([0], np.cumsum(inter_arrival_times)))

# Generate inter-service times
inter_service_times = (1 / mu) * np.log(1 / np.random.rand(num_events - 1))  # Corrected shape

# Calculate departure timestamps based on inter-service times
departure_timestamps = np.concatenate(([0], np.cumsum(inter_service_times)))

# Initialize arrays for modified timestamps
departure_timestamps_new = np.empty_like(departure_timestamps)
arrival_timestamps_new = np.empty_like(arrival_timestamps)

erov_s = 0
er_indi = np.random.rand() > erov_s  # Corrected indexing

# Handle the first event
if er_indi == 0:
    departure_timestamps_new[0] = departure_timestamps[1]
    arrival_timestamps_new[0] = arrival_timestamps[0]
else:
    departure_timestamps_new[0] = departure_timestamps[0]
    arrival_timestamps_new[0] = arrival_timestamps[0]

# Handle the remaining events
for i in range(1, num_events - 1):
    if arrival_timestamps[i] < departure_timestamps_new[i - 1]:
        departure_timestamps_new[i] = np.nan
        arrival_timestamps_new[i] = np.nan
    else:
        er_indi = np.random.rand() > erov_s  # Corrected indexing
        if er_indi == 0:
            departure_timestamps_new[i] = np.nan
            arrival_timestamps_new[i] = np.nan
        else:
            departure_timestamps_new[i] = departure_timestamps[i]
            arrival_timestamps_new[i] = arrival_timestamps[i]

# Handle the last event
if arrival_timestamps[num_events - 1] < departure_timestamps_new[num_events - 2]:
    departure_timestamps_new[num_events - 1] = departure_timestamps[num_events - 1]
    arrival_timestamps_new[num_events - 1] = arrival_timestamps[num_events - 2]
else:
    er_indi = np.random.rand() > erov_s  # Corrected indexing
    if er_indi == 0:
        departure_timestamps_new[num_events - 1] = departure_timestamps[num_events - 1]
        arrival_timestamps_new[num_events - 1] = arrival_timestamps[num_events - 2]
    else:
        departure_timestamps_new[num_events - 1] = departure_timestamps[num_events - 1]
        arrival_timestamps_new[num_events - 1] = arrival_timestamps[num_events - 1]
departure_timestamps_valid = departure_timestamps_new[~np.isnan(departure_timestamps_new)]
arrival_timestamps_valid = arrival_timestamps_new[~np.isnan(arrival_timestamps_new)]

def av_age_func(v, T, lambha):
    p = lambha * 0.001
    times = np.arange(0, v[-1] + p, p)
    num = v.shape[0]
    for i in range(1, num):
        dummy = np.arange(v[i - 1], v[i] + p, p)
        times = np.concatenate((times, dummy))
    ii = 1
    offset = 0
    age = times.copy()
    for i in range(len(times)):
        if ii <= num and times[i] >= v[ii]:
            offset = T[ii]
            ii = ii + 1
        age[i] = age[i] - offset
    av_age = np.trapz(age, times) / max(times)
    return av_age

age_simulation = av_age_func(departure_timestamps_valid, arrival_timestamps_valid, serv)
erov = erov_s
age_theory = (1 / (lambda1 * (1 - erov))) + (1 / ((1 - erov) * mu)) + ((lambda1) / (((mu) + lambda1) * (mu)))

print(age_simulation)
print(age_theory)

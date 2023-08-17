clear all
close all
serv = 1;
mu = 1 / serv;
lambda1 = mu;
num_events = 100;

inter_arrival_times = 1 / lambda1 * log(1 ./ rand(1, num_events));

arrival_timestamps = [0 cumsum(inter_arrival_times(1:num_events - 1))];
inter_service_times = 1 / mu * log(1 ./ rand(1, num_events));

departure_timestamps = arrival_timestamps + inter_service_times;
departure_timestamps_new = [];
arrival_timestamps_new = [];

erov_s=0;
er_indi= rand(1,1)>erov_s;  
  if isequal( er_indi,0)
        departure_timestamps_new(1) = departure_timestamps(2);
        arrival_timestamps_new(1) = arrival_timestamps(1);
        
        
    else
        departure_timestamps_new(1) = departure_timestamps(1);
        arrival_timestamps_new(1) = arrival_timestamps(1);
    end

for i = 2:(num_events-1)
    if arrival_timestamps(i) < departure_timestamps(i - 1) 
        departure_timestamps_new(i) = nan;
        arrival_timestamps_new(i) = nan;
    else 
       er_indi= rand(1,1)>erov_s;   
    if isequal( er_indi,0)
        departure_timestamps_new(i) = nan;
        arrival_timestamps_new(i) = nan;
        
    else
        departure_timestamps_new(i) = departure_timestamps(i);
        arrival_timestamps_new(i) = arrival_timestamps(i);
    end
    end
end

    if arrival_timestamps(num_events) < departure_timestamps(num_events - 1) 
        departure_timestamps_new(num_events) =  departure_timestamps(num_events);
        arrival_timestamps_new(num_events) =   arrival_timestamps(num_events-1);
    else 
       er_indi= rand(1,1)>erov_s;   
    if isequal( er_indi,0)
        departure_timestamps_new(num_events) =  departure_timestamps(num_events);
        arrival_timestamps_new(num_events) =  arrival_timestamps(num_events-1);
        
    else
        departure_timestamps_new(num_events) = departure_timestamps(num_events);
        arrival_timestamps_new(num_events) = arrival_timestamps(num_events);
    end
    end
departure_timestamps_valid = departure_timestamps_new(~isnan(departure_timestamps_new));
arrival_timestamps_valid = arrival_timestamps_new(~isnan(arrival_timestamps_new));

 age_simulation=  av_age_func(departure_timestamps_valid,arrival_timestamps_valid,serv);
    erov=erov_s;
age_theory =(1/(lambda1*(1-erov)))+(1/((1-erov)*mu))+((lambda1)/(((mu)+lambda1)*(mu)));

 
disp(age_simulation)
disp(age_theory)
 
 function [av_age] = av_age_func(v, T, lambha)
    p = lambha * 0.001;
    times = 0:p:v(end);
    [~, num] = size(v);
    for i = 2:num
        dummy = v(i - 1):p:v(i);
        times = [times dummy];
    end
    ii = 1;
    offset = 0;
    age = times;
    for i = 1:length(times)
        if ii <= num && times(i) >= v(ii)
            offset = T(ii);
            ii = ii + 1;
        end
        age(i) = age(i) - offset;
    end
    av_age = trapz(times, age) / max(times);
end
 


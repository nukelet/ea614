import numpy as np
import matplotlib.pyplot as plt

def coef(k, omega):
    T = 2*np.pi/omega
    if k == 0:
        return 1/T
    elif k%4 == 0:
        return 0
    elif k%2 == 1:
        result = -1/(k*k * np.pi * omega) + np.sin(k * 0.5*np.pi)/(np.pi * k)
        return result
    else:
        return -2/(np.pi * k * k * omega)

def fourier(t, omega, k_range):
    result = 0
    for k in range(-k_range, k_range+1):
        result += coef(k, omega) * np.cos(k * omega * t)
    return result

def original(t, period):
    # "map" t to [-period, period]
    t -= int(t/period)*period
    if t > period/2:
        t -= period
    if t < -period/2:
        t += period

    if -period/4 <= t and t <= 0:
        return -t
    elif 0 <= t and t <= period/4:
        return t
    else:
        return 0

def quad_error(original_vals, x_vals, t_vals):
    delta_t = t_vals[1] - t_vals[0]
    period = t_vals[-1] - t_vals[1]
    error = 0
    for x, y in zip(original_vals, x_vals):
        error += (x-y) * (x-y) * delta_t
    return error/period

def freq_filter(omega):
    cutoff = 10
    if omega == 0:
        return 0
    else:
        return 1/(1 - 1j*(cutoff/omega))

def filter_response(t, omega):
    response = 0
    for k in range(-50, 51):
        response += coef(k, omega) * freq_filter(k * omega) * \
            np.exp(1j * k * omega * t)
    return response

def main():
    period = 4
    omega = (2 * np.pi)/period
    t_vals = np.arange(-period/2, period/2, 0.01)
    
    original_vals = []
    for t in t_vals:
        original_vals.append(original(t, period))

    coefs = []
    for k in range(-50, 51):
        coefs.append(coef(k, omega))


    fig, axs = plt.subplots(2, 2)

    for k_range, ax in zip([1, 10, 20, 50], axs.flat):
        x_vals = []
        for t in t_vals:
            x_vals.append(fourier(t, omega, k_range))

        ax.plot(t_vals, x_vals, 'b', t_vals, original_vals, 'r')
        ax.set_title(f"N = {k_range}")
        # error = quad_error(original_vals, x_vals, t_vals)
        # print(f"N = {k_range}, error: {error}")
            
    plt.show()

    plt.stem(np.arange(-50, 51), coefs)
    plt.xlabel('k') 
    plt.ylabel('|c_k|')
    plt.show()

    filter_abs_vals = []
    filter_angle_vals = []
    omega_vals = np.arange(-30, 30, 0.01)
    for w in omega_vals:
        response = freq_filter(w)
        filter_abs_vals.append(np.abs(response))
        filter_angle_vals.append(np.angle(response))
    
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(omega_vals, filter_abs_vals)
    axs[0].set_title("Modulo")
    axs[1].plot(omega_vals, filter_angle_vals)
    axs[1].set_title("Fase")
    plt.show()

    filtered_vals = []
    print(f"omega: {omega}")
    for t in t_vals:
        filtered_vals.append(filter_response(t, omega))

    plt.plot(t_vals, filtered_vals)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()

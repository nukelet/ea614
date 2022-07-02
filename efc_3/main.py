import numpy as np
import matplotlib.pyplot as plt

"""Implementation of HW assignment 3

This assignment covered real-world (a.k.a. non-ideal) examples of low-pass filters
and their behavior within the passband and stopband, as well as in region of transition
between them.

"""

def chebyshev_table(w, wc, n):
    """Generates a table with values of Chebyshev polynomials.

    This evalutes T_n(w/wc) (i.e. the n-th Chebyshev polynomial at w/wc)
    for all values in the frequency array w.

    Args:
        w: array containing frequency values
        wc: the filter cutoff frequency
        n: order of the Chebyshev filter

    Returns:
        An array containing the evaluated Chebyshev polynomials
    """
    Tn = np.zeros((w.size,))
    Tn[abs(w) < wc] = np.cos(n*np.arccos(w[abs(w) < wc] / wc))
    Tn[abs(w) >= wc] = np.cosh(n*np.arccosh(w[abs(w) >= wc] / wc))
    return Tn

def chebyshev_filter(w, wc, n, epsilon):
    """Calculates the magnitude response of a Chebyshev filter.

    This evaluates the magnitude response of a Chebyshev filter for an array
    of frequencies.

    Args:
        w: array containing frequency values
        wc: the filter cutoff frequency
        n: order of the Chebyshev filter
        epsilon: ripple factor

    """
    h_vals = np.zeros(w.size)
    chebyshev_coefs = chebyshev_table(w, wc, n)
    for i, omega in enumerate(w):
        coef = chebyshev_coefs[i]
        h_vals[i] = 1/np.sqrt(1 + (epsilon*coef)**2)
    return h_vals

def butterworth_filter(w, wc, n):
    h_vals = np.zeros(w.size)
    for i, omega in enumerate(w):
        h_vals[i] = 1/np.sqrt(1 + (omega/wc)**(2*n))
    return h_vals

def ideal_filter(w, wc):
    h_vals = np.zeros(w.size)
    for i, omega in enumerate(w):
        h_vals[i] = 1 if omega <= wc else 0
    return h_vals

def signal_transform(w, wm):
    x_vals = np.zeros(w.size)
    for i, omega in enumerate(w):
        x_vals[i] = 2 * np.sin((omega/wm) * np.pi) / omega
    return x_vals

def filter_signal(x_vals, h_vals):
    output = []
    for x, h in zip(x_vals, h_vals):
        output.append(x*h)
    return output

def main():
    n_vals = [1, 2, 3, 4, 5]
    color_vals = ['red', 'green', 'blue', 'orange', 'purple']
    w = np.arange(0, 40, 0.01)
    x_vals = signal_transform(w, 5)

    # for (n, color) in zip(n_vals, color_vals):
    #     h_vals = chebyshev_filter(w, 10, n, 0.2)
    #     plt.plot(w, h_vals, color=color, label=f"n = {n}")

    # plt.legend(loc="lower left")
    # plt.xlabel("frequency (rad/s)")
    # plt.title("Magnitude response of the Chebyshev filter (cutoff=10)")
    # plt.show()

    # epsilon_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
    # for (epsilon, color) in zip(epsilon_vals, color_vals):
    #     h_vals = chebyshev_filter(w, 10, 3, epsilon)
    #     plt.plot(w, h_vals, color=color, label=f"epsilon = {epsilon}")

    # plt.legend(loc="lower left")
    # plt.xlabel("frequency (rad/s)")
    # plt.title("Magnitude response of the Chebyshev filter (cutoff=10)")
    # plt.show()

    # for (n, color) in zip(n_vals, color_vals):
    #     h_vals = butterworth_filter(w, 10, n)
    #     plt.plot(w, h_vals, color=color, label=f"n = {n}")

    # plt.legend(loc="lower left")
    # plt.xlabel("frequency (rad/s)")
    # plt.title("Magnitude response of the Butterworth filter (cutoff=10)")
    # plt.show()

    plt.plot(w, x_vals)
    plt.plot(w, np.zeros(w.size))

    plt.legend(loc="lower left")
    plt.xlabel("frequency (rad/s)")
    plt.title("Fourier transform of input signal (rectangle function)")
    plt.show()

    ideal_response = ideal_filter(w, 10)
    ideal_output = filter_signal(x_vals, ideal_response)

    chebyshev_response = chebyshev_filter(w, 10, 3, 0.9)
    chebyshev_output = filter_signal(x_vals, chebyshev_response)

    butterworth_response = butterworth_filter(w, 10, 2)
    butterworth_output = filter_signal(x_vals, butterworth_response)

    # plt.plot(w, ideal_response, label="ideal", color='r')
    # plt.plot(w, chebyshev_response, label="chebyshev", color='g')
    # plt.plot(w, butterworth_response, label="butterworth", color='b')

    # plt.legend(loc="upper right")
    # plt.xlabel("frequency (rad/s)")
    # plt.title("Magnitude response for different lowpass filters")
    # plt.show()

    # plt.plot(w, ideal_output, label="ideal", color='r')
    # plt.plot(w, chebyshev_output, label="chebyshev", color='g')
    # plt.plot(w, butterworth_output, label="butterworth", color='b')

    # plt.legend(loc="upper right")
    # plt.xlabel("frequency (rad/s)")
    # plt.title("Output in frequency domain for different lowpass filters")
    # plt.show()
        

if __name__ == '__main__':
    main()

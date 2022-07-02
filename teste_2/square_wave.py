import matplotlib.pyplot as plt
import numpy as np

def fourier(t):
    result = 0
    period = 1
    k_count = 99
    w = 2 * np.pi / period
    for k in range(-k_count, k_count+1, 2):
        u = 2/(k * np.pi)**3
        print(f"k: {k}, u: {u}")
        result += u * np.sin(k * w * t)

    return result
def main():
    t_vals = np.arange(0, 4, 0.01)
    x_vals = []
    for t in t_vals:
        x_vals.append(fourier(t))
        
    print(x_vals)
    plt.plot(t_vals, x_vals, 'r', t_vals, 8*np.sin(t_vals*np.pi), 'b')
    plt.show()

if __name__ == '__main__':
    main()

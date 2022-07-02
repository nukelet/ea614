import numpy as np
import random
import matplotlib.pyplot as plt

def main():
    convmode = 'valid'

    h = np.array([1, -0.5, 0, 0])
    w1 = np.array([1, 0.5, 0.25, 0.125, 0.0625])
    filter1 = np.convolve(h, w1)
    print(f"g1: {filter1}")
    w2 = np.array([1, -0.75, 1.5, -0.2, 0.3])
    filter2 = np.convolve(h, w2)
    print(f"g2: {filter2}")

    s = random.choices([-3, -1, 1, 3], k = 100)
    x = np.convolve(s, h, mode = convmode)

    print(x)

    ticks = np.arange(100)


    plt.figure()

    # plt.subplot(121)
    plt.stem(s)
    plt.title("sinal original")
    plt.plot()
    plt.show()

    # plt.subplot(122)
    plt.stem(x)
    plt.title("sinal do canal (distorcido)")
    plt.plot()
    plt.show()

    plt.hist(s)
    plt.title("histograma dos simbolos no sinal original")
    plt.plot()
    plt.show()

    # plt.subplot(223)
    convoluted = np.convolve(filter1, x, mode = convmode)
    # print(convoluted, convoluted.size)
    plt.stem(convoluted, markerfmt='r')
    plt.stem(s, markerfmt='b')
    plt.title("sinal equalizado (filtro 1)")
    # plt.plot()
    plt.show()

    # plt.subplot(224)
    convoluted = np.convolve(filter2, x, mode = convmode)
    plt.stem(convoluted, markerfmt='r')
    plt.stem(s, markerfmt='b')
    plt.title("sinal equalizado (filtro 2)")
    # plt.plot()

    plt.show()

    pass

if __name__ == "__main__":
    main()

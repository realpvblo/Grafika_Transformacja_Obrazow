import cv2
import matplotlib.pyplot as plt
import numpy as np
from mttkinter import mtTkinter as tk
from PIL import Image


def onclick1():
    # Read an image
    img_bgr = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\Ratusz_Bialystok.jpg",
                         cv2.IMREAD_UNCHANGED)
    img_bgr2 = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\KampusUwB1.jpg",
                          cv2.IMREAD_UNCHANGED)

    plt.imshow(img_bgr)
    plt.show()

    # Histogram plotting of the image
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img_bgr],
                             [i], None,
                             [256],
                             [0, 256])

        plt.plot(histr, color=col)

        # Limit X - axis to 256
        plt.xlim([0, 256])

    plt.show()


def onclick2():
    img_bgr = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\Ratusz_Bialystok.jpg",
                         cv2.IMREAD_UNCHANGED)
    img_bgr2 = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\KampusUwB1.jpg",
                          cv2.IMREAD_UNCHANGED)

    # get height and width of the image
    height, width, _ = img_bgr.shape

    for i in range(0, height - 1):
        for j in range(0, width - 1):
            # Get the pixel value
            pixel = img_bgr[i, j]

            # Negate each channel by
            # subtracting it from 255

            # 1st index contains red pixel
            pixel[0] = 255 - pixel[0]

            # 2nd index contains green pixel
            pixel[1] = 255 - pixel[1]

            # 3rd index contains blue pixel
            pixel[2] = 255 - pixel[2]

            # Store new values in the pixel
            img_bgr[i, j] = pixel

    # Display the negative transformed image
    plt.imshow(img_bgr)
    plt.show()

    # Histogram plotting of the
    # negative transformed image
    color = ('b', 'g', 'r')

    for i, col in enumerate(color):
        histr = cv2.calcHist([img_bgr],
                             [i], None,
                             [256],
                             [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


def onclick3():
    # Multiply image
    img_bgr = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\Ratusz_Bialystok.jpg",
                         cv2.IMREAD_UNCHANGED)
    img_bgr2 = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\KampusUwB1.jpg",
                          cv2.IMREAD_UNCHANGED)

    a = img_bgr.astype(float) / 255
    b = img_bgr2.astype(float) / 255
    np.zeros_like(a)
    ab = (a * b)

    # for i in range(0, height - 1):
    #     for j in range(0, width - 1):
    #         # Get the pixel value
    #         pixel = img_bgr[i, j]
    #
    #         # 1st index contains red pixel
    #         pixel[0] = pixel[0] * (255 - pixel[0])
    #
    #         # 2nd index contains green pixel
    #         pixel[1] = pixel[1] * (255 - pixel[1])
    #
    #         # 3rd index contains blue pixel
    #         pixel[2] = pixel[2] * (255 - pixel[2])
    #
    #         # Store new values in the pixel
    #         img_bgr[i, j] = pixel

    # Display the negative transformed image
    plt.imshow(ab)
    plt.show()

    # Histogram plotting of the image
    color = ('b', 'g', 'r')

    PIL_image = Image.fromarray(ab.astype('uint8'), 'RGB')
    for i, col in enumerate(color):
        histr = cv2.calcHist([PIL_image],
                             [i], None,
                             [256],
                             [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


def onclick4():
    def on_change(val):
        print(val)
        alpha = val / 100
        beta = (1.0 - alpha)
        result = cv2.addWeighted(img1, alpha, img2, beta, 0.0)
        cv2.imshow('blend', result)

    img1 = cv2.imread(r'C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\kremowka.png')
    img2 = cv2.imread(r'C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\KampusUwB1.jpg')

    img1 = cv2.resize(img1, (400, 400))
    img2 = cv2.resize(img2, (400, 400))

    cv2.imshow('blend', img2)

    cv2.createTrackbar('slider', 'blend', 0, 100, on_change)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def onclick5():
    # Suma image
    img_bgr = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\Ratusz_Bialystok.jpg",
                         cv2.IMREAD_UNCHANGED)
    img_bgr2 = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\KampusUwB1.jpg",
                          cv2.IMREAD_UNCHANGED)

    a = img_bgr.astype(float) / 255
    b = img_bgr2.astype(float) / 255
    np.zeros_like(a)
    ab = (a + b)

    # Display the image
    plt.imshow(ab)
    plt.show()

    # Histogram plotting of the image
    color = ('b', 'g', 'r')

    for i, col in enumerate(color):
        histr = cv2.calcHist([img_bgr2],
                             [i], None,
                             [256],
                             [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


def onclick6():
    # Odejmowanie image
    img_bgr = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\Ratusz_Bialystok.jpg",
                         cv2.IMREAD_UNCHANGED)
    img_bgr2 = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\KampusUwB1.jpg",
                          cv2.IMREAD_UNCHANGED)

    a = img_bgr.astype(float) / 255
    b = img_bgr2.astype(float) / 255
    np.zeros_like(a)
    ab = (a + b - 1)

    # Display the image
    plt.imshow(ab)
    plt.show()

    # Histogram plotting of the image
    color = ('b', 'g', 'r')

    for i, col in enumerate(color):
        histr = cv2.calcHist([img_bgr],
                             [i], None,
                             [256],
                             [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


def onclick7():
    # Roznica image
    img_bgr = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\Ratusz_Bialystok.jpg",
                         cv2.IMREAD_UNCHANGED)
    img_bgr2 = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\KampusUwB1.jpg",
                          cv2.IMREAD_UNCHANGED)

    a = img_bgr.astype(float) / 255
    b = img_bgr2.astype(float) / 255
    np.zeros_like(a)
    ab = abs(a - b)

    # Display the image
    plt.imshow(ab)
    plt.show()

    # Histogram plotting of the image
    color = ('b', 'g', 'r')

    for i, col in enumerate(color):
        histr = cv2.calcHist([img_bgr],
                             [i], None,
                             [256],
                             [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


def onclick8():
    # Mnozenie odwrotnosci image
    img_bgr = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\Ratusz_Bialystok.jpg",
                         cv2.IMREAD_UNCHANGED)
    img_bgr2 = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\KampusUwB1.jpg",
                          cv2.IMREAD_UNCHANGED)

    a = img_bgr.astype(float) / 255
    b = img_bgr2.astype(float) / 255
    np.zeros_like(a)
    ab = (1 - (1 - a) * (1 - b))

    plt.imshow(ab)
    plt.show()

    # Histogram plotting of the image
    color = ('b', 'g', 'r')

    for i, col in enumerate(color):
        histr = cv2.calcHist([img_bgr2],
                             [i], None,
                             [256],
                             [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


def onclick9():
    # Ciemniejsze image
    img_bgr = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\Ratusz_Bialystok.jpg",
                         cv2.IMREAD_UNCHANGED)
    img_bgr2 = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\KampusUwB1.jpg",
                          cv2.IMREAD_UNCHANGED)

    a = img_bgr.astype(float) / 255
    b = img_bgr2.astype(float) / 255
    np.zeros_like(a)
    mask = a >= 0.5  # generate boolean mask of everywhere a > 0.5
    ab = np.zeros_like(a)  # generate an output container for the blended image

    ab[~mask] = (a)[~mask]  # 2ab everywhere a<0.5
    ab[mask] = (b)[mask]  # else this

    plt.imshow(ab)
    plt.show()

    # Histogram plotting of the image
    color = ('b', 'g', 'r')

    for i, col in enumerate(color):
        histr = cv2.calcHist([img_bgr],
                             [i], None,
                             [256],
                             [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


def onclick10():
    # Wyłączenie image
    img_bgr = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\Ratusz_Bialystok.jpg",
                         cv2.IMREAD_UNCHANGED)
    img_bgr2 = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\KampusUwB1.jpg",
                          cv2.IMREAD_UNCHANGED)

    a = img_bgr.astype(float) / 255
    b = img_bgr2.astype(float) / 255
    np.zeros_like(a)
    ab = a + b - (2 * a * b)

    plt.imshow(ab)
    plt.show()

    # Histogram plotting of the image
    color = ('b', 'g', 'r')

    for i, col in enumerate(color):
        histr = cv2.calcHist([img_bgr2],
                             [i], None,
                             [256],
                             [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


def onclick11():
    # Nakładka image
    img_bgr = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\Ratusz_Bialystok.jpg",
                         cv2.IMREAD_UNCHANGED)
    img_bgr2 = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\KampusUwB1.jpg",
                          cv2.IMREAD_UNCHANGED)

    a = img_bgr.astype(float) / 255
    b = img_bgr2.astype(float) / 255
    np.zeros_like(a)

    if a.all() < 0.5:
        ab = 2 * a * b
    else:
        ab = 1 - 2 * (1 - a) * (1 - b)

    plt.imshow(ab)
    plt.show()

    # Histogram plotting of the image
    color = ('b', 'g', 'r')

    for i, col in enumerate(color):
        histr = cv2.calcHist([img_bgr],
                             [i], None,
                             [256],
                             [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


def onclick12():
    # Ostre światło image
    img_bgr = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\Ratusz_Bialystok.jpg",
                         cv2.IMREAD_UNCHANGED)
    img_bgr2 = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\KampusUwB1.jpg",
                          cv2.IMREAD_UNCHANGED)

    a = img_bgr.astype(float) / 255
    b = img_bgr2.astype(float) / 255
    np.zeros_like(a)

    if b.all() < 0.5:
        ab = 2 * a * b
    else:
        ab = 1 - 2 * (1 - a) * (1 - b)

    plt.imshow(ab)
    plt.show()

    # Histogram plotting of the image
    color = ('b', 'g', 'r')

    for i, col in enumerate(color):
        histr = cv2.calcHist([img_bgr],
                             [i], None,
                             [256],
                             [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


def onclick13():
    # Rozcieńczanie image
    img_bgr = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\Ratusz_Bialystok.jpg",
                         cv2.IMREAD_UNCHANGED)
    img_bgr2 = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\KampusUwB1.jpg",
                          cv2.IMREAD_UNCHANGED)

    a = img_bgr.astype(float) / 255
    b = img_bgr2.astype(float) / 255
    np.zeros_like(a)

    ab = a / (a - b)

    plt.imshow(ab)
    plt.show()

    # Histogram plotting of the image
    color = ('b', 'g', 'r')

    for i, col in enumerate(color):
        histr = cv2.calcHist([img_bgr2],
                             [i], None,
                             [256],
                             [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


def onclick14():
    # Wypalanie image
    img_bgr = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\Ratusz_Bialystok.jpg",
                         cv2.IMREAD_UNCHANGED)
    img_bgr2 = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\KampusUwB1.jpg",
                          cv2.IMREAD_UNCHANGED)

    a = img_bgr.astype(float) / 255
    b = img_bgr2.astype(float) / 255
    np.zeros_like(a)

    ab = 1 - ((1 - a) / b)

    plt.imshow(ab)
    plt.show()

    # Histogram plotting of the image
    color = ('b', 'g', 'r')

    for i, col in enumerate(color):
        histr = cv2.calcHist([img_bgr],
                             [i], None,
                             [256],
                             [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


def onclick15():
    # Reflect mode image
    img_bgr = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\Ratusz_Bialystok.jpg",
                         cv2.IMREAD_UNCHANGED)
    img_bgr2 = cv2.imread(r"C:\Users\unfit\Desktop\Metody\Nowy folder Python\TransormacjaObrazow\KampusUwB1.jpg",
                          cv2.IMREAD_UNCHANGED)

    a = img_bgr.astype(float) / 255
    b = img_bgr2.astype(float) / 255
    np.zeros_like(a)

    ab = (a * a) / (1 - b)

    plt.imshow(ab)
    plt.show()

    # Histogram plotting of the image
    color = ('b', 'g', 'r')

    for i, col in enumerate(color):
        histr = cv2.calcHist([img_bgr],
                             [i], None,
                             [256],
                             [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


root = tk.Tk()
root.title("Transformacja liniowa obrazów 2D")
root.geometry("400x450")
l = tk.Label(root, text="Transformacja liniowa obrazów 2D")
l.config(font=("Courier", 14))
p = tk.Label(root, text="Paweł Waszkiewicz")
p.config(font=("Courier", 14))

l.pack()
btn1 = tk.Button(root, text="Zwykły obrazek", command=onclick1)
btn2 = tk.Button(root, text="Negacja", command=onclick2)
btn3 = tk.Button(root, text="Mnożenie", command=onclick3)
btn4 = tk.Button(root, text="Blend", command=onclick4)
btn5 = tk.Button(root, text="Suma", command=onclick5)
btn6 = tk.Button(root, text="Odejmowanie", command=onclick6)
btn7 = tk.Button(root, text="Różnica", command=onclick7)
btn8 = tk.Button(root, text="Mnożenie odwrotności", command=onclick8)
btn9 = tk.Button(root, text="Ciemniejsze", command=onclick9)
btn10 = tk.Button(root, text="Wyłączenie", command=onclick10)
btn11 = tk.Button(root, text="Nakładka", command=onclick11)
btn12 = tk.Button(root, text="Ostre światło", command=onclick12)
btn13 = tk.Button(root, text="Rozcieńczanie", command=onclick13)
btn14 = tk.Button(root, text="Wypalanie", command=onclick14)
btn15 = tk.Button(root, text="Reflect mode", command=onclick15)

btn1.pack()
btn4.pack()
btn5.pack()
btn6.pack()
btn7.pack()
btn3.pack()
btn8.pack()
btn2.pack()
btn9.pack()
btn10.pack()
btn11.pack()
btn12.pack()
btn13.pack()
btn14.pack()
btn15.pack()
p.pack()

root.mainloop()

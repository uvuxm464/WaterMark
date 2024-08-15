import numpy as np
import matplotlib.pyplot as plt
import cv2    # install package opencv-python


class LSB():
    @staticmethod
    def bitPlane(img):                             # 將灰階值根據2進位拆開，灰階值0~255，所以分成8個
        w, h = img.shape
        a = 1                                      # 2的倍數,1,2,4,8,16,32,64,128
        bitPlane = np.zeros((8, w, h))             # 建立8個，wxh的陣列
        for i in range(8):
            b = img & a                            # img和a做and運算
            b[b != 0] = 1                          # b中的值若不是0則改成1
            bitPlane[i, :, :] = b
            a = a << 1                             # a乘2
        return bitPlane.astype(np.uint8)
    @staticmethod
    def putin(image, watermark, lsb_bit):          # water放入原圖中
        w, h = watermark.shape
        bitPlane_image = LSB.bitPlane(image)
        bitPlane_watermark = LSB.bitPlane(watermark)
        output = np.zeros_like(image)              # 值設為0
        for i in range(lsb_bit):                   # 將watermark放入原圖，watermark高位元放入原圖的低位元
            bitPlane_image[i, 0:w, 0:h] = bitPlane_watermark[(8 - lsb_bit + i), 0:w, 0:h]
        for i in range(8):                         # 將前面放好的圖片乘2的倍數，依序復原圖片的灰階值
            output = output + bitPlane_image[i, :, :] * np.power(2, i)
        return output.astype(np.uint8)

    @staticmethod
    def reduction(output, lsb_bit):                # 從output取出watermark
        bitPlane_output = LSB.bitPlane(output)
        reduction = np.zeros_like(output)          # 設為0
        for i in range(8):
            if i < lsb_bit:                        # 將watermark從合成後的圖取出
                reduction = reduction + bitPlane_output[i, ...] * np.power(2, (8 - lsb_bit + i))
        return reduction.astype(np.uint8)


if __name__ == '__main__':
    image = cv2.imread(r'C:\Users\user\VSCode\Python\image processing\WaterMark\image.jpg', cv2.IMREAD_GRAYSCALE)              # 以灰階讀取圖片，路徑前的r是為了取消路徑中所有可能的轉義，ex:\n不會被解釋為換行
    watermark = cv2.imread(r'C:\Users\user\VSCode\Python\image processing\WaterMark\watermark.jpg', cv2.IMREAD_GRAYSCALE)      # 以灰階讀取圖片

    for i in range(3):
        lsb_bit = i + 1

        output = LSB.putin(image, watermark, lsb_bit)
        plt.imshow(output, cmap="gray")
        plt.xticks([]), plt.yticks([])             # 隱藏座標軸
        name = str(lsb_bit) + 'bit_out.png'
        plt.savefig(name, bbox_inches='tight', pad_inches=0)
        plt.show()

        reduction = LSB.reduction(output, lsb_bit)
        plt.imshow(reduction, cmap="gray")
        plt.xticks([]), plt.yticks([])             # 隱藏座標軸
        name = str(lsb_bit) + 'bit_reduction.png'
        plt.savefig(name, bbox_inches='tight', pad_inches=0)
        plt.show()

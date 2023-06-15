import  numpy as np

class GaussNoise():

    def gauss_noise_matrix(matrix, mu,sigma):
        # 1. 定义一个与多维矩阵等大的高斯噪声矩阵
        channel_size = len(matrix)
        height = len(matrix[0])
        width = len(matrix[0][0])
        noise_matrix = np.random.normal(mu, sigma, size=[channel_size, height, width]).astype(
            np.float32)  # 这里在生成噪声矩阵的同时将其元素数据类型转换为float32
        # print("noise_matrix_element_type: {}".format(type(noise_matrix[0][0][0]))) # numpy.float32
        print(noise_matrix[0][0])  # 这里为了方便观察，只输出了第一个channel的第一行元素

        # 2. 与原来的多维矩阵相加，即可达到添加高斯噪声的效果
        matrix += noise_matrix

        # 3. 输出添加噪声后的矩阵
        print(">>>>>>>>>added gaussain noise with method 2")
        print(matrix[0][0])  # 这里为了方便观察，只输出了第一个channel的第一行元素
        return matrix


if __name__ == '__main__':
    # 生成一个三维的小数矩阵，模拟4张特征图，每一张特征图有20行，15列
    matrix = np.random.random(size=[4, 20, 15])
    # print(type(matrix)) # numpy.ndarray
    # print(type(matrix[0][0][0])) # numpy.float64

    # 转换成numpy.float32
    matrix_new = matrix.astype(np.float32)
    # print(type(matrix_new[0][0][0])) # numpy.float32

    print(">>>>>>>>>before adding gaussain noise")
    print(matrix_new[0][0])

    # 加入高斯噪声
    gauss_noise_matrix(matrix_new,0.06, 0.20)

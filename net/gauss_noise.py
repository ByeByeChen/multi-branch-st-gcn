import  numpy as np

class GaussNoise():

    def gauss_noise_matrix(matrix, mu,sigma):
        # 1. 定义一个与多维矩阵等大的高斯噪声矩阵
        channel_size = len(matrix)
        height = len(matrix[0])
        width = len(matrix[0][0])
        noise_matrix = np.random.normal(mu, sigma, size=[channel_size, height, width]).astype(
            np.float32)  
        print(noise_matrix[0][0])  
        matrix += noise_matrix
        print(">>>>>>>>>added gaussain noise with method 2")
        print(matrix[0][0])  
        return matrix

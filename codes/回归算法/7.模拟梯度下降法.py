import numpy as np
import matplotlib.pyplot as plt

# 创建绘图数据
plot_X = np.linspace(-1, 6, 141)
plot_Y = (plot_X - 2.5) ** 2 - 1

# 绘制代价函数图像
plt.plot(plot_X, plot_Y)
plt.xlabel('theta')
plt.ylabel('J(theta)')
plt.title('Cost Function')
plt.show()


def gradient(theta):
    """
    计算梯度（导数）

    参数：
    theta - 当前的 theta 值

    返回值：
    梯度值
    """
    return 2 * (theta - 2.5)


def cost_function(theta):
    """
    计算代价函数值

    参数：
    theta - 当前的 theta 值

    返回值：
    代价函数值
    """
    return (theta - 2.5) ** 2 - 1


def gradient_descent(theta_initial, eta=0.1, epsilon=1e-8):
    """
    执行梯度下降算法

    参数：
    theta_initial - 初始的 theta 值
    eta - 学习率，默认为 0.1
    epsilon - 收敛精度，默认为 1e-8

    返回值：
    theta_history - 梯度下降过程中 theta 的历史值
    """
    theta = theta_initial  # 初始值
    theta_history = [theta]  # 保存 theta 的历史值

    while True:
        gradient_value = gradient(theta)
        last_theta = theta
        theta = theta - eta * gradient_value
        theta_history.append(theta)

        if abs(cost_function(theta) - cost_function(last_theta)) < epsilon:
            break

    return theta_history


def gradient_descent_2(theta_initial, eta=0.1, epsilon=1e-8, max_iters=1000):
    """
    执行梯度下降算法（带最大迭代次数限制）

    参数：
    theta_initial - 初始的 theta 值
    eta - 学习率，默认为 0.1
    epsilon - 收敛精度，默认为 1e-8
    max_iters - 最大迭代次数，默认为 1000

    返回值：
    theta_history - 梯度下降过程中 theta 的历史值
    """
    theta = theta_initial  # 初始值
    theta_history = [theta]  # 保存 theta 的历史值
    iterations = 0  # 迭代次数

    while iterations < max_iters:
        gradient_value = gradient(theta)
        last_theta = theta
        theta = theta - eta * gradient_value
        theta_history.append(theta)

        if abs(cost_function(theta) - cost_function(last_theta)) < epsilon:
            break

        iterations += 1

    return theta_history


def visualize_oscillating_gradient():
    """
    可视化不同初始 theta 值和学习率下的梯度下降路径
    """
    initial_thetas = [0, 5]  # 选择初始 theta 值
    etas = [0.8, 0.01]  # 不同的学习率
    max_iters = 1000  # 最大迭代次数

    for eta in etas:
        for theta_initial in initial_thetas:
            theta_history = gradient_descent_2(theta_initial, eta, max_iters=max_iters)

            # 创建单独的图
            plt.figure()
            plt.plot(plot_X, cost_function(plot_X), label='Cost Function')
            plt.plot(np.array(theta_history), cost_function(np.array(theta_history)), marker='+',
                     label=f'Initial theta = {theta_initial}, eta = {eta}')
            plt.xlabel('theta')
            plt.ylabel('J(theta)')
            plt.title(f'Gradient Descent Path\nInitial theta = {theta_initial}, eta = {eta}')
            plt.legend()
            plt.show()


# 执行可视化
visualize_oscillating_gradient()

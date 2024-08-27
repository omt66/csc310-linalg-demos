import numpy as np
import matplotlib.pyplot as plt

print("Linear Algebra Demo1")

def demo1():
    print("Demo1")
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    print("a = ", a)
    print("b = ", b)

    c = a + b
    print("c = a + b = ", c)


def demo2():
    print("Demo2: Matrix Operations")
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    print("A = ", A)
    print("B = ", B)
    C = np.dot(A, B)
    print("C = A * B = ", C)

def demo3():
    print("Let's solve some linear equations")
    # Ax = b
    A = np.array([[2, 1], [1, 3]])
    b = np.array([4, 5])

    print("A = ", A)
    print("b = ", b)

    print("Let's solve Ax = b")
    x = np.linalg.solve(A, b)
    print("x = ", x)


def demo4():
    print("How about finding some eigenvalues and eigenvectors")
    A = np.array([[-2, -7], [1, 6]])
    eigen_values, eigen_vectors = np.linalg.eig(A)
    print("Eigen Values = ", eigen_values)
    print("Eigen Vectors = ", eigen_vectors)

def demo5():
    A = np.array([[2, 1], [1, 2]])
    x, y = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
    pos = np.dstack((x, y))
    transformed = np.einsum('ij,mni->mnj', A, pos)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(x, y, color="green", alpha=0.5)
    ax1.set_title("Original Grid")

    ax2.scatter(transformed[:, :, 0], transformed[:, :, 1], color="blue", alpha=0.25)
    ax2.set_title("Transformed Grid")

    plt.tight_layout()
    plt.show()


    # print(x)
    # print(y)

# demo1()
# demo2()
# demo3()
# demo4()
demo5()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
import math
from scipy import interpolate

Ref_prof = open('Reference_profiles_1.txt', 'r')
List_ref_prof = []
Ref_prof.readline()
for x in Ref_prof:
    List_ref_prof.append(x)
Ref_prof.close()


# функция поворота облака точек на три эйлеровых угла относительно центра вращения
# со сдвигом shift


def rot_point(point_coord, alpha, beta, gamma, rot_center, shift):
    alpha, beta, gamma = np.radians(alpha), np.radians(beta), np.radians(gamma)
    M_alpha = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
    M_beta = np.array([[np.cos(beta), 0, -np.sin(beta)], [0, 1, 0], [np.sin(beta), 0, np.cos(beta)]])
    M_gamma = np.array([[1, 0, 0], [0, np.cos(gamma), -np.sin(gamma)], [0, np.sin(gamma), np.cos(gamma)]])
    return np.asarray(shift) + np.asarray(rot_center) + M_alpha.dot(
        M_beta.dot(M_gamma.dot(np.asarray(point_coord) - np.asarray(rot_center))))


# Три функции реализующие интерполяцию точек кубическими сплайнами


def point_spline(s_points, y_points, s_new, k_i):
    tck = interpolate.splrep(s_points, y_points, k=k_i)
    return interpolate.splev(s_new, tck)


def s_i(i, n, m_i):
    a = list(np.linspace(i / (n - 1), (i + 1) / (n - 1), m_i + 2))
    del a[0]
    del a[m_i]
    return a


def spline_list(list_of_ref_prof, s_cur):
    n = len(list_of_ref_prof)  # number of ref profiles
    k = len(list_of_ref_prof[0])  # number of points in each
    s_0 = np.linspace(0, 1, n)
    new_points = []
    for j in range(k):
        x_list = []
        y_list = []
        z_list = []
        for i in range(n):
            x_list.append(list_of_ref_prof[i][j][0])
            y_list.append(list_of_ref_prof[i][j][1])
            z_list.append(list_of_ref_prof[i][j][2])

        new_points.append([float(point_spline(s_0, x_list, s_cur, 3)),
                           float(point_spline(s_0, y_list, s_cur, 3)),
                           float(point_spline(s_0, z_list, s_cur, 1))])
    return new_points


def naca(name_profile, up_or_down, x):
    L = int(name_profile[5:6])
    if name_profile[6:8] == '30':
        m = 0.2025
        k_1 = 15.957

        ###---Y_coordinate_of_camber_line_and_its_derivative---###
        y_c = 0.
        dy_c = 0.
        if 0 <= x < m:
            y_c = (k_1 / 6) * (pow(x, 3) - 3 * m * pow(x, 2) +
                               pow(m, 2) * (3 - m) * x)
            dy_c = (k_1 / 6) * (3 * pow(x, 2) - 6 * m * x +
                                pow(m, 2) * (3 - m))
        elif m <= x <= 1:
            y_c = (k_1 * pow(m, 3) / 6) * (1 - x)
            dy_c = -(k_1 * pow(m, 3) / 6)
        else:
            print('Error, x out of range')
        y_c = (L / 2) * y_c
        dy_c = (L / 2) * dy_c

        ###---Y_coordinate_of_thickness_line---###
        t = int(name_profile[9:11]) / 100
        a_0 = 0.2969
        a_1 = -0.126
        a_2 = -0.3516
        a_3 = 0.2843
        a_4 = -0.1015
        y_t = (t / 0.2) * (a_0 * pow(x, 0.5) + a_1 * x +
                           a_2 * pow(x, 2) + a_3 * pow(x, 3) +
                           a_4 * pow(x, 4))

        ###---Y_coordinate---###
        theta = math.atan(dy_c)
        if up_or_down == 'up':
            x_u = x - y_t * math.sin(theta)
            y_u = y_c + y_t * math.cos(theta)
            return [x_u, y_u]
        elif up_or_down == 'down':
            x_d = x + y_t * math.sin(theta)
            y_d = y_c - y_t * math.cos(theta)
            return [x_d, y_d]
        else:
            print('Error! No "up" or "down" is written')
            return 0
    else:
        print('We need another coefficients')
        return 0


class Airfoil:
    def __init__(self, order_number_of_profile):
        self.num_profile = order_number_of_profile
        if self.num_profile >= len(List_ref_prof):
            print('No such reference profile')

    def info(self):
        return List_ref_prof[self.num_profile].split('\t')

    def name_profile(self):
        return self.info()[2]

    def phi(self):
        return float(self.info()[3])

    def beta(self):
        return float(self.info()[4])

    def psi(self):
        return float(self.info()[5])

    def x_0(self):
        return float(self.info()[6])

    def y_0(self):
        return float(self.info()[7])

    def z_0(self):
        return float(self.info()[8])

    def point_cloud(self, n):
        k = (n - 1) // 2  # n - always odd number of points
        cloud = []
        x_coord = np.linspace(0, 1, k + 2)
        if self.name_profile()[0:4] == 'NACA':
            for i in range(k, 0, -1):
                cloud.append(naca(self.name_profile(), 'down', pow(x_coord[i], 1)))  # x coordinate to the power of 2,
                # to move points closer to the front edge
            for i in range(1, k+2):
                cloud.append(naca(self.name_profile(), 'up', pow(x_coord[i], 1)))  # same
            return cloud
        else:
            print('Error, not NACA airfoil profile')
            return 0


class Aero_surface:
    def __init__(self, total_profiles, n):
        self.total_profiles = total_profiles  # Number of all reference profiles
        self.N = n  # Number of points in each profile

    def point_cloud_total(self, m):
        if len(m) != self.total_profiles - 1:
            print('Incorrect length of intermediate profiles list')
        else:
            P_Cloud = []
            for i in range(self.total_profiles):
                Airfoil_cur = Airfoil(i)  # создаем экземпляр профиля i
                cloud_2d = Airfoil_cur.point_cloud(self.N)  # Создаем облако точек соответств этому профилю
                for j in range(self.N):
                    cloud_2d[j].append(Airfoil_cur.z_0())  # добавляем каждой точке коорд z, соотв опорному сечению
                    cloud_2d[j] = list(rot_point(cloud_2d[j],
                                                 Airfoil_cur.phi(), Airfoil_cur.beta(), Airfoil_cur.psi(),
                                                 [Airfoil_cur.x_0(), Airfoil_cur.y_0(), Airfoil_cur.z_0()],
                                                 [0, 0, 0]))  # поворачиваем все точки опорного профиля на заданные углы
                P_Cloud.append(cloud_2d)

            Cloud_new = []
            for i in range(0, self.total_profiles - 1):
                for k in range(m[i]):
                    Cloud_new.append(spline_list(P_Cloud, s_i(i, self.total_profiles, m[i])[k]))  # создаем все точки
                    # промежуточных сечений

                    # print("##1 ", s_i(i, self.total_profiles, m[i])[k])

            step = 1
            for i in range(0, self.total_profiles - 1):
                for k in range(m[i]):
                    P_Cloud.insert(step + k, Cloud_new[k + step - i - 1])  # вставляем все точки промежуточных сечений
                step = step + m[i] + 1
                print("step ", step)

        return P_Cloud


class Propeller:
    def __init__(self, n_blades, n_levels, ini_blade):
        self.n_blades = n_blades  # Number of blades in one surface
        self.n_levels = n_levels  # Number of propeller surfaces
        self.ini_blade = ini_blade  # Point cloud initial

    def blade_multiply(self):
        result = []
        for level in range(1, self.n_levels + 1):
            propeller_lvl = []
            angles = np.array(np.arange(0, 360, 360 / self.n_blades))
            for angle in angles:
                new_blade = []
                for section in self.ini_blade:
                    new_section = []
                    for point in section:
                        new_point = rot_point(point, (level - 1) * 180, angle, 0, [0, 0, 0],
                                              [0, 0 + (level - 1) * 10, 0])
                        new_section.append(new_point)
                    new_blade.append(new_section)
                propeller_lvl.append(new_blade)
            result += propeller_lvl
        return result


# first = Airfoil(4)
# print(first.info())
# print(first.name_profile())
# print(first.beta())
# print(first.point_cloud(20))

blade = Aero_surface(len(List_ref_prof), 31)
a = blade.point_cloud_total([0, 1, 4, 12])
blade_lvl = Propeller(3, 2, a)

b = blade_lvl.blade_multiply()

# print("len(b) ", len(b))
xx = []
yy = []
zz = []
# buf = []

# ---Recording to file---#
output_list = open('output_list.txt', 'w')

for i in range(len(b)):
    for j in range(len(b[i])):
        for k in range(len(b[i][j])):
            print(b[i][j][k][0], b[i][j][k][1], b[i][j][k][2], file=output_list, sep='\t')

            xx.append(b[i][j][k][0])
            yy.append(b[i][j][k][1])
            zz.append(b[i][j][k][2])

output_list.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xx, yy, zz, marker='o')
ax.set_aspect('auto')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.xlim([-15, 15])
plt.ylim([-15, 15])

plt.show()

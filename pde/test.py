import numpy as np


d1 = 5
d2 = 1
dt = 0.001
u0 = np.array([1., 2])
p0 = np.array([1., 2])

u_face_forward = d2/(d1 + d2) * u0[0] + d1/(d1+d2) * u0[1]
p_face_forward = d2/(d1 + d2) * p0[0] + d1/(d1+d2) * p0[1]
# print(f'{u_face_forward = }, {p_face_forward = }')
u_face_reverse = d1/(d1 + d2) * u0[1] + d2/(d1+d2) * u0[0]
p_face_reverse = d1/(d1 + d2) * p0[1] + d2/(d1+d2) * p0[0]

u1, p1 = np.zeros_like(u0), np.zeros_like(p0)

u1[0] = u0[0] - dt * (p_face_forward) / (2 * d1)
u1[1] = u0[1] - dt * (- p_face_reverse)   / (2 * d2)

p1[0] = p0[0] - dt * u_face_forward     / (2 * d1)
p1[1] = p0[1] - dt * (0 - u_face_reverse)   / (2 * d2)

E0 = 0.5 * (2 * d1 * u0[0] ** 2 + 2 * d2 * u0[1] ** 2) + 0.5 * (2 * d1 * p0[0] ** 2 + 2 * d2 * p0[1] ** 2)
E1 = 0.5 * (2 * d1 * u1[0] ** 2 + 2 * d2 * u1[1] ** 2) + 0.5 * (2 * d1 * p1[0] ** 2 + 2 * d2 * p1[1] ** 2)

E0, E1 = float(E0), float(E1)
print(f'face: {np.stack([u_face_forward, p_face_forward])}')
print(f'{u1 = }, {p1 = }')
print(f'{E0 = }, {E1 = }')
dEdt = (E1 - E0) / dt
print(f'{dEdt = }')

a = d2 / (d1 + d2)
dEdt_pred = - 2 * a * u0[0] * p0[0] + 2 * (1 - a) * p0[1] * u0[1] + (2 * a - 1) * (p0[0] * u0[1] + p0[1] * u0[0])
dEdt_pred = dEdt_pred.item()
print(f'{dEdt_pred = }')


# Total mass:
#print(f'{(d1 * u1[0] ** 2 + d2 * u1[1] ** 2)}, {(p1[0] ** 2 + p1[1] ** 2)}')


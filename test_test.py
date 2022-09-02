from jplephem.spk import SPK
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
kernel = SPK.open('de421.bsp')

AU = 149597870691 #m
Ms = 1.9891e30 #kg
G = 6.67e-11 * AU**-3 * 86400**2 #in AU, kg, day
epsilon = np.radians(23.4392911)
k = 2*np.pi/365

def gen_phat(ra_list, dec_list):
    ra_list = np.radians(ra_list)
    dec_list = np.radians(dec_list)
    phat_list = []
    for i in range(1,ra_list.size):
        x = np.cos(dec_list[i])*np.cos(ra_list[i])
        y = np.cos(dec_list[i])*np.sin(ra_list[i])
        z = np.sin(dec_list[i])
        phat_list.append((x,y,z))
    return np.array(phat_list)

def laplace(ra_list, dec_list, t_list):
    phat_list = gen_phat(ra_list, dec_list)
    phat = phat_list[0]
    phatd = (phat_list[1] - phat_list[0]) / k / (t_list[1] - t_list[0])
    phatd2 = (phat_list[-1] - phat_list[-2]) / k / (t_list[-1] - t_list[-2])
    phatdd = (phatd2 - phatd) / k / (t_list[-1] - t_list[0])
    
    t = t_list[0]
    R = -1000/AU*(-kernel[0,10].compute(t) + kernel[0, 3].compute(t) + kernel[3, 399].compute(t))

    A = np.dot(phat, np.cross(phatd,accel(R)/G/Ms)) / np.dot(phat, np.cross(phatd, phatdd))
    B = -np.dot(phat, np.cross(phatd, R)) / np.dot(phat, np.cross(phatd, phatdd))
    C = .5*np.dot(phat, np.cross(accel(R)/G/Ms, phatdd)) / np.dot(phat, np.cross(phatd, phatdd))
    D = .5*np.dot(phat, np.cross(R, phatdd)) / np.dot(phat, np.cross(phatd, phatdd))

    rmag = 2.5
    diff = 1
    while diff>1e-6:
        p = A + B/rmag**3
        rnew = np.linalg.norm(p*phat-R)
        diff = np.abs(rnew - rmag)
        rmag = rnew

    pdot = C + D/rmag**3

    rvec = rotate(p*phat - R, -epsilon)

    t2 = t_list[1]
    R2 = -1000/AU*(-kernel[0,10].compute(t2) + kernel[0, 3].compute(t2) + kernel[3, 399].compute(t2))
    Rdot = (R2 - R) / k / (t_list[1] - t_list[0])
    rdot = rotate(p*phatd + pdot*phat - Rdot, -epsilon)*k
    return(np.append(rvec, rdot))
    
def accel(r):
    return -G * Ms * r / np.linalg.norm(r)**3

def integrate(t_init, t_end, dt, r0, rdot0, t_list):
    pos_list = []
    t = t_init
    r1 = r0 + rdot0 * dt + .5 * accel(r0) * dt**2
    while t < t_end:
        r2 = 2 * r1 - r0 + accel(r1) * dt**2
        if round(t, 3) in t_list:
            pos_list.append(r0)
        t += dt
        r0 = r1
        r1 = r2
    return np.array(pos_list)

#rotates ccw by an angle in the yz plane
def rotate(v, angle):
    answer = np.matmul([[1, 0, 0], 
                        [0, np.cos(angle), -np.sin(angle)],
                        [0, np.sin(angle), np.cos(angle)]], v)
    return answer

def ephem(r, t):
    R = -1000/AU*(-kernel[0,10].compute(t) + kernel[0, 3].compute(t) + kernel[3, 399].compute(t))
    r = rotate(r, epsilon)
    pvec = R + r
    phat = pvec / np.linalg.norm(pvec)
    dec = np.degrees(np.arcsin(phat[2]))
    ra = np.degrees(np.arctan(phat[1] / phat[0]))
    
    if phat[0]<0:
        ra += 180
    if phat[0]>0 and phat[1]<0:
        ra += 360

    return (ra,dec)

def chisq(ra_list, dec_list, t_list, pos_list):
    ra_chisq = 0
    dec_chisq = 0
    for i in range(len(pos_list)):
        radec = ephem(pos_list[i], t_list[i])
        ra_chisq += (ra_list[i] - radec[0])**2
        dec_chisq += (dec_list[i] - radec[1])**2
    return (ra_chisq, dec_chisq)


t_list, ra_list, dec_list = np.loadtxt('K2.csv', unpack=True, delimiter=",")
ra_list *= 15  

t_init = t_list[0]
t_end = t_list[-1]
dt = 0.001

#best_kid = laplace(ra_list, dec_list, t_list)
#best_kid = np.array([-0.15453135, -2.61253973 , 0.76929622 , 0.00498348,  0.01019746, -0.01452398])
best_kid = np.array([-1.17209649e-01, -2.63045060e+00 , 7.92863964e-01 , 1.62467712e-03   ,6.28219119e-03 ,-1.39055917e-02])
r, rdot = best_kid[0:3], best_kid[3:6]
std = np.array([0.01, 0.01, 0.01, 0.005, 0.01, 0.001])

#var = 1.1
pop = 20

pos_list = integrate(t_init, t_end, dt, r, rdot, t_list)
best_error = chisq(ra_list, dec_list, t_list, pos_list)
max_error = 0.01

print(best_kid)
print(best_error)
print()

print(np.linalg.norm(1/(+rdot**2/G/Ms - 2/r)))
#NASA values for r and v
# r0 = np.array([-1.164072106908085E-01, -2.661710538204198E+00 , 8.071760918627587E-01])
# rdot0 = np.array([7.353288412144546E-04, 4.891338042832597E-03 , -1.372041149354296E-02])
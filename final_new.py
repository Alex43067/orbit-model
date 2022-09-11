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
c = 3E8/AU*86400

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
t_list = np.round(t_list, 3)

t_init = t_list[0]
t_end = t_list[-1]
dt = 0.001

#best_kid = laplace(ra_list, dec_list, t_list)
best_kid = np.array( [-1.17209649e-01, -2.63045060e+00 , 7.92863964e-01 , 1.62467712e-03, 6.28219119e-03, -1.39055917e-02])
r = best_kid[0:3]
rdot = best_kid[3:6]
std = 0.25*np.abs(best_kid)

#var = 1.1
pop = 20

pos_list = integrate(t_init, t_end, dt, r, rdot, t_list)
best_error = chisq(ra_list, dec_list, t_list, pos_list)
max_error = 0.01

print(best_kid)
print(best_error)
print()

while best_error[0]>max_error or best_error[0]>max_error:
    offspring = np.random.normal(best_kid, std, (pop, 6))
    for kid in offspring:
        #r = kid[0:3]
        rdot = kid[3:6]

        pos_list = integrate(t_init, t_end, dt, r, rdot, t_list)
        error = chisq(ra_list, dec_list, t_list, pos_list)

        #if error[0] < best_error[0] and error[1] < best_error[1]:
        if error[0]**2 + error[1]**2 < best_error[0]**2 + best_error[1]**2:
            best_error = error
            best_kid = kid
            std = 0.25*np.abs(best_kid)

        
        print(best_kid)
        print(std)
        print(best_error)
        print()

#NASA values for r and v
# r0 = np.array([-1.164072106908085E-01, -2.661710538204198E+00 , 8.071760918627587E-01])
# rdot0 = np.array([7.353288412144546E-04, 4.891338042832597E-03 , -1.372041149354296E-02])
#t = 2459759.629

# [-1.17209649e-01 -2.63045060e+00  7.92863964e-01  1.62467712e-03
#   6.28219119e-03 -1.39055917e-02]
# [2.93024122e-02 6.57612650e-01 1.98215991e-01 4.06169280e-04
#  1.57054780e-03 3.47639793e-03]
# (0.22694472175294828, 0.038981804152990435)

#laplace: [-0.04797812 -2.1933032   0.57843411  0.02125951  0.05202568 -0.02600549]

#[-0.10914028 -2.8042367   0.99281937 -0.01118222 -0.02213113 -0.005771  ]

#[-0.03244806, -2.33878841,  0.61850394,  0.00840873,  0.02073062, -0.01536292]

#[-0.13453135 -2.61253973  0.76929622  0.00498348  0.01019746 -0.01452398]
#(5.115789994242112, 1.2348791735257705)

# -0.09802814 -2.49893683  0.71166788  0.00603069  0.01322295 -0.01448304]
# [0.02450703 0.62473421 0.17791697 0.00150767 0.00330574 0.00362076]
# (2.595195284027227, 2.5037912158152613)

# [-0.08922872 -2.47857007  0.70731037  0.00461253  0.01092952 -0.01317389]
# [0.02230718 0.61964252 0.17682759 0.00115313 0.00273238 0.00329347]
# (0.5169646720184226, 1.2966372484077295)

# [-0.05230871 -2.26390163  0.61576726  0.00431721  0.00475372 -0.01069447]
# [0.01307718 0.56597541 0.15394181 0.0010793  0.00118843 0.00267362]
# (0.377796022708526, 0.4284646645740239)

# [-0.06630597 -2.35377054  0.66240534  0.00352248  0.00541412 -0.01162664]
# [0.01657649 0.58844264 0.16560134 0.00088062 0.00135353 0.00290666]
# (0.19842912233486865, 0.32962068424278224)

# [-1.11998313e-01 -2.64518227e+00  7.94706070e-01  1.16289531e-03
#   6.05836110e-03 -1.36134746e-02]
# [0.01      0.01      0.01      0.0011629 0.001     0.001    ]
# (0.04990410190525105, 0.11498221712294482)

# [-1.17209649e-01 -2.63045060e+00  7.92863964e-01  1.62467712e-03
#   6.28219119e-03 -1.39055917e-02]
# [2.93024122e-02 6.57612650e-01 1.98215991e-01 4.06169280e-04
#  1.57054780e-03 3.47639793e-03]
# (0.22694472175294828, 0.038981804152990435)

# [-9.55202780e-02 -2.52400330e+00  7.37808111e-01  1.39134773e-03
#   3.29284774e-03 -1.20333062e-02]
# [2.38800695e-02 6.31000826e-01 1.84452028e-01 3.47836933e-04
#  8.23211934e-04 3.00832656e-03]
# (0.06792040527011003, 0.06789014400319042)
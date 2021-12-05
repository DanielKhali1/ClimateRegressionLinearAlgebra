from numpy import *
import pandas as pd
from matplotlib.pyplot import *
from Functions import *
import warnings
warnings.filterwarnings("ignore")

dat = array(pd.read_csv('Data.csv',header=None))
dat.shape
ids_1 = sorted(set(dat[:,1]))

#list of 
xPositions = []
yPositions = []
heights = []


# for each unique ID (time series)
for i in range(size(ids_1)):
    # append x position of time series to a list
    xPositions.append(dat[dat[:,1] == ids_1[i],3][0])
    # append y position of time series to a list
    yPositions.append(dat[dat[:,1] == ids_1[i],4][0])
    
    t = dat[dat[:,1] == ids_1[i],9]  ## time instances corresponding to this id
    h = dat[dat[:,1] == ids_1[i],15] ## corresponding height measurements
    temp_dat = np.concatenate((t.reshape(-1,1),h.reshape(-1,1)),axis = 1)

    p = 4;q=2
    [n,lamb,sigmasq] = full_search_nk(temp_dat,p,q)
    c = n+p
    U = Kno_pspline_opt(temp_dat,p,n)
    B = Basis_Pspline(n,p,U,temp_dat[:,0])
    P = Penalty_p(q,c)
    theta = np.linalg.solve(B.T.dot(B) + lamb*P, B.T.dot(temp_dat[:,1].reshape(-1,1)))
    ### Getting mean of the prediction
    num = 200
    xpred = linspace(temp_dat[0,0],temp_dat[-1,0],num)
    Bpred = Basis_Pspline(n,p,U,xpred)
    ypred1 = Bpred.dot(theta)
    std_t1,std_n1 = Var_bounds(temp_dat,Bpred,B,theta,P,lamb)
    
    xpred_2006 = np.array([2006])
    Bpred_2006 = Basis_Pspline(n,p,U,xpred_2006)
    ypred_2006 = Bpred_2006.dot(theta)
    # append height at x & y to a point
    heights.append(ypred_2006[0][0])
    
    print((i+1), '/' , len(ids_1))
    #print('height at t = 2006: x = ', xPositions[len(xPositions)-1], ' y = ', yPositions[len(yPositions)-1], ' ' ,ypred_2006[0][0] , ' ', i+1 , '/' , len(ids_1))
    
df = pd.DataFrame({'x':xPositions, 'y':yPositions, 'height':heights})
df.to_csv('dat_2006.csv', index=False)


print(df)

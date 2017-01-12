import matplotlib.pyplot as plt

def plotData(x,y,fig_num):
    plt.figure(fig_num)
    plt.plot(x,y,'xr')
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.ion()



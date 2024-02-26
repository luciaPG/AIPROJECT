from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("C:/Users/User/eclipse-workspace/AIPROJECT/data/test_clean.csv")

df["lot_size"]*=0.093 # change to squared meters

list_size = [x for x in df["lot_size"]]
list_price = [x/1000000 for x in df["price"]]


def plot_price_size():
    plt.plot(list_size,list_price)
    plt.title("Price according total size")
    plt.xlabel("Squared meters")
    plt.ylabel("Millions of $")

    plt.show()
    # there is no progressive relation between the price and the size, so it must depend on more factors


# Plots the relation between price and size of houses with less than 2500 sqft
def small_price_size():
    sizes = [x for x in df["lot_size"] if x < 2500]
    print(len(sizes))
    prices = [x/1000000 for x in df.loc[df["lot_size"]<2500, "price"]]
    #df.loc[df_train['lot_size_units'] == 'acre', 'lot_size'] *= 43560

    plt.plot(sizes,prices)
    plt.title("Price according total size")
    plt.xlabel("Squared meters")
    plt.ylabel("Millions of $")

    plt.show()

#Plots the relation between price and size in houses that are located
#in the zip code given as a parameter
def plot_price_size_code(code):
    sizes = [x for x in df.loc[df["zip_code"]==code, "lot_size"]]
    prices = [x/1000000 for x in df.loc[df["zip_code"]==code, "price"]]
    plt.plot(sizes, prices, marker = ".")
    title = "Price according total size, code:"+str(code)
    plt.title(title)
    plt.xlabel("Squared meters")
    plt.ylabel("Millions of $")

    plt.show()

#Plots the relation between price and size of the first n houses
#one line represents houses with 1 bath, the other one houses
# with 2 baths
def plot_price_size_baths(n):
    sizes1 = [x for x in df.loc[df["baths"]==1, "lot_size"]]
    prices1 = [x/1000000 for x in df.loc[df["baths"]==1, "price"]]

    sizes2 = [x for x in df.loc[df["baths"]==2, "lot_size"]]
    prices2 = [x/1000000 for x in df.loc[df["baths"]==2, "price"]]



    plt.style.use("ggplot")
    plt.plot(sizes1[:n], prices1[:n], marker = ".")
    plt.plot(sizes2[:n],prices2[:n], marker = ".")
    plt.title("Price according total size")
    plt.xlabel("Squared meters")
    plt.ylabel("Millions of $")
    plt.legend(["1 baths", "2 baths"])
    plt.grid(True)


    plt.show()


def plot_scatter_baths_price_size():
    sizes1 = [x for x in df.loc[df["baths"]==1, "lot_size"]]
    prices1 = [x/1000000 for x in df.loc[df["baths"]==1, "price"]]

    sizes2 = [x for x in df.loc[df["baths"]==2, "lot_size"]]
    prices2 = [x/1000000 for x in df.loc[df["baths"]==2, "price"]]

    sizes3 = [x for x in df.loc[df["baths"]==3, "lot_size"]]
    prices3 = [x/1000000 for x in df.loc[df["baths"]==3, "price"]]

    plt.style.use("ggplot")

    plt.scatter(sizes1, prices1, marker='o', c='b')
    plt.scatter(sizes2,prices2, marker="o")
    plt.scatter(sizes3,prices3, marker="o")


    plt.title('Price according to size')
    plt.xlabel('Squared feet')
    plt.ylabel('Millions of $')
    plt.legend(["1 baths","2 baths", "3 baths"])
    plt.grid(True)
    plt.show()

def scatter_price_size_zipCode(z1, z2, z3):
    sizes1 = [x for x in df.loc[df["zip_code"]==z1, "lot_size"]]
    prices1 = [x/1000000 for x in df.loc[df["zip_code"]==z1, "price"]]

    sizes2 = [x for x in df.loc[df["zip_code"]==z2, "lot_size"]]
    prices2 = [x/1000000 for x in df.loc[df["zip_code"]==z2, "price"]]

    sizes3 = [x for x in df.loc[df["zip_code"]==z3, "lot_size"]]
    prices3 = [x/1000000 for x in df.loc[df["zip_code"]==z3, "price"]]

    plt.style.use("ggplot")

    plt.scatter(sizes1, prices1, marker='o', c='b')
    plt.scatter(sizes2,prices2, marker="o")
    plt.scatter(sizes3,prices3, marker="o")


    plt.title('Price according to size')
    plt.xlabel('Squared feet')
    plt.ylabel('Millions of $')
    plt.legend([str(z1),str(z2), str(z3)])
    plt.grid(True)
    plt.show()

def J(x,y,m,theta_0,theta_1):
    returnValue = 0
    h = lambda theta_0,theta_1,x: theta_0 + theta_1*x
    for i in range(m):
        pred = h(theta_0,theta_1,x[i])
        returnValue += (pred-y[i])**2  # prediction for a given x example
    returnValue = returnValue/(2*m)   # loss for that given x example
    return returnValue ##the mean loss fo the batch of the dataset

def grad_J(x,y,m,theta_0,theta_1):  # Calculate the grdient for the batch data
    returnValue = np.array([0.,0.])  # A list of teo number
    h = lambda theta_0,theta_1,x: theta_0 + theta_1*x
    for i in range(m): # For every element in the data
        returnValue[0] += (h(theta_0,theta_1,x[i])-y[i]) # the gradient for thetha 0 with respect to
        returnValue[1] += (h(theta_0,theta_1,x[i])-y[i])*x[i]
    returnValue = returnValue/(m)
    return returnValue

def compute_price_size_gradient(n):

    sizes1 = [x for x in df.loc[df["lot_size"]<=2500, "lot_size"]]
    prices1 = [x/1000000 for x in df.loc[df["lot_size"]<=2500, "price"]]

    x = sizes1
    y = prices1
    m = len(x)

    theta_old = np.array([0.,0.])
    theta_new = np.array([1.,1.]) # The algorithm starts at [1,1]
    n_k = 0.001 # step size, note this is constant
    precision = 0.01
    num_steps = 0
    s_k = float("inf")

    while np.linalg.norm(s_k) > precision:
        num_steps += 1
        theta_old = theta_new
        s_k = -grad_J(x,y,m,theta_old[0],theta_old[1])
        theta_new = theta_old + n_k * s_k*0.001


    print("Local minimum occurs where:")

    print("theta_0 =", theta_new[0])
    print("theta_1 =", theta_new[1])
    print("This took",num_steps,"steps to converge")

    return [sizes1[:n], prices1[:n], theta_new]

def scatter_price_size_gradient():

    res = compute_price_size_gradient(4) #[sizes1, prices1, theta_new]

    xx = np.linspace(0,2500,2500)
    h = lambda theta_0,theta_1,x: theta_0 + theta_1*x

    plt.style.use("ggplot")

    plt.scatter(res[0], res[1], marker='o', c='b')
    plt.plot(xx,h(res[2][0],res[2][1],xx))
    plt.title('Price according to size')
    plt.xlabel('Squared feet')
    plt.ylabel('Millions of $')
    plt.grid(True)
    #plt.show()
    for x in xx:
        print("x:", x, " h: ", h(x))



#plot_price_size()
#small_price_size()
#plot_price_size_code(98117)
#plot_price_size_baths(15)
#plot_scatter_baths_price_size()
#scatter_price_size_zipCode(98119, 98106, 98125)
scatter_price_size_gradient()
#compute_price_size_gradient()
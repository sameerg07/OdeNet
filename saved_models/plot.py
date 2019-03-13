import matplotlib.pyplot as plt
import pandas as pd
import csv

# # RELU vs SELU with VAL ACC for Faces95
# df1 = pd.read_csv("./faces95/RELUepochVaccuracy.csv")
# df2 = pd.read_csv("./faces95/SELUepochVaccuracy.csv")

# # RELU vs SELU with VAL LOSS for Faces95
# df1 = pd.read_csv("./faces95/RELUepochVloss.csv")
# df2 = pd.read_csv("./faces95/SELUepochVloss.csv")

# # RELU vs SELU with VAL ACC for Faces96
df1 = pd.read_csv("./faces96/RELUepochVaccuracy.csv")
df2 = pd.read_csv("./faces96/SELUepochVaccuracy.csv")

# # RELU vs SELU with VAL LOSS for Faces96
# df1 = pd.read_csv("./faces96/RELUepochVloss.csv")
# df2 = pd.read_csv("./faces96/SELUepochVloss.csv")

fields = ["epochs","accuracy"]
epochs = [i for i in range(1,51)]
df1Values = list(df1.accuracy)
df2Values = list(df2.accuracy)

reluPlot = plt.plot(epochs,df1Values,label='RELU')
seluPlot = plt.plot(epochs,df2Values,label='SELU')

plt.legend(loc="upper left")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("SELU vs RELU on Testing Accuracy")
# plt.title("SELU vs RELU on Testing Loss")
plt.savefig("faces96Accuracy.png")
# plt.savefig("faces96Loss.png")
plt.show()
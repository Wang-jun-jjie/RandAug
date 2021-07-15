import matplotlib.pyplot as plt
import numpy as np

# first 40 epochs
x=range(1,41)
y_train=[21.8962, 28.0373, 32.4898, 37.1199, 41.3762, 45.6433, 49.2569, 52.3545, 55.6977, 58.5651, 
         61.0292, 63.5336, 65.2345, 67.4808, 68.9130, 70.2308, 71.6290, 72.6981, 73.6961, 74.6246, 
         75.4449, 76.3194, 76.9559, 77.3946, 78.4035, 78.9890, 79.6070, 79.7522, 80.3810, 80.7564, 
         80.8630, 81.3960, 81.7699, 82.2658, 82.3183, 82.5516, 83.4198, 83.1016, 83.6732, 83.8432,]

y_val  =[30.9914, 39.1810, 42.4569, 46.7672, 52.2845, 60.3017, 60.9483, 65.4741, 67.8448, 69.0517, 
         73.1897, 72.8448, 75.6466, 78.1034, 79.7414, 80.2586, 80.6897, 79.4397, 82.2414, 81.8534, 
         84.4828, 83.8362, 84.3534, 83.8362, 85.4310, 85.9483, 86.2931, 86.2500, 87.7155, 86.8103, 
         86.4655, 87.8879, 88.6207, 86.0776, 88.3621, 89.0517, 87.4569, 86.9397, 88.6207, 88.9655,]

plt.plot(x, y_train, label='training acc', color='tab:blue')
plt.plot(x, y_val, label='testing acc', color='tab:orange')

plt.title('Classification performance on Very Restricted Imagenet')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.legend(loc = 'lower right')

plt.show()
plt.savefig('plot.png')
plt.close()
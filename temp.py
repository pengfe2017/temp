import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
# create a two column matrix
total_section_num = 20
x = np.array([])
y = np.array([])

a_list = np.linspace(40, 100, 20)

for idy in range(total_section_num):
    if idy < 1:
        x_step1 = np.arange(0, 100, 1)
    else:
        x_step1 = np.arange(x[-1], x[-1]+100, 1)

    y_step1 = []
    for idx, content in enumerate(x_step1):
        a = a_list[idy]
        if idx < 1:
            if idy < 1:
                y0 = 0
            else:
                y0 = y[-1]
            #y0 = 0
            b = y0 - a*content
        y_content = a*content + b
        y_step1.append(y_content)

    x_step2 = np.arange(x_step1[-1], x_step1[-1]+100, 1)
    y_step2 = []
    for idx, content in enumerate(x_step2):
        a = 1
        if idx < 1:
            y0 = y_step1[-1]
        y_content = y0
        y_step2.append(y_content)

    x = np.hstack((x, x_step1, x_step2))
    y_step1 = np.array(y_step1)
    y = np.hstack((y, y_step1, y_step2))

plt.plot(x, y, "ro")
plt.grid(True)

plt.show()

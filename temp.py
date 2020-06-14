from scipy import signal as sg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
# create a signal with my own code
total_section_num = 20
x = np.array([])
y = np.array([])

a_list = np.linspace(40, 100, 20)

for idy in range(total_section_num):
    if idy < 1:
        x_step1 = np.arange(0, 100, 1)
    else:
        x_step1 = np.arange(x[-1]+1, x[-1]+1+100, 1)

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

    x_step2 = np.arange(x_step1[-1]+1, x_step1[-1]+1+100, 1)
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

plt.plot(x, y, "b*")
plt.grid(True)

plt.show()
# %% create a signal with scipy module
t = np.linspace(0, 100, 100)
# duty here must be smaller than 1
signal_x = 3 * sg.square(2*np.pi*t, duty=0.1)

# %% create a signal with Fourier's series
t = np.linspace(0, 5, 1000)
frequency = 1
omega = 2*np.pi*frequency
# square wave


def square_wave_sum(omega, t, iteration_number):
    y_Fourier = 0
    for content in range(iteration_number):
        n = content+1
        y_Fourier = y_Fourier + np.sin(omega*t*(2*n - 1))/(2*n-1)**1
    return 4*y_Fourier/np.pi


y_square_wave = []

for content in t:
    y_square_wave_one_point = square_wave_sum(
        omega, content, iteration_number=10000)
    y_square_wave.append(y_square_wave_one_point)

plt.plot(t, y_square_wave)
plt.grid(True)
iff(x)
diff_y = np.diff(y)
slope = np.diff(y)/np.diff(x)

slope_slope = np.diff(slope)
nonezero_index = np.nonzero(abs(slope_slope) > 1e-6)
nonezero_index_reshape = np.reshape(nonezero_index, (39, 1)) + 1
#m = slope_slope[abs(slope_slope)>1e-6]
y_selected = y[nonezero_index_reshape]

plt.plot(nonezero_index_reshape, y_selected, "ro")
# %%
diff_x = np.diff(x)
diff_y = np.diff(y)
slope = np.diff(y)/np.diff(x)

slope_slope = np.diff(slope)
nonezero_index = np.nonzero(abs(slope_slope) > 1e-6)
nonezero_index_reshape = np.reshape(nonezero_index, (39, 1)) + 1
#m = slope_slope[abs(slope_slope)>1e-6]
y_selected = y[nonezero_index_reshape]

plt.plot(nonezero_index_reshape, y_selected, "ro")

# %%

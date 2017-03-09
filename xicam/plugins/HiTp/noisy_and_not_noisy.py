"""
Created on Feb 7 2017

@author: Fang Ren
Contributor: Yijin Liu
"""

from scipy.signal import medfilt, savgol_filter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.stats import linregress

def func(x, *params):
    """
    create a Lorentzian fitted curve according to params
    """
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
    return y

def get_axis_limits(ax):
    return (ax.get_xlim()[1]-ax.get_xlim()[0])*0.05+ax.get_xlim()[0], (ax.get_ylim()[1]-ax.get_ylim()[0])*0.8+ax.get_ylim()[0]

path = 'C:\\Research_FangRen\\Publications\\on_the_fly_paper\\Sample_data\\'
file1 = path + 'amorphous.csv'
file2 = path + 'good_2.csv'
file3 = path + 'good_3.csv'
file4 = path + 'noisy_data_set.csv'


filter_window = 15

data1 = np.genfromtxt(file1, delimiter = ',')
Qlist1 = data1[196:550,0]
IntAve1 = data1[196:550,1]
IntAve_smoothed1 = savgol_filter(IntAve1, filter_window, 2)
noise1 = IntAve1 - IntAve_smoothed1

data2 = np.genfromtxt(file2, delimiter = ',')
Qlist2 = data2[216:570,0]
IntAve2 = data2[216:570,1]/8+800
IntAve_smoothed2 = savgol_filter(IntAve2, filter_window, 2)
noise2 = IntAve2 - IntAve_smoothed2

data3 = np.genfromtxt(file3, delimiter = ',')
Qlist3 = data3[170:500,0]
IntAve3 = data3[170:500,1]*20
IntAve_smoothed3 = savgol_filter(IntAve3,  filter_window, 2)
noise3 = IntAve3 - IntAve_smoothed3

data4 = np.genfromtxt(file4, delimiter = ',', skip_header= 1)

Qlist4 = data4[:, 1]
IntAve4 = data4[:, 2]*250000
IntAve_smoothed4 =  savgol_filter(IntAve4,  filter_window, 2)
noise4 = IntAve4 - IntAve_smoothed4
noise4 = np.nan_to_num(noise4)

Qlist5 = Qlist4
IntAve5 = data4[:, 4]*250000
Qlist6 = Qlist4
IntAve6 = data4[:, 5]*250000
Qlist7 = Qlist4
IntAve7 = data4[:, 6]*250000
Qlist8 = Qlist4
IntAve8 = data4[:, 7]*250000
Qlist9 = Qlist4
IntAve9 = data4[:, 8]*250000
Qlist10 = Qlist4
IntAve10 = data4[:, 9]*250000
Qlist11 = Qlist4
IntAve11 = data4[:, 10]*250000
Qlist12 = Qlist4
IntAve12 = data4[:, 11]*250000
Qlist13 = Qlist4
IntAve13 = data4[:, 12]*250000

IntAve_sum = (IntAve4+IntAve5+IntAve6+IntAve7+IntAve8+IntAve9+IntAve10+IntAve11+IntAve12+IntAve13)/10

IntAve_smoothed_sum = savgol_filter(IntAve_sum,  filter_window, 2)
noise_sum = IntAve_sum - IntAve_smoothed_sum
noise_sum = np.nan_to_num(noise_sum)


fig = plt.figure(1, (12,9))
fig.suptitle('data1: blue; data2: green; data3: red; data4: cyan; data4 with 10x exposure: magenta')
ax1 = plt.subplot2grid((5,2), (0,0), rowspan= 2)
ax1.plot(Qlist1, IntAve1, label = 'data 1')
ax1.plot(Qlist2, IntAve2+500, label = 'data 2')
ax1.plot(Qlist3, IntAve3+1100, label = 'data 3')
ax1.plot(Qlist4, IntAve4+1500, label = 'data4')
ax1.plot(Qlist4, IntAve_sum+1500, 'm', label = 'data4(10x e.t.)')
plt.ylabel('Intensity')
plt.xlim(1.7, 3.7)
plt.ylim(700, 3100)
ax1.annotate('(a)', xy=get_axis_limits(ax1))
# print '(a) is at ', get_axis_limits(ax1)
# plt.legend(fontsize = 12)

# ax2 = plt.subplot2grid((4,2), (1,0))
# ax2.plot(Qlist1, IntAve_smoothed1, label = 'smoothed good data 1')
# ax2.plot(Qlist2, IntAve_smoothed2, label = 'smoothed good data 2')
# ax2.plot(Qlist3, IntAve_smoothed3, label = 'smoothed good data 3')
# ax2.plot(Qlist4, IntAve_smoothed4, label = 'smoothed noisy data')
# plt.ylabel('Smoothed Intensity')
# plt.xlim(1.7, 3.7)
# plt.ylim(700, 1500)
# # plt.legend(fontsize = 12)

ax3 = plt.subplot2grid((5,2), (2,0))
ax3.plot(Qlist1, noise1, label = 'good data')
ax3.plot(Qlist2, noise2, label = 'good data 2')
ax3.plot(Qlist3, noise3, label = 'good data 3')
ax3.plot(Qlist4, noise4, label = 'noisy data')
ax3.plot(Qlist4, noise_sum, 'm', label = '10x exposure time')
plt.xlabel('Q')
plt.xlim(1.7, 3.7)
plt.ylim(-400, 400)
plt.ylabel('Noise')
ax3.annotate('(b)', xy=get_axis_limits(ax3))



guess = [0, 5, 10]
high = [0.5, 300, 1000]
low = [-0.5, 0, 0.1]
bins = np.arange(-100, 100, 0.5)


# the histogram of the noise

ax4 = plt.subplot2grid((5,2), (4,1))
n_sum, bins_sum = np.histogram(noise_sum, bins= bins)
# n1[200] = (n1[199]+n1[201])/2
ax4.bar(bins_sum[:-1], n_sum, color = 'm', edgecolor = 'none', label = '10x exposure time')
plt.xlim(-110, 110)
popt_sum, pcov_sum = curve_fit(func, bins_sum[:-1], n_sum, p0=guess, bounds = (low, high))
fit_sum = func(bins_sum[:-1], *popt_sum)
ax4.plot(bins_sum[:-1], fit_sum, 'k--', linewidth=2)
# plt.legend(fontsize = 12)
plt.ylabel('Counts')
plt.xlabel('noise')
ax4.annotate('(c$_5$)', xy=get_axis_limits(ax4))

ax5 = plt.subplot2grid((5, 2), (0,1))
n1, bins1 = np.histogram(noise1, bins= bins)
# n1[200] = (n1[199]+n1[201])/2
ax5.bar(bins1[:-1], n1, color = 'blue', edgecolor = 'none', label = 'good data 1')
plt.yscale('log')
plt.xlim(-50, 50)
plt.ylim(0, 200)
popt1, pcov1 = curve_fit(func, bins1[:-1], n1, p0=guess, bounds = (low, high))
fit1 = func(bins1[:-1], *popt1)
ax5.plot(bins1[:-1], fit1, 'k--', linewidth=2)
# plt.legend(fontsize = 12)
plt.ylabel('Counts (log scale)')
ax5.annotate('(c$_1$)', xy=(get_axis_limits(ax5)[0], 90))


ax6 = plt.subplot2grid((5, 2), (1,1))
n2, bins2 = np.histogram(noise2, bins= bins)
# n2[200] = (n2[199]+n2[201])/2
ax6.bar(bins2[:-1], n2, color = 'g', edgecolor = 'none', label = 'good data 2')
plt.yscale('log')
plt.xlim(-50, 50)
plt.ylim(0, 200)
popt2, pcov2 = curve_fit(func, bins2[:-1], n2, p0=guess, bounds = (low, high))
fit2 = func(bins2[:-1], *popt2)
ax6.plot(bins2[:-1], fit2, 'k--', linewidth=2)
# plt.legend(fontsize = 12)
plt.ylabel('Counts(log scale)')
ax6.annotate('(c$_2$)', xy=(get_axis_limits(ax6)[0], 90))

ax7 = plt.subplot2grid((5, 2), (2,1))
n3, bins3 = np.histogram(noise3, bins= bins)
# n3[200] = (n3[199]+n3[201])/2
n3 = medfilt(n3, kernel_size = 3)
ax7.bar(bins3[:-1], n3, color = 'red', edgecolor = 'none', label = 'good data 3')
plt.xlim(-50, 50)
# plt.ylim(0, 10)
popt3, pcov3 = curve_fit(func, bins3[:-1], n3, p0=guess, bounds = (low, high))
fit3 = func(bins3[:-1], *popt3)
ax7.plot(bins3[:-1], fit3, 'k--', linewidth=2)
# plt.legend(fontsize = 12)
plt.ylabel('Counts')
ax7.annotate('(c$_3$)', xy=get_axis_limits(ax7))


ax8 = plt.subplot2grid((5, 2), (3,1))
n4, bins4 = np.histogram(noise4, bins= bins)
# n4[200] = (n4[199]+n4[201])/2
ax8.bar(bins4[:-1], n4, color = 'cyan', edgecolor = 'none', label = 'noisy data')
plt.xlim(-110, 110)
# plt.ylim(0, 10)
popt4, pcov4 = curve_fit(func, bins4[:-1], n4, p0=guess, bounds = (low, high))
fit4 = func(bins4[:-1], *popt4)
ax8.plot(bins4[:-1], fit4, 'k--', linewidth=2)
# plt.legend(fontsize = 12)
plt.ylabel('Counts')
ax8.annotate('(c$_4$)', xy=get_axis_limits(ax8))

print popt1[2], popt2[2], popt3[2], popt4[2], popt_sum[2]
print popt1[1], popt2[1], popt3[1], popt4[1], popt_sum[1]


power_noise1 = np.sum(np.square(noise1))/len(noise1)
power_signal1 = np.sum(np.square(IntAve1))/len(IntAve1)
SNR1 = power_signal1/power_noise1
SNR1 = np.log10(SNR1)*10

power_noise2 = np.sum(np.square(noise2))/len(noise2)
power_signal2 = np.sum(np.square(IntAve2))/len(IntAve2)
SNR2 = power_signal2/power_noise2
SNR2 = np.log10(SNR2)*10

SNR2_2 = power_signal2/power_noise1
SNR2_2 = np.log10(SNR2_2)*10


power_noise3 = np.sum(np.square(noise3))/len(noise3)
power_signal3 = np.sum(np.square(IntAve3))/len(IntAve3)
SNR3 = power_signal3/power_noise3
SNR3 = np.log10(SNR3)*10

power_noise4 = np.sum(np.square(noise4))/len(noise4)
power_signal4 = np.sum(np.square(IntAve4))/len(IntAve4)
SNR4 = power_signal4/power_noise4
SNR4 = np.log10(SNR4)*10

power_noise_sum = np.sum(np.square(noise_sum))/len(noise_sum)
power_signal_sum = np.sum(np.square(IntAve_sum))/len(IntAve_sum)
SNR_sum = power_signal_sum/power_noise_sum
SNR_sum = np.log10(SNR_sum)*10

print SNR1, SNR2_2, SNR3, SNR4, SNR_sum

ax9 = plt.subplot2grid((5,2), (3,0), rowspan= 2)
x = np.array([np.log(1/popt1[2]), np.log(1/popt2[2]), np.log(1/popt3[2]), np.log(1/popt4[2]), np.log(1/popt_sum[2])])
y = np.array([SNR1, SNR2_2, SNR3, SNR4, SNR_sum])
ax9.plot(x[0], y[0], 'o', color = 'b')
ax9.plot(x[1], y[1], 'o', color = 'g')
ax9.plot(x[2], y[2], 'o', color = 'r')
ax9.plot(x[3], y[3], 'o', color = 'c')
ax9.plot(x[4], y[4], 'o', color = 'm')
ax9.plot(np.log(1/popt2[2]), SNR2, 's', color = 'g')
ax9.annotate('miscalculated noise', xy=(np.log(1/popt2[2]), SNR2+1), xytext=(np.log(1/popt2[2])-2, SNR2+10),
             arrowprops=dict(facecolor='black', shrink=0.05))
# ax9.plot([1]*len(np.arange(ax9.get_ylim()[0], ax9.get_ylim()[1])), np.arange(ax9.get_ylim()[0], ax9.get_ylim()[1]), 'k')
ax9.add_patch(patches.Rectangle((-3.15, 31.65), 7.15, 38.35, hatch='\\',alpha = 0.5, facecolor= 'yellow'))
# ax9.arrow(1, 55, 1, 0, head_width=2, head_length=0.5, fc='k', ec='k')
ax9.annotate('good quality data', xy=(2, 60), xytext=(-1, 53))
slope, intercept, r_value, p_value, std_err = linregress(x, y)
fit = slope *x + intercept
ax9.plot(x, fit, 'r--', label = 'fit')
plt.xlabel('log(1/Gaussian peak FWHM)')
plt.ylabel('SNR')
plt.xlim(-5, 4)
plt.ylim(20, 70)
plt.legend()
ax9.annotate('(d)', xy=get_axis_limits(ax9))


plt.savefig(path+'signal_to_noise_ratio', dpi = 600)
# plt.close('all')
#
# plt.figure(2)
# x = np.array([1/np.log(popt1[2]), 1/np.log(popt2[2]), 1/np.log(popt3[2]), 1/np.log(popt4[2]), 1/np.log(popt_sum[2])])
# y = np.array([SNR1, SNR2_2, SNR3, SNR4, SNR_sum])
# plt.plot(x[0], y[0], 'o', color = 'b')
# plt.plot(x[1], y[1], 'o', color = 'g')
# plt.plot(x[2], y[2], 'o', color = 'r')
# plt.plot(x[3], y[3], 'o', color = 'c')
# plt.plot(x[4], y[4], 'o', color = 'm')
# plt.xlabel('1/log(peak_width)')
# plt.ylabel('SNR')
#
# plt.figure(3)
# x = np.array([np.log(1/popt1[2]), np.log(1/popt2[2]), np.log(1/popt3[2]), np.log(1/popt4[2]), np.log(1/popt_sum[2])])
# y = np.array([SNR1, SNR2_2, SNR3, SNR4, SNR_sum])
# plt.plot(x[0], y[0], 'o', color = 'b')
# plt.plot(x[1], y[1], 'o', color = 'g')
# plt.plot(x[2], y[2], 'o', color = 'r')
# plt.plot(x[3], y[3], 'o', color = 'c')
# plt.plot(x[4], y[4], 'o', color = 'm')
# plt.xlabel('log(1/peak_width)')
# plt.ylabel('SNR')
#
#
# plt.figure(4)
# x = np.array([1/popt1[2], 1/popt2[2], 1/popt3[2], 1/popt4[2], 1/popt_sum[2]])
# y = np.array([SNR1, SNR2_2, SNR3, SNR4, SNR_sum])
# plt.plot(x[0], y[0], 'o', color = 'b')
# plt.plot(x[1], y[1], 'o', color = 'g')
# plt.plot(x[2], y[2], 'o', color = 'r')
# plt.plot(x[3], y[3], 'o', color = 'c')
# plt.plot(x[4], y[4], 'o', color = 'm')
# plt.xlabel('1/peak_width)')
# plt.ylabel('SNR')

print slope, intercept
import numpy as np
import HamiltonianPy as hp
import scipy.interpolate as itp
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import sys
import pdb

directory = '.'
destination = '../../paper'

def thermal_hall_with_temperature():
    import matplotlib.gridspec as mg
    from matplotlib import colors
    plt.ion()

    fig = plt.figure()
    gs = mg.GridSpec(3, 3, width_ratios=[np.sqrt(3), np.sqrt(3), np.sqrt(3)], height_ratios=[np.sqrt(3),np.sqrt(3), np.sqrt(3)])
    fig.subplots_adjust(left=0.1, right=0.975, top=0.99, bottom=0.04, hspace=0.3, wspace=0.4)

    tparams = ['0.29', '0.32', '0.308']
    uparams = ['1.2','1.2','1.0']
    tags=['a','b','c']

    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        ax.minorticks_on()
        ax.set_xlim(0.0, 0.08)
        ax.set_xticks([0.0, 0.02, 0.04, 0.06, 0.08])
        # if i==0: ticks = [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        # if i==1: ticks = [0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7]
        # if i==2: ticks = [-0.2, -0.15, -0.1, -0.05, 0.0]
        if i==0: ticks = [-2, 0, 2, 4, 6]
        if i==1: ticks = [-9, -6, -3, 0]
        if i==2: ticks = [-3, -2, -1, 0]
        ax.set_ylim(min(ticks)-0.001, max(ticks)+0.001)
        ax.set_yticks(ticks)
        for nk in (18, 24, 30):
            name = '%s/HCI_H2(1P-1P)_up_1.0_%s_0.656_1.2_%s_FBFM_THT%s(1.0,1.0).dat'%(directory, tparams[i], uparams[i], nk)
            data = np.loadtxt(name)
            ax.plot(data[:, 0], data[:, 1]/data[:, 0], label="$N_{\mathbf{k}}=%s\\times%s$"%(nk, nk))
        ax.legend(fontsize=7, loc='upper left' if i==0 else 'lower left' if i==1 else "center right")
        for tick in ax.get_xticklabels(): tick.set_fontsize(9)
        for tick in ax.get_yticklabels(): tick.set_fontsize(9)
        ax.set_xlabel("$T/t$",fontdict={'fontsize':10})
        if i==0: ax.set_ylabel("$\kappa_{xy}/T(-\\frac{k^2_Bt}{K\hbar})$", fontdict={'fontsize':12})
        ax.text(0.08, max(ticks), "($%s_1$)"%tags[i], va='top', ha='right')

        ax = fig.add_subplot(gs[1, i])
        name = '%s/HCI_H2(1P-1P)_up_1.0_%s_0.656_1.2_%s_FBFM_EB30.dat'%(directory, tparams[i], uparams[i])
        data = np.loadtxt(name)
        ax.plot(data[:,0], data[:, 1:3], color='green', lw=1.5, zorder=4)
        ax.fill_between(data[:, 0], np.max(data[:, 3:], axis=1), np.min(data[:, 3:], axis=1), color=(0.8,0.8,0.8), alpha=0.9, zorder=2)

        ax.axvline(x=10, ls=':', color='black', lw=1, zorder=0, alpha=0.5)
        ax.axvline(x=15, ls=':', color='black', lw=1, zorder=0, alpha=0.5)
        ax.axvline(x=20, ls=':', color='black', lw=1, zorder=0, alpha=0.5)
        
        if i==0: ax.text(15, 0.1, '$C=+1$', ha='center', va='center', fontsize=10, color='black')
        if i==1: ax.text(15, 0.1, '$C=-1$', ha='center', va='center', fontsize=10, color='black')
        if i==2: ax.text(15, 0.1, '$C=0$', ha='center', va='center', fontsize=10, color='black')

        ax.minorticks_on()
        ax.set_xlim(0.0, 30.0)
        ax.set_xticks(np.linspace(0, 30,7))
        ax.set_xticklabels(['$\Gamma$', '', '$K_1$', '$M$', '$K_2$','','$\Gamma$'])
        for tick in ax.get_xticklabels(): tick.set_fontsize(10)
        ax.set_xlabel('q',fontdict={'fontsize':10})

        ax.set_ylim(0.0,0.6)
        ax.set_yticks(np.linspace(0.0,0.6,7))
        ax.set_yticklabels(['0','','0.2','','0.4','','0.6'])
        for tick in ax.get_yticklabels(): tick.set_fontsize(9)
        if i==0: ax.set_ylabel('$E/t$', fontdict={'fontsize':12})
        ax.text(3.0, 0.6, '($%s_2$)'%tags[i], ha='left', va='top', color='black')

        ax = fig.add_subplot(gs[2, i])
        ax.axis('off')
        ax.axis('equal')
        ax.set_ylim(-0.5, 6.5)
        name = '%s/HCI_H2(1P-1P)_up_1.0_%s_0.656_1.2_%s_FBFM_THC30(1.0,1.0).dat'%(directory, tparams[i], uparams[i])
        data = np.loadtxt(name).reshape((30, 30, 3))
        X, Y, Z = data[:, :, 0], data[:, :, 1], data[:, :, 2]
        if i==0:
            vmin, vmax = -0.04, 0.12
            ticks = [-0.04, 0.0, 0.04, 0.08, 0.12]
        if i==1:
            vmin, vmax = -0.12, 0.04
            ticks = [-0.12, -0.08, -0.04, 0.0, 0.04]
        if i==2:
            vmin, vmax = -0.06, 0.06
            ticks = [-0.06, -0.03, 0.0, 0.03, 0.06]
        divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        pcolor = ax.pcolormesh(Y, X, Z, alpha=1.0, norm=divnorm, cmap='bwr')
        cbar = fig.colorbar(pcolor, ax=ax, ticks=ticks, orientation="horizontal", fraction=0.06, anchor=(1.0, 0.0))
        for tick in cbar.ax.get_xticklabels(): tick.set_fontsize(6)
        
        G = np.array([0.0, 0.0])
        K1, K2 =np.array([1.0/3.0, 2.0/3.0]), np.array([2.0/3.0, 1.0/3.0])
        M1, M2, M3 = np.array([0.0, 0.5]), np.array([0.5, 0.5]), np.array([0.5, 0.0])
        reciprocals = hp.Hexagon('H2')('1P-1P').reciprocals
        for (p, symbol) in zip((G, K1, K2, M1, M2, M3), ("$\Gamma$", "$K_1$", "$K_2$", "$M_1$", "$M_2$", "$M_3$")):
            x, y = p[0]*reciprocals[0] + p[1]*reciprocals[1]
            ax.scatter([y], [x], s=10, marker=".", edgecolors='face', color='black', alpha=0.1, zorder=4)
            ax.text(y, x, symbol,ha='left', va='bottom', fontsize=8, color='black')

        m1 = M1[0]*reciprocals[0] + M1[1]*reciprocals[1]
        k1 = K1[0]*reciprocals[0] + K1[1]*reciprocals[1]
        k2 = K2[0]*reciprocals[0] + K2[1]*reciprocals[1]
        m3 = M3[0]*reciprocals[0] + M3[1]*reciprocals[1]
        ax.plot([m1[1], k1[1]], [m1[0], k1[0]], ls='--', color='black', alpha=0.2, zorder=4)
        ax.plot([k1[1], k2[1]], [k1[0], k2[0]], ls='--', color='black', alpha=0.2, zorder=4)
        ax.plot([k2[1], m3[1]], [k2[0], m3[0]], ls='--', color='black', alpha=0.2, zorder=4)
        ax.text(6.7, 5.7, "($%s_3$)"%tags[i], va='top', ha='right')

    pdb.set_trace()
    plt.savefig('%s/ThermalHallWithTemperature.pdf'%destination)
    plt.close()


def thermal_hall_with_hopping():
    plt.ion()
    fig, axes=plt.subplots(nrows=1, ncols=3)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.16, hspace=0.2, wspace=0.25)

    # phase diagram
    ax=axes[0]
    ts=np.array([0.2445,0.2600,0.2700,0.2732,0.2770,0.2790,0.2800,0.2850,0.2900,0.2950,0.3000,0.3080,0.3100,0.3155,0.3200,0.3250,0.3300,0.3350,0.3400,0.3450,0.3495])
    us=np.array([0.0000,0.2825,0.4285,0.4705,0.5195,0.5415,0.5515,0.5975,0.6375,0.6695,0.6945,0.7075,0.7065,0.6945,0.6775,0.6505,0.6135,0.5335,0.3545,0.1665,0.0000])
    X=np.linspace(ts.min(),ts.max(),201)
    Y=itp.splev(X,itp.splrep(ts,us,k=3),der=0)
    ax.plot(X,Y,lw=2,color='blue',zorder=1)

    ts=np.array([0.2732,0.2800,0.2850,0.2900,0.2950,0.3000,0.3080])
    us=np.array([0.4705,0.3835,0.3215,0.2615,0.2005,0.1315,0.0000])
    X=np.linspace(ts.min(),ts.max(),201)
    Y=itp.splev(X,itp.splrep(ts,us,k=3),der=0)
    ax.plot(X,Y,lw=2,color='blue',zorder=1)

    ts=np.array([0.3080,0.3100,0.3155,0.3200,0.3250,0.3300,0.3350])
    us=np.array([0.0000,0.0235,0.1325,0.2545,0.3585,0.4605,0.5335])
    X=np.linspace(ts.min(),ts.max(),201)
    Y=itp.splev(X,itp.splrep(ts,us,k=3),der=0)
    ax.plot(X,Y,lw=2,color='blue',zorder=1)

    ax.plot([0.29, 0.32], [0.0, 0.0], ls='solid', lw=3, color='green', zorder=0, alpha=0.6, clip_on=False)
    ax.plot([0.29, 0.32], [0.1, 0.1], ls='solid', lw=3, color='red', zorder=0, alpha=0.6)

    ax.minorticks_on()
    ax.set_xlim(0.24,0.36)
    ax.set_ylim(0.00,0.75)
    ax.set_xticks(np.linspace(0.24,0.36,7))
    ax.set_xticklabels(['0.24','','0.28','','0.32','','0.36'])
    ax.set_yticks(np.linspace(0.0,0.7,8))
    ax.set_yticklabels(['0','','0.2','','0.4','','0.6',''])
    for tick in ax.get_xticklabels(): tick.set_fontsize(9)
    for tick in ax.get_yticklabels(): tick.set_fontsize(9)
    ax.set_xlabel("$t^\prime/t$",fontdict={'fontsize':10})
    ax.set_ylabel("$\Delta U/t$",fontdict={'fontsize':10})

    ax.text(0.26,0.53,"NFM",fontsize=9,color='black',ha='center',va='center')
    ax.text(0.305,0.43,"FM",fontsize=9,color='black',ha='center',va='center')
    ax.text(0.305,0.31,"$(C=0)$",fontsize=9,color='black',ha='center',va='center')
    ax.text(0.275,0.2,"TFM$^+$",fontsize=9,color='black',ha='center',va='center')
    ax.text(0.275,0.1,"$(C=+1)$",fontsize=9,color='black',ha='center',va='center')
    ax.text(0.3305,0.2,"TFM$^-$",fontsize=9,color='black',ha='center',va='center')
    ax.text(0.3305,0.1,"$(C=-1)$",fontsize=9,color='black',ha='center',va='center')

    # line 1
    ax = axes[1]
    name = '%s/HCI_H2(1P-1P)_up_1.0_0.656_1.2_1.2_FBFM_THP30(1.0,1.0)_0.08.dat'%directory
    data = np.loadtxt(name)
    ax.plot(data[:, 0], data[:, 1]/0.08, lw=2, color='green')
    ax.text(0.300, -0.20, "TFM$^+$", fontsize=9, color='black', ha='center', va='center')
    ax.text(0.315, -0.20, "TFM$^-$", fontsize=9, color='black', ha='center', va='center')
    ax.minorticks_on()
    ax.set_xlim(0.29, 0.32)
    ax.set_xticks(np.linspace(0.29, 0.32, 4))
    # ax.set_ylim(-0.7, 0.4)
    # ax.set_yticks(np.linspace(-0.6, 0.4, 6))
    for tick in ax.get_xticklabels(): tick.set_fontsize(9)
    for tick in ax.get_yticklabels(): tick.set_fontsize(9)
    ax.set_xlabel("$t^\prime/t$",fontdict={'fontsize':10})
    ax.set_ylabel("$\kappa_{xy}/T(-\\frac{k^2_Bt}{K\hbar})$",fontdict={'fontsize':10})

    # line 2
    ax = axes[2]
    name = '%s/HCI_H2(1P-1P)_up_1.0_0.656_1.2_1.1_FBFM_THP30(1.0,1.0)_0.08.dat'%directory
    data = np.loadtxt(name)
    ax.plot(data[:, 0], data[:, 1]/0.08, lw=2, color='red')
    ax.text(0.296, +0.20, "TFM$^+$", fontsize=9, color='black', ha='center', va='center')
    ax.text(0.308, +0.00, "FM", fontsize=9, color='black', ha='center', va='center')
    ax.text(0.317, -0.45, "TFM$^-$", fontsize=9, color='black', ha='center', va='center')
    ax.minorticks_on()
    ax.set_xlim(0.29, 0.32)
    ax.set_xticks(np.linspace(0.29, 0.32, 4))
    # ax.set_ylim(-0.8, 0.6)
    # ax.set_yticks(np.linspace(-0.8, 0.6, 8))
    for tick in ax.get_xticklabels(): tick.set_fontsize(9)
    for tick in ax.get_yticklabels(): tick.set_fontsize(9)
    ax.set_xlabel("$t^\prime/t$",fontdict={'fontsize':10})
    ax.set_ylabel("$\kappa_{xy}/T(-\\frac{k^2_Bt}{K\hbar})$",fontdict={'fontsize':10})

    pdb.set_trace()
    plt.savefig('%s/ThermalHallWithHopping.pdf'%destination)
    plt.close()


if __name__=='__main__':
    for arg in sys.argv:
        if arg in ('1', 'all'): thermal_hall_with_temperature()
        if arg in ('2', 'all'): thermal_hall_with_hopping()
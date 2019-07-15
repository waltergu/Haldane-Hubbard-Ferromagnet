import HamiltonianPy as HP
import numpy as np
import scipy.interpolate as itp
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import sys
import pdb

def lattice():
    from HamiltonianPy import PID,translation,rotation,Lattice
    from HamiltonianPy import Hexagon
    import itertools as it

    plt.ion()
    fig,axes=plt.subplots(nrows=1,ncols=2)
    fig.subplots_adjust(left=0.05,right=0.98,top=1.0,bottom=0.0,hspace=0.2,wspace=0.1)

    ax=axes[0]
    ax.axis('off')
    ax.axis('equal')
    ax.set_ylim(-0.5,1.2)

    # Lattice
    H6=Hexagon('H6')('1O-1O')
    for bond in H6.bonds:
        if bond.neighbour==0:
            x,y=bond.spoint.rcoord[0],bond.spoint.rcoord[1]
            ax.scatter(x,y,s=np.pi*5**2,color='brown' if bond.spoint.pid.site%2==0 else 'blue',zorder=3)
        else:
            x1,y1=bond.spoint.rcoord[0],bond.spoint.rcoord[1]
            x2,y2=bond.epoint.rcoord[0],bond.epoint.rcoord[1]
            ax.plot([x1,x2],[y1,y2],color='black',linewidth=2,zorder=1)

    # A, B site
    coord=H6.rcoord(PID('H6',0))
    ax.text(coord[0]-0.05,coord[1],'A',ha='right',va='center',fontsize=20)
    coord=H6.rcoord(PID('H6',1))
    ax.text(coord[0]-0.05,coord[1],'B',ha='right',va='center',fontsize=20)

    # t term
    ax.annotate(s='',xy=H6.rcoord(PID('H6',0)),xytext=H6.rcoord(PID('H6',1)),arrowprops={'arrowstyle':'<->','color':'black','linewidth':2.0,'zorder':3})
    coord=(H6.rcoord(PID('H6',0))+H6.rcoord(PID('H6',1)))/2
    ax.text(coord[0]-0.05,coord[1],"$t$",fontweight='bold',color='black',ha='right',va='center',fontsize=22)

    # t' term
    b1=np.array([1.0,0.0])
    b2=np.array([0.5,np.sqrt(3)/2])
    def farrow(ax,coord,inc):
        x,y=coord,coord+inc
        ax.plot([x[0],y[0]],[x[1],y[1]],color='green',linewidth=3.0)
        center,disp=(x+y)/2,inc/40
        ax.annotate(s='',xy=center-disp,xytext=center,arrowprops={'color':'green','linewidth':2,'arrowstyle':'->','zorder':3})
    farrow(ax,H6.rcoord(PID('H6',0)),b2)
    farrow(ax,H6.rcoord(PID('H6',2)),(b1-b2))
    farrow(ax,H6.rcoord(PID('H6',4)),-b1)
    farrow(ax,H6.rcoord(PID('H6',5)),-b2)
    farrow(ax,H6.rcoord(PID('H6',3)),-(b1-b2))
    farrow(ax,H6.rcoord(PID('H6',1)),b1)
    ax.text(0.5,np.sqrt(3)/6,"$t'e^{i\phi}$",fontweight='bold',color='black',ha='center',va='center',fontsize=22)

    # Hubbard term
    coord,disp,inc,delta=H6.rcoord(PID('H6',3)),np.array([0.02,0.0]),np.array([0.0,0.12]),np.array([0.02,0.0])
    ax.annotate(s='',xy=coord-disp-inc-delta-0.01,xytext=coord-disp+inc+delta-0.01,arrowprops={'arrowstyle':'->','linewidth':2,'color':'purple','zorder':4})
    ax.annotate(s='',xy=coord+disp+inc+delta+0.01,xytext=coord+disp-inc-delta+0.01,arrowprops={'arrowstyle':'->','linewidth':2,'color':'purple','zorder':4})
    ax.text(coord[0]+0.07,coord[1]-0.02,"$U_B$",ha='left',va='center',fontweight='bold',color='black',fontsize=20)

    coord,disp,inc,delta=H6.rcoord(PID('H6',4)),np.array([0.02,0.0]),np.array([0.0,0.12]),np.array([0.02,0.0])
    ax.annotate(s='',xy=coord-disp-inc-delta-0.01,xytext=coord-disp+inc+delta-0.01,arrowprops={'arrowstyle':'->','linewidth':2,'color':'purple','zorder':4})
    ax.annotate(s='',xy=coord+disp+inc+delta+0.01,xytext=coord+disp-inc-delta+0.01,arrowprops={'arrowstyle':'->','linewidth':2,'color':'purple','zorder':4})
    ax.text(coord[0]+0.07,coord[1]-0.02,"$U_A$",ha='left',va='center',fontweight='bold',color='black',fontsize=20)

    # numbering of subplot
    ax.text(0.0,0.85,"(a)",ha='left',va='center',fontsize=16,color='black')

    # The Brillouin Zone
    ax=axes[1]
    ax.axis('equal')
    ax.axis('off')
    ax.set_ylim(-1.0,1.0)
    ax.set_xlim(-0.7,0.7)

    # FBZ
    center=(H6.rcoords[0]+H6.rcoords[5])/2
    bzrcoords=translation(rotation(cluster=H6.rcoords,angle=np.pi/6,center=center),-center)
    BZ=Lattice(name="BZ",rcoords=bzrcoords)
    for bond in BZ.bonds:
        if bond.neighbour==1:
            x1,y1=bond.spoint.rcoord[0],bond.spoint.rcoord[1]
            x2,y2=bond.epoint.rcoord[0],bond.epoint.rcoord[1]
            ax.plot([x1,x2],[y1,y2],linewidth=2,color='black')

    # connection
    p=(BZ.rcoord(PID('BZ',2))+BZ.rcoord(PID('BZ',5)))/2
    ax.plot([0.0,p[0]],[0.0,p[1]],ls='--',lw=2,color='k')
    p=BZ.rcoord(PID('BZ',2))
    ax.plot([0.0,p[0]],[0.0,p[1]],ls='--',lw=2,color='k')

    # axis
    for disp in [np.array([0.60,0.0]),np.array([0.0,0.75])]:
        ax.arrow(-disp[0],-disp[1],2*disp[0],2*disp[1],head_width=0.03,head_length=0.05,fc='k',ec='k')
    ax.text(0.62,-0.03,"$\mathrm{k_x}$",va='top',ha='center',fontsize=14,color='black')
    ax.text(-0.02,0.75,"$\mathrm{k_y}$",va='center',ha='right',fontsize=14,color='black')

    # high-symmetric points
    ax.text(-0.05,0.01,"$\Gamma$",ha='center',va='bottom',fontsize=16,color='black')
    ax.text(0.6,0.05,"$K$",ha='center',fontsize=16,color='black')
    ax.text(0.3,0.5,"$K'$",fontsize=16,color='black')
    ax.text(0.495,0.295,"$M$",ha='center',va='center',fontsize=16,color='black')

    # numbering of subplot
    ax.text(-0.65,0.77,"(b)",ha='left',va='center',fontsize=16,color='black')

    pdb.set_trace()
    plt.savefig('lattice.pdf')
    plt.close()

def constH0spectra():
    plt.ion()
    fig,axes=plt.subplots(nrows=2,ncols=2)
    fig.subplots_adjust(left=0.11,right=0.98,top=0.98,bottom=0.125,hspace=0.1,wspace=0.1)

    for i,(tag,parameter) in enumerate(zip(['a','b','c','d'],['1.2','1.1','0.8','0.501'])):
        ax=axes[i//2][i%2]
        name='../result/fbfm/HCI_H2(1P-1P)_up_1.0_0.314_0.656_1.2_%s_FBFM_EB60.npz'%parameter
        data=np.load(name)['data']
        ax.plot(data[:,0],data[:,1:3],color='green',lw=2.5,zorder=4)
        # ax.plot(data[:,0],data[:,3:],color=(0.8,0.8,0.8),lw=3.5,alpha=0.02,zorder=2)
        ax.fill_between(data[:,0],np.max(data[:,3:],axis=1),np.min(data[:,3:],axis=1),color=(0.8,0.8,0.8),alpha=0.9,zorder=2)

        name='../result/fbfm/HCI_H2(1P-1P)_up_1.0_0.314_0.656_1.2_%s_FBFM_EB60(0.0,1.0).npz'%parameter
        data=np.load(name)['data']
        ax.plot(data[:,0],data[:,1:3],ls='-',color='blue',alpha=0.5,lw=1.5,zorder=3)
        # ax.plot(data[:,0],data[:,3:],color='blue',lw=2.0 if i==0 else 1.0,alpha=0.01,zorder=1)
        ax.fill_between(data[:,0],np.max(data[:,3:],axis=1),np.min(data[:,3:],axis=1),color='blue',alpha=0.2,zorder=2)

        ax.axvline(x=20,ls='--',color='black',lw=1,zorder=0)
        ax.axvline(x=30,ls='--',color='black',lw=1,zorder=0)
        ax.axvline(x=40,ls='--',color='black',lw=1,zorder=0)

        ax.minorticks_on()
        ax.set_xlim(0.0,60.0)
        ax.set_xticks(np.linspace(0,60,7))
        ax.set_xticklabels(['$\Gamma$','','$K$','$M$','$K^\prime$','','$\Gamma$'] if i in (2,3) else ['']*7)
        for tick in ax.get_xticklabels(): tick.set_fontsize(16)
        if i in (2,3): ax.set_xlabel('q',fontdict={'fontsize':14})

        ax.set_ylim(0.0,0.7)
        ax.set_yticks(np.linspace(0.0,0.7,8))
        ax.set_yticklabels(['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7'] if i in (0,2) else ['']*8)
        for tick in ax.get_yticklabels(): tick.set_fontsize(13)
        if i in (0,2): ax.set_ylabel('$E/t$',fontdict={'fontsize':16})
        ax.text(3.0,0.68,'(%s)'%tag,ha='left',va='top',fontsize=16,color='black')

    pdb.set_trace()
    plt.savefig('constH0spectra.pdf')
    plt.close()

def constHUspectra():
    plt.ion()
    fig,axes=plt.subplots(nrows=1,ncols=3)
    fig.subplots_adjust(left=0.115,right=0.98,top=0.975,bottom=0.18,hspace=0.1,wspace=0.1)

    for i,(tag,parameter) in enumerate(zip(['a','b','c'],['0.29','0.308','0.32'])):
        ax=axes[i]
        name='../result/fbfm/HCI_H2(1P-1P)_up_1.0_%s_0.656_1.2_1.2_FBFM_EB60.npz'%parameter
        data=np.load(name)['data']
        ax.plot(data[:,0],data[:,1:3],color='green',lw=2,zorder=4)
        # ax.plot(data[:,0],data[:,3:],color=(0.8,0.8,0.8),lw=3.5,alpha=0.9,zorder=2)
        ax.fill_between(data[:,0],np.max(data[:,3:],axis=1),np.min(data[:,3:],axis=1),color=(0.8,0.8,0.8),alpha=0.9,zorder=2)

        ax.axvline(x=20,ls='--',color='black',lw=1,zorder=0)
        ax.axvline(x=30,ls='--',color='black',lw=1,zorder=0)
        ax.axvline(x=40,ls='--',color='black',lw=1,zorder=0)

        ax.minorticks_on()
        ax.set_xlim(0.0,60.0)
        ax.set_xticks(np.linspace(0,60,7))
        ax.set_xticklabels(['$\Gamma$','','$K$','$M$','$K^\prime$','','$\Gamma$'])
        for tick in ax.get_xticklabels(): tick.set_fontsize(18)
        ax.set_xlabel('q',fontdict={'fontsize':16})

        ax.set_ylim(0.0,0.6)
        ax.set_yticks(np.linspace(0.0,0.6,4))
        ax.set_yticklabels(['0','0.2','0.4','0.6'] if i in (0,) else ['']*4)
        for tick in ax.get_yticklabels(): tick.set_fontsize(15)
        if i in (0,): ax.set_ylabel('$E/t$',fontdict={'fontsize':18})
        ax.text(3.0,0.58,'(%s)'%tag,ha='left',va='top',fontsize=18,color='black')

    pdb.set_trace()
    plt.savefig('constHUspectra.pdf')
    plt.close()

def phasediagram():
    plt.ion()
    fig,axes=plt.subplots(nrows=1,ncols=1)
    fig.subplots_adjust(left=0.14,right=0.96,top=0.97,bottom=0.145,hspace=0.2,wspace=0.40)

    ax=axes
    ts=np.array([0.2445,0.2600,0.2700,0.2732,0.2770,0.2790,0.2800,0.2850,0.2900,0.2950,0.3000,0.3080,0.3100,0.3155,0.3200,0.3250,0.3300,0.3350,0.3400,0.3450,0.3495])
    us=np.array([0.0000,0.2825,0.4285,0.4705,0.5195,0.5415,0.5515,0.5975,0.6375,0.6695,0.6945,0.7075,0.7065,0.6945,0.6775,0.6505,0.6135,0.5335,0.3545,0.1665,0.0000])
    ax.plot(ts,us,lw=2,color='blue',zorder=1)

    ts=np.array([0.2732,0.2800,0.2850,0.2900,0.2950,0.3000,0.3080])
    us=np.array([0.4705,0.3835,0.3215,0.2615,0.2005,0.1315,0.0000])
    ax.plot(ts,us,lw=2,color='blue',zorder=1)

    ts=np.array([0.3080,0.3100,0.3155,0.3200,0.3250,0.3300,0.3350])
    us=np.array([0.0000,0.0235,0.1325,0.2545,0.3585,0.4605,0.5335])
    ax.plot(ts,us,lw=2,color='blue',zorder=1)

    xs=[0.29,0.308,0.32]
    ys=[0.000,0.000,0.000]
    ax.scatter(xs,ys,s=60,color='green',marker='o',zorder=2,clip_on=False)

    xs=[0.314,0.314,0.314,0.314]
    ys=[0.000,0.100,0.400,0.699]
    ax.scatter(xs,ys,s=100,color='red',marker='*',zorder=2,clip_on=False)

    ax.minorticks_on()
    ax.set_xlim(0.24,0.36)
    ax.set_ylim(0.00,0.75)
    ax.set_xticks(np.linspace(0.24,0.36,7))
    ax.set_yticks(np.linspace(0.0,0.75,4))
    ax.set_yticklabels(['0','0.25','0.5','0.75'])
    for tick in ax.get_xticklabels():
        tick.set_fontsize(18)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(18)
    ax.set_xlabel("$t^\prime/t$",fontdict={'fontsize':22})
    ax.set_ylabel("$\Delta U/t$",fontdict={'fontsize':22})
    ax.text(0.26,0.53,"NFM",fontsize=22,color='black',ha='center',va='center')
    ax.text(0.305,0.43,"FM",fontsize=22,color='black',ha='center',va='center')
    ax.text(0.305,0.33,"$(C=0)$",fontsize=22,color='black',ha='center',va='center')
    ax.text(0.27,0.22,"TFM",fontsize=22,color='black',ha='center',va='center')
    ax.text(0.27,0.12,"$(C=1)$",fontsize=22,color='black',ha='center',va='center')
    ax.text(0.33,0.22,"TFM",fontsize=22,color='black',ha='center',va='center')
    ax.text(0.33,0.12,"$(C=-1)$",fontsize=22,color='black',ha='center',va='center')

    pdb.set_trace()
    plt.savefig('phasediagram.pdf')
    plt.close()

def domainwallspectra():
    from mpl_toolkits.axes_grid.inset_locator import inset_axes

    plt.ion()
    fig,axes=plt.subplots(nrows=3,ncols=1)
    fig.subplots_adjust(left=0.12,right=0.98,top=0.99,bottom=0.09,hspace=0.1,wspace=0.1)

    names=[ '../result/fbfm/HCI_H2(1P-18P)_up_1.0_0.28_0.656_2.0_2.0_2.7_1.7_FBFM_Domain80.npz',
            '../result/fbfm/HCI_H2(1P-18P)_up_1.0_0.33_0.656_2.0_2.0_2.7_1.7_FBFM_Domain80.npz',
            '../result/fbfm/HCI_H2(1P-18P)_up_1.0_0.656_2.0_2.0_0.28_0.33_0.305_FBFM_Domain80.npz'
            ]
    lnames=['../result/fbfm/HCI_H2(1P-1P)_up_1.0_0.28_0.656_2.0_2.0_FBFM_EB60.npz',
            '../result/fbfm/HCI_H2(1P-1P)_up_1.0_0.33_0.656_2.0_2.0_FBFM_EB60.npz',
            '../result/fbfm/HCI_H2(1P-1P)_up_1.0_0.28_0.656_2.0_2.0_FBFM_EB60.npz'
            ]
    rnames=['../result/fbfm/HCI_H2(1P-1P)_up_1.0_0.28_0.656_2.7_1.7_FBFM_EB60.npz',
            '../result/fbfm/HCI_H2(1P-1P)_up_1.0_0.33_0.656_2.7_1.7_FBFM_EB60.npz',
            '../result/fbfm/HCI_H2(1P-1P)_up_1.0_0.33_0.656_2.0_2.0_FBFM_EB60.npz'
            ]
    tags=['a','b','c']
    LCS=['+1','-1','+1']
    RCS=['0','0','-1']

    for i,(tag,name,lname,rname,LC,RC) in enumerate(zip(tags,names,lnames,rnames,LCS,RCS)):
        # main figure
        ax=axes[i]
        data=np.load(name)['data']
        ax.plot(data[:,0],data[:,1:36],lw=2.5,color=(0.7,0.7,0.7),zorder=0)
        ax.plot(data[:,0],data[:,36:38],lw=2.5,color='green',zorder=1)
        ax.plot(data[:,0],data[:,38:],lw=2.5,color=(0.7,0.7,0.7),zorder=0)

        ax.axvline(x=40,ls='--',color='black',lw=1,zorder=0)

        ax.minorticks_on()
        ax.set_xlim(0.0,80.0)
        ax.set_xticks(np.linspace(0,80,5))
        ax.set_xticklabels(['$-\pi$','$-\pi/2$','$0$','$\pi/2$','$\pi$'] if i in (2,) else ['']*5)
        for tick in ax.get_xticklabels(): tick.set_fontsize(16)
        if i in (2,): ax.set_xlabel('q',fontdict={'fontsize':16})

        ax.set_ylim(0.0,0.7)
        ax.set_yticks(np.linspace(0.0,0.7,8))
        ax.set_yticklabels(['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7'])
        for tick in ax.get_yticklabels(): tick.set_fontsize(13)
        ax.set_ylabel('$E/t$',fontdict={'fontsize':16})
        ax.text(2.0,0.68,'(%s)'%tag,ha='left',va='top',fontsize=16,color='black')

        # left inset figure
        ax=inset_axes(axes[i],width="30%",height="40%",loc='lower left')
        data=np.load(lname)['data']
        ax.plot(data[:,0],data[:,1:3],color='green',lw=1.5,zorder=4)
        ax.fill_between(data[:,0],np.max(data[:,3:],axis=1),np.min(data[:,3:],axis=1),color=(0.8,0.8,0.8),alpha=0.9,zorder=2)
        ax.axvline(x=20,ls='--',color='grey',lw=1,zorder=0)
        ax.axvline(x=30,ls='--',color='grey',lw=1,zorder=0)
        ax.axvline(x=40,ls='--',color='grey',lw=1,zorder=0)
        ax.text(30,0.05,"$C=%s$"%LC,ha='center',va='bottom',fontsize=16,color='black')

        ax.minorticks_on()
        ax.xaxis.tick_top()
        ax.set_xlim(0.0,60.0)
        ax.set_xticks(np.linspace(0,60,7))
        ax.set_xticklabels(['$\Gamma$','','$K$','$M$','$K^\prime$','','$\Gamma$'])
        for tick in ax.get_xticklabels():
            tick.set_fontsize(13)
            tick.set_color('blue')

        ax.yaxis.tick_right()
        ax.set_ylim(0.0,1.0)
        ax.set_yticks(np.linspace(0.0,1.0,3))
        ax.set_yticklabels(['0','0.5','1.0'])
        for tick in ax.get_yticklabels():
            tick.set_fontsize(13)
            tick.set_color('blue')

        # right inset figure
        ax=inset_axes(axes[i],width="30%",height="40%",loc='lower right')
        data=np.load(rname)['data']
        ax.plot(data[:,0],data[:,1:3],color='green',lw=1.5,zorder=4)
        ax.fill_between(data[:,0],np.max(data[:,3:],axis=1),np.min(data[:,3:],axis=1),color=(0.8,0.8,0.8),alpha=0.9,zorder=2)
        ax.axvline(x=20,ls='--',color='grey',lw=1,zorder=0)
        ax.axvline(x=30,ls='--',color='grey',lw=1,zorder=0)
        ax.axvline(x=40,ls='--',color='grey',lw=1,zorder=0)
        ax.text(30,0.05,"$C=%s$"%RC,ha='center',va='bottom',fontsize=16,color='black')

        ax.minorticks_on()
        ax.xaxis.tick_top()
        ax.set_xlim(0.0,60.0)
        ax.set_xticks(np.linspace(0,60,7))
        ax.set_xticklabels(['$\Gamma$','','$K$','$M$','$K^\prime$','','$\Gamma$'])
        for tick in ax.get_xticklabels():
            tick.set_fontsize(13)
            tick.set_color('blue')

        ax.set_ylim(0.0,1.0)
        ax.set_yticks(np.linspace(0.0,1.0,3))
        ax.set_yticklabels(['0','0.5','1.0'])
        for tick in ax.get_yticklabels():
            tick.set_fontsize(13)
            tick.set_color('blue')

    pdb.set_trace()
    plt.savefig('domainwallspectra.pdf')
    plt.close()

if __name__=='__main__':
    for arg in sys.argv:
        if arg in ('1','all'): lattice()
        if arg in ('2','all'): constH0spectra()
        if arg in ('3','all'): constHUspectra()
        if arg in ('4','all'): phasediagram()
        if arg in ('5','all'): domainwallspectra()

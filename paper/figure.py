import HamiltonianPy as HP
import numpy as np
import scipy.interpolate as itp
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import pdb

def lattice():
    from HamiltonianPy import PID,Point,tiling,translation,rotation,azimuthd,Lattice,SuperLattice
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
    ax.text(0.475,0.275,"$M$",ha='center',va='center',fontsize=16,color='black')

    # numbering of subplot
    ax.text(-0.65,0.77,"(b)",ha='left',va='center',fontsize=16,color='black')

    pdb.set_trace()
    plt.savefig('lattice.pdf')
    plt.close()

def spectrum():
    plt.ion()
    fig,axes=plt.subplots(nrows=1,ncols=3)
    fig.subplots_adjust(left=0.10,right=0.98,top=0.96,bottom=0.14,hspace=0.2,wspace=0.25)

    start,inc,vmin,vmax=11,-3,0,12
    cmap=cmx.ScalarMappable(norm=colors.Normalize(vmin=vmin,vmax=vmax),cmap=plt.get_cmap('viridis'))
    for i,parameter in enumerate([0.02,0.409,0.78]):
        names=[
            '../result/ed/1DIF_S2x(8P-1O)_FSTR(32,8,3.0)_1.0_1.4_-0.04_%s_1.0_TrFED_TREDEB.dat'%parameter,
            '../result/fbfm/1DIF_S2x(1P-1O)_up_1.0_1.4_-0.04_%s_1.0_FBFM_EB8.dat'%parameter,
            '../result/fbfm/1DIF_S2x(1P-1O)_up_1.0_1.4_-0.04_%s_1.0_FBFM_EB16.dat'%parameter,
            '../result/fbfm/1DIF_S2x(1P-1O)_up_1.0_1.4_-0.04_%s_1.0_FBFM_EB24.dat'%parameter,
            '../result/fbfm/1DIF_S2x(1P-1O)_up_1.0_1.4_-0.04_%s_1.0_FBFM_EB60.dat'%parameter,
            '../result/fbfm/1DIF_S2x(1P-1O)_up_1.0_1.4_-0.04_%s_1.0_FBFM_EB800.dat'%parameter,
        ]
        for j,(name,label) in enumerate(zip(names,('','$N_q=8$','$N_q=16$','$N_q=24$','$N_q=60$','$N_q=800$'))):
            result=np.loadtxt(name)
            if j==0:
                result=result[np.array([4,5,6,7,0,1,2,3,4]),:]
                xs=np.array(range(result.shape[0]))*1.0/(result.shape[0]-1)
                axes[i].plot(xs,(result[:,1:]+16.0)/parameter,'.',color='red',lw=8,zorder=2,clip_on=False)
            else:
                result=result[np.array(range(result.shape[0])+[0]),:]
                xs=np.array(range(result.shape[0]))*1.0/(result.shape[0]-1)
                if i==1:
                    axes[i].plot(xs,result[:,1]/parameter,color=cmap.to_rgba(start+(j-1)*inc),ls='-',lw=1.5,zorder=1)
                    axes[i].plot(xs,result[:,2]/parameter,color=cmap.to_rgba(start+(j-1)*inc),ls='-',label=label,lw=1.5,zorder=1)
                else:
                    axes[i].plot(xs,result[:,1:]/parameter,color=cmap.to_rgba(start+(j-1)*inc),ls='-',label=label,lw=1.5,zorder=1)
            if i==0 and label=='$N_q=800$':
                from mpl_toolkits.axes_grid.inset_locator import inset_axes
                ax=inset_axes(axes[0],width="40%",height=1.0,loc=1)
                ax.plot(xs,result[:,1:]/parameter,ls='-',color=cmap.to_rgba(start+(j-1)*inc),lw=1.5,zorder=1)
                ax.minorticks_on()
                ax.set_xlim(0.25,0.75)
                ax.set_ylim(0.5,0.6)
                ax.set_xticks([0.25,0.5,0.75])
                ax.set_xticklabels(['','$\pi$',''])
                ax.set_yticks([0.52,0.58])
                for tick in ax.get_xticklabels():
                    tick.set_fontsize(14)
                for tick in ax.get_yticklabels():
                    tick.set_fontsize(14)
        axes[i].minorticks_on()
        axes[i].set_xlim(-0.01,1.01)
        axes[i].set_xticks(np.linspace(0.0,1.0,5))
        axes[i].set_xticklabels(['0','','$\pi$','','$2\pi$'])
        axes[i].set_xlabel('q',fontdict={'fontsize':18})
        if i==0:
            axes[i].set_ylim(-0.01,2.0)
            axes[i].set_yticks(np.linspace(0.0,2.0,5))
            axes[i].set_ylabel('$E/U_s$',fontdict={'fontsize':18})
            axes[i].text(0.1,1.8,"(a)",fontsize=18,color='black')
        if i==1:
            axes[i].set_ylim(-0.003,0.6)
            axes[i].set_yticks(np.linspace(0.0,0.6,3))
            axes[i].text(0.1,0.54,"(b)",fontsize=18,color='black')
            leg=axes[i].legend(loc='lower center',fancybox=True,shadow=False,prop={'size': 14})
            leg.get_frame().set_alpha(0.5)
        if i==2:
            axes[i].set_ylim(-0.0025,0.5)
            axes[i].set_yticks(np.linspace(0.0,0.5,6))
            axes[i].text(0.1,0.45,"(c)",fontsize=18,color='black')
        for tick in axes[i].get_xticklabels():
            tick.set_fontsize(18)
        for tick in axes[i].get_yticklabels():
            tick.set_fontsize(18)

    pdb.set_trace()
    plt.savefig('spectrum.pdf')
    plt.close()

def phasediagram():
    plt.ion()
    fig,axes=plt.subplots(nrows=1,ncols=1)
    fig.subplots_adjust(left=0.14,right=0.96,top=0.97,bottom=0.16,hspace=0.2,wspace=0.40)

    ax=axes
    ts=np.array([0.2445,0.2600,0.2700,0.2732,0.2770,0.2790,0.2800,0.2850,0.2900,0.2950,0.3000,0.3080,0.3100,0.3155,0.3200,0.3250,0.3300,0.3350,0.3400,0.3450,0.3495])
    us=np.array([0.0000,0.2825,0.4285,0.4705,0.5195,0.5415,0.5515,0.5975,0.6375,0.6695,0.6945,0.7075,0.7065,0.6945,0.6775,0.6505,0.6135,0.5335,0.3545,0.1665,0.0000])
    ax.plot(ts,us,lw=2,color='blue')

    ts=np.array([0.2732,0.2800,0.2850,0.2900,0.2950,0.3000,0.3080])
    us=np.array([0.4705,0.3835,0.3215,0.2615,0.2005,0.1315,0.0000])
    ax.plot(ts,us,lw=2,color='blue')

    ts=np.array([0.3080,0.3100,0.3155,0.3200,0.3250,0.3300,0.3350])
    us=np.array([0.0000,0.0235,0.1325,0.2545,0.3585,0.4605,0.5335])
    ax.plot(ts,us,lw=2,color='blue')

    xs=[0.300,0.308,0.316]
    ys=[0.000,0.000,0.000]
    ax.scatter(xs,ys,color='red',zorder=2,clip_on=False)

    # xs=[0.295,0.308,0.320]
    # ys=[0.000,0.000,0.000]
    # ax.scatter(xs,ys,color='blue',zorder=2,clip_on=False)

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

def edge():
    plt.ion()
    gs=plt.GridSpec(3,3)
    gs.update(left=0.14,right=0.96,top=0.97,bottom=0.11,hspace=0.55,wspace=0.20)

    ax=plt.subplot(gs[0:2,:])
    result=np.loadtxt('../result/fbfm/1DIF_S2x(60O-1O)_up_1.0_1.4_-0.04_1.0_FBFM_EDGE.dat')
    xs=result[:,0]
    ax.plot(xs,result[:,1:59]/xs[:,np.newaxis],color='grey',lw=1,alpha=0.5,zorder=1)
    ax.plot(xs,result[:,59]/xs,color='blue',lw=2,zorder=2)
    ax.plot(xs,result[:,60]/xs,color='green',lw=2,zorder=2)
    ax.plot(xs,result[:,61]/xs,color='purple',lw=2,zorder=2)
    ax.plot(xs,result[:,62]/xs,color='red',lw=2,zorder=2)
    ax.plot(xs,result[:,63:110]/xs[:,np.newaxis],color='grey',lw=2,alpha=0.5,zorder=1)

    ax.minorticks_on()
    ax.set_ylim(0.19,0.6)
    ax.set_xticks(np.linspace(0.0,0.8,5))
    ax.set_yticks(np.linspace(0.2,0.6,5))
    for tick in ax.get_xticklabels():
        tick.set_fontsize(18)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(18)
    ax.set_xlabel("$U_s/U_d$",fontdict={'fontsize':20})
    ax.set_ylabel("$E/Us$",fontdict={'fontsize':20})
    ax.text(0.7,0.56,"(a)",fontsize=18,color='black')

    for i,(parameter,tag) in enumerate(zip(['0.02','0.3','0.78'],['(b)','(c)','(d)'])):
        ax=plt.subplot(gs[2,i])
        result=np.loadtxt('../result/fbfm/1DIF_S2x(60O-1O)_up_1.0_1.4_-0.04_%s_1.0_FBFM_POS.dat'%parameter)
        ax.plot(result[:,0],result[:,2].real,color='blue',lw=2)
        ax.plot(result[:,0],result[:,3].real,color='green',lw=2)
        ax.plot(result[:,0],result[:,4].real,color='purple',lw=2)
        ax.plot(result[:,0],result[:,5].real,color='red',lw=2)
        ax.minorticks_on()
        ax.set_ylim(0.0,0.45)
        ax.set_xticks(np.linspace(0,120,4))
        ax.set_yticks(np.linspace(0.0,0.4,3))
        for tick in ax.get_xticklabels():
            tick.set_fontsize(16)
        ax.set_xlabel('position',fontdict={'fontsize':20})
        if i==0:
            for tick in ax.get_yticklabels():
                tick.set_fontsize(18)
            ax.set_ylabel('$\Delta\langle S_z\\rangle$',fontdict={'fontsize':18})
        else:
            ax.set_yticklabels(['']*3)
        ax.text(10,0.32,tag,fontsize=18,color='black')

    pdb.set_trace()
    plt.savefig('edge.pdf')
    plt.close()

if __name__=='__main__':
    for arg in sys.argv:
        if arg in ('1','all'): lattice()
        if arg in ('2','all'): spectrum()
        if arg in ('3','all'): phasediagram()
        if arg in ('4','all'): edge()


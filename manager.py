import mkl
import numpy as np
from HamiltonianPy import *
from source import *
from collections import OrderedDict
from fractions import Fraction

def tbatasks(name,parameters,lattice,terms,nk=50,jobs=()):
    import HamiltonianPy.FreeSystem as TBA
    tba=tbaconstruct(name,parameters,lattice,terms)
    if 'EB' in jobs:
        if len(lattice.vectors)==2:
            path=FBZ(lattice.reciprocals,nks=(120,120)).path(KMap(lattice.reciprocals,'H:G-K2,K2-M1,M1-K1,K1-G'),mode='P')
            tba.register(EB(name='EB',path=path,run=TBA.TBAEB))
            #tba.register(EB(name='EB',path=hexagon_gkm(reciprocals=lattice.reciprocals,nk=100),run=TBA.TBAEB))
        elif len(lattice.vectors)==1:
            tba.register(EB(name='EB',path=KSpace(reciprocals=lattice.reciprocals,segments=[(0,1.0)],end=True,nk=401),run=TBA.TBAEB))
        else:
            tba.register(EB(name='EB',run=TBA.TBAEB))
        #data=tba.records['EB']
        #print("bandwidth: %s"%(data[:,1].max()-data[:,1].min()))
        #print("gap: %s"%(data[:,3].min()-data[:,1].max()))
    if 'CN1' in jobs:
        assert len(lattice.vectors)==2
        tba.register(CN(name='CN1',BZ=FBZ(lattice.reciprocals,nks=(nk,nk)),ns=(0,1),run=TBA.TBACN))
    if 'CN2' in jobs:
        assert len(lattice.vectors)==2
        tba.register(BC(name='CN2',BZ=KSpace(lattice.reciprocals,nk=nk),mu=-0.5,run=TBA.TBABC))
    if 'Domain' in jobs:
        assert len(lattice.vectors)==2
        tba.register(EB(name='Domain',path=KSpace(reciprocals=[lattice.reciprocals[0]],nk=100,),run=TBA.TBAEB))
    tba.summary()
    return tba

def fbfmtasks(name,parameters,lattice,terms,interactions,nk=50,scalefree=1.0,scaleint=1.0,jobs=(),**kwargs):
    import HamiltonianPy.FBFM as FB
    from mpi4py import MPI
    nprocs=MPI.COMM_WORLD.Get_size()
    rank=MPI.COMM_WORLD.Get_rank()
    assert len(lattice.vectors)==2
    ns,ne=len(lattice),len(lattice)//2
    if 'EB' in jobs:
        basis=FB.FBFMBasis(BZ=FBZ(lattice.reciprocals,nks=(nk,nk)),filling=Fraction(ne,ns*2))
        fbfm=fbfmconstruct(name,parameters,basis,lattice,terms,interactions)
        #fbfm.register(FB.EB(name='EB%s'%nk,path='H:G-K1',ne=nk**2,scalefree=scalefree,scaleint=scaleint,np=nprocs,run=FB.FBFMEB))
        fbfm.register(FB.EB(name='EB%s'%nk,path='H:G-K1,K1-M1,M1-K2,K2-G',ne=nk**2,scalefree=scalefree,scaleint=scaleint,np=nprocs,run=FB.FBFMEB))
    if 'EEB' in jobs:
        basis=FB.FBFMBasis(BZ=FBZ(lattice.reciprocals,nks=(nk,nk)),filling=Fraction(ne,ns*2))
        fbfm=fbfmconstruct(name,parameters,basis,lattice,terms,interactions)
        fbfm.register(EEB(name='EEB%s'%nk,path='H:G-K1,K1-M1,M1-K2,K2-G',showms=kwargs.get('showms',()),schmidt=kwargs.get('schmidt',False),scalefree=scalefree,scaleint=scaleint,run=FBFMEEB))
    if 'CN' in jobs:
        basis=FB.FBFMBasis(BZ=FBZ(lattice.reciprocals,nks=(nk,nk)),filling=Fraction(ne,ns*2))
        fbfm=fbfmconstruct(name,parameters,basis,lattice,terms,interactions)
        fbfm.register(FB.CN(name='CN%s'%nk,BZ=basis.BZ,ns=(0,1),scalefree=scalefree,scaleint=scaleint,run=FB.FBFMCN))
    if 'Domain' in jobs:
        basis=FB.FBFMBasis(BZ=FBZ([lattice.reciprocals[0]],nks=(nk,)),filling=Fraction(ne,ns*2))
        path=basis.BZ.path(KMap([lattice.reciprocals[0]],'L:X2-X1'),mode='Q')
        fbfm=fbfmconstruct(name,parameters,basis,lattice,terms,interactions)
        if rank==0: print("%s_%s"%(fbfm,nk))
        fbfm.register(FB.EB(name='Domain%s'%nk,path=path,ne=ns*2,scalefree=scalefree,scaleint=scaleint,method='eigsh',np=nprocs,run=FB.FBFMEB))
    if 'THT' in jobs:
        basis=FB.FBFMBasis(BZ=FBZ(lattice.reciprocals,nks=(nk,nk)),filling=Fraction(ne,ns*2))
        fbfm=fbfmconstruct(name,parameters,basis,lattice,terms,interactions)
        fbfm.add(TH(name='TH%s'%nk,BZ=basis.BZ,ns=[0,1],ncontinuum=1,scalefree=scalefree,scaleint=scaleint))
        fbfm.register(THT(name='THT%s'%nk,Ts=np.linspace(0.0,0.5,51)+10**-3,dependences=['TH%s'%nk],run=FBFMTHT))
    if 'THP' in jobs:
        basis=FB.FBFMBasis(BZ=FBZ(lattice.reciprocals,nks=(nk,nk)),filling=Fraction(ne,ns*2))
        fbfm=fbfmconstruct(name,parameters,basis,lattice,terms,interactions)
        fbfm.add(TH(name='TH%s'%nk,BZ=basis.BZ,ns=[0,1],ncontinuum=1,T=0.1,scalefree=scalefree,scaleint=scaleint))
        fbfm.register(THP(name='THP%s'%nk,params=BaseSpace(('t2',np.linspace(kwargs['t2tp'], kwargs['t2bt'], 31))),dependences=['TH%s'%nk],run=FBFMTHP))
    fbfm.summary()
    return fbfm

def fbfmphaseboundary(task,name,parameters,lattice,terms,interactions,ranges,nk=50,scalefree=1.0,scaleint=1.0):
    import HamiltonianPy.FBFM as FB
    import HamiltonianPy.Misc as hm
    assert  len(lattice.vectors)==2
    ns,ne=len(lattice),len(lattice)//2
    basis=FB.FBFMBasis(BZ=FBZ(lattice.reciprocals,nks=(nk,nk)),filling=Fraction(ne,ns*2))
    def updateparam(parameters,key,value):
        if key=='t2': parameters['t2']=value
        if key=='dU': parameters['Ub']=parameters['Ua']-value
    if task=='nfm-fm: t2' or task=='nfm-fm: dU':
        def phaseboundary(param):
            updateparam(parameters,task[-2:],param)
            fbfm=fbfmconstruct(name,parameters,basis,lattice,terms,interactions)
            print(fbfm)
            fbfm.register(FB.EB(name='EB%s'%nk,path='H:G-K1,K1-M1,M1-K2,K2-G',ne=nk**2,scalefree=scalefree,scaleint=scaleint,plot=False,savedata=False,run=FB.FBFMEB))
            data=fbfm.records['EB%s'%nk]
            return -1 if np.any(data[:,1]<-10**-8) else +1
    elif task=='tfm-fm: t2' or task=='tfm-fm: dU':
        def phaseboundary(param):
            updateparam(parameters,task[-2:],param)
            fbfm=fbfmconstruct(name,parameters,basis,lattice,terms,interactions)
            fbfm.register(FB.CN(name='CN%s'%nk,BZ=basis.BZ,ns=(0,1),scalefree=scalefree,scaleint=scaleint,run=FB.FBFMCN))
            data=fbfm.records['CN%s'%nk]
            return -1 if np.abs(data[0])<10**-4 else +1
    else:
        raise ValueError("fbfmphaseboundary error: not supported task(%s)."%task)
    result=hm.bisect(phaseboundary,ranges)
    print(result)
    return result

if __name__=='__main__':
    mkl.set_num_threads(1)
    Engine.DEBUG=True
    Engine.MKDIR=False

    #nk=20

    # parameters
    parameters=OrderedDict()
    parameters['t1']=1.0
    parameters['t2']=0.3155
    # parameters['t2']=0.29
    parameters['phi']=0.656
    # delta=0.694
    delta=0.0

    # tba
    #tbatasks(name1,parameters,H2('1P-1P',nnb),[t,ci],jobs=['EB'])
    #tbatasks(name1,parameters,H2('1P-1P',nnb),[t,ci],nk=nk,jobs=['CN1'])
    #tbatasks(name1,parameters,H2('1P-1P',nnb),[t,ci],nk=nk,jobs=['CN2'])

    #parameters['dm']=1.0#3*np.sqrt(3)*parameters['t2']
    #tbatasks(name1,parameters,H2('1P-1P',nnb),[t,ci,dm],jobs=['EB'])
    #tbatasks(name1,parameters,H2('50O-1P',nnb),[t,ci,dm],nk=200,jobs=['EB'])

    #N=6
    #lattice=H2('1P-%sP'%N,nnb)
    #pos=2*N/2

    # tba domain wall
    #parameters['dml']=0.2
    #parameters['dmr']=0.0
    #tbatasks(name1,parameters,lattice,[t,ci,dml(pos),dmr(pos)],jobs=['Domain'])

    #parameters['t2l']=0.3
    #parameters['t2r']=-0.3
    #parameters['t2a']=0.0
    #tbatasks(name1,parameters,lattice,[t,t2l(pos),t2r(pos),t2a(pos),dml(pos),dmr(pos)],jobs=['Domain'])

    # fbfm
    parameters['Ua']=1.2
    parameters['Ub']=parameters['Ua']-delta

    #nk=60
    #fbfmtasks(name1,parameters,H2('1P-1P',nnb),[t,ci],[Ua,Ub],nk=nk,scalefree=1.0,scaleint=1.0,jobs=['EB'])
    #fbfmtasks(name1,parameters,H2('1P-1P',nnb),[t,ci],[Ua,Ub],nk=nk,scalefree=0.0,scaleint=1.0,jobs=['EB'])
    #fbfmtasks(name1,parameters,H2('1P-1P',nnb),[t,ci],[Ua,Ub],nk=nk,scalefree=1.0,scaleint=1.0,schmidt=True,showms=(0,nk//3,nk//3*2),jobs=['EEB'])

    #nk=20
    #ts=np.linspace(0.2,0.3,101)
    #fbfmphaseboundary('nfm-fm: t2',name1,parameters,H2('1P-1P',nnb),[t,ci],[Ua,Ub],ts,nk=nk,scalefree=1.0,scaleint=1.0)
    #fbfmphaseboundary('tfm-fm: t2',name1,parameters,H2('1P-1P',nnb),[t,ci],[Ua,Ub],ts,nk=nk,scalefree=1.0,scaleint=1.0)

    #us=np.linspace(0.0,0.8,801)
    #fbfmphaseboundary('nfm-fm: dU',name1,parameters,H2('1P-1P',nnb),[t,ci],[Ua,Ub],us,nk=nk,scalefree=1.0,scaleint=1.0)
    #fbfmphaseboundary('tfm-fm: dU',name1,parameters,H2('1P-1P',nnb),[t,ci],[Ua,Ub],us,nk=nk,scalefree=1.0,scaleint=1.0)

    #fbfmtasks(name1,parameters,H2('1P-1P',nnb),[t,ci],[Ua,Ub],nk=nk,scalefree=1.0,scaleint=1.0,jobs=['CN'])
    #fbfmtasks(name1,parameters,H2('1P-1P',nnb),[t,ci],[Ua,Ub],nk=nk,scalefree=0.0,scaleint=1.0,jobs=['CN'])

    # thermal hall
    nk=24
    #fbfmtasks(name1,parameters,H2('1P-1P',nnb),[t,ci],[Ua,Ub],nk=nk,scalefree=1.0,scaleint=1.0,jobs=['THT'])
    #fbfmtasks(name1,parameters,H2('1P-1P',nnb),[t,ci],[Ua,Ub],nk=nk,scalefree=0.0,scaleint=1.0,jobs=['THT'])

    t2tp, t2bt = 0.29, 0.32
    fbfmtasks(name1,parameters,H2('1P-1P',nnb),[t,ci],[Ua,Ub],nk=nk,scalefree=1.0,scaleint=1.0,jobs=['THP'],t2tp=t2tp,t2bt=t2bt)
    #fbfmtasks(name1,parameters,H2('1P-1P',nnb),[t,ci],[Ua,Ub],nk=nk,scalefree=0.0,scaleint=1.0,jobs=['THT'],t2tp=t2tp,t2bt=t2bt)

    # fbfm domain wall
    #nk=60
    #parameters['Ual']=1.9
    #parameters['Ubl']=1.9
    #parameters['Uar']=2.7
    #parameters['Ubr']=1.7
    #fbfmtasks(name1,parameters,lattice,[t,ci],[Ual(pos),Uar(pos),Ubl(pos),Ubr(pos)],nk=nk,scalefree=1.0,scaleint=1.0,jobs=['Domain'])

    #nk=30
    #parameters['t2l']=1.2
    #parameters['t2r']=1.2
    #parameters['t2a']=1.2*2
    #fbfmtasks(name1,parameters,lattice,[t,ci],[Ual(pos),Uar(pos),Ubl(pos),Ubr(pos)],nk=nk,scalefree=1.0,scaleint=1.0,jobs=['Domain'])

    # ti
    #nk=30
    #fbfmtasks(name2,parameters,H2('1P-1P',nnb),[t,ti],[Ua,Ub],nk=nk,scalefree=1.0,scaleint=1.0,jobs=['EB'])
    #fbfmtasks(name2,parameters,H2('1P-1P',nnb),[t,ti],[Ua,Ub],nk=nk,scalefree=0.0,scaleint=1.0,jobs=['EB'])
    #fbfmtasks(name2,parameters,H2('1P-1P',nnb),[t,ti],[Ua,Ub],nk=nk,scalefree=0.0,scaleint=0.0,schmidt=True,showms=(0,nk//3,nk//3*2),jobs=['EEB'])
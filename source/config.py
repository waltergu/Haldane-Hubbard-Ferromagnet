from HamiltonianPy import *
import numpy as np

__all__=['name1','name2','nnb','parametermap','idfmap','haldane','km','t','ci','ti','dm','Ua','Ub','t2l','t2r','t2a','dml','dmr','Ual','Uar','Ubl','Ubr','H2','H4','effectiveH','EEB','FBFMEEB','FBFMECN','TH','THT','FBFMTHT','THP','FBFMTHP']

# The configs of the model
name1="HCI"
name2="HTI"
nnb=2

# parametermap
parametermap=None

# idfmap
idfmap=lambda pid: Fock(atom=pid.site%2,nspin=2,norbital=1,nnambu=1)

# haldane hopping
def haldane(bond,phi):
    assert bond.spoint.pid.site%2==bond.epoint.pid.site%2
    phase,theta,site=np.exp(1.0j*phi),azimuthd(bond.rcoord),bond.epoint.pid.site%2
    if np.allclose(theta,60) or np.allclose(theta,180) or np.allclose(theta,300):
        result=phase
    else:
        result=phase.conjugate()
    if site==1: result=result.conjugate()
    return FockPack(value=result,spins=(0,0))+FockPack(value=result,spins=(1,1))

# km hopping
def km(bond,phi):
    assert bond.spoint.pid.site%2==bond.epoint.pid.site%2
    phase,theta,site=np.exp(1.0j*phi),azimuthd(bond.rcoord),bond.epoint.pid.site%2
    if np.allclose(theta,60) or np.allclose(theta,180) or np.allclose(theta,300):
        result=phase
    else:
        result=phase.conjugate()
    if site==1: result=result.conjugate()
    return FockPack(value=result,spins=(0,0))+FockPack(value=result.conjugate(),spins=(1,1))

# terms
t=lambda **parameters: Hopping('t1',parameters['t1'],neighbour=1,statistics='f')
ci=lambda **parameters: Hopping('t2',parameters['t2'],neighbour=2,indexpacks=lambda bond: haldane(bond,parameters['phi']),statistics='f',modulate=True)
ti=lambda **parameters: Hopping('t2',parameters['t2'],neighbour=2,indexpacks=lambda bond: km(bond,parameters['phi']),statistics='f',modulate=True)
dm=lambda **parameters: Onsite('dm',parameters['dm'],indexpacks=sigmaz('sl'),statistics='f',modulate=True)
Ua=lambda **parameters: Hubbard('Ua',parameters['Ua'],atom=0,statistics='f',modulate=True)
Ub=lambda **parameters: Hubbard('Ub',parameters['Ub'],atom=1,statistics='f',modulate=True)

# for domain walls
def left(pos):
    def amplitude(bond):
        s1,s2=bond.spoint.pid.site,bond.epoint.pid.site
        return 1 if s1<pos and s2<pos else 0
    return amplitude

def right(pos):
    def amplitude(bond):
        s1,s2=bond.spoint.pid.site,bond.epoint.pid.site
        return 1 if s1>=pos and s2>=pos else 0
    return amplitude

def accoss(pos):
    def amplitude(bond):
        s1,s2=bond.spoint.pid.site,bond.epoint.pid.site
        return 1 if (s1>=pos and s2<pos) or (s1<pos and s2>=pos) else 0
    return amplitude

t2l=lambda pos: lambda **params: Hopping('t2l',params['t2l'],neighbour=2,amplitude=left(pos),indexpacks=lambda bond: haldane(bond,params['phi']),statistics='f')
t2r=lambda pos: lambda **params: Hopping('t2r',params['t2r'],neighbour=2,amplitude=right(pos),indexpacks=lambda bond: haldane(bond,params['phi']),statistics='f')
t2a=lambda pos: lambda **params: Hopping('t2r',params['t2r'],neighbour=2,amplitude=accoss(pos),indexpacks=lambda bond: haldane(bond,params['phi']),statistics='f')
dml=lambda pos: lambda **params: Onsite('dml',params['dml'],amplitude=left(pos),indexpacks=sigmaz('sl'),statistics='f')
dmr=lambda pos: lambda **params: Onsite('dmr',params['dmr'],amplitude=right(pos),indexpacks=sigmaz('sl'),statistics='f')
Ual=lambda pos: lambda **params: Hubbard('Ual',params['Ual'],atom=0,amplitude=left(pos),statistics='f')
Uar=lambda pos: lambda **params: Hubbard('Uar',params['Uar'],atom=0,amplitude=right(pos),statistics='f')
Ubl=lambda pos: lambda **params: Hubbard('Ubl',params['Ubl'],atom=1,amplitude=left(pos),statistics='f')
Ubr=lambda pos: lambda **params: Hubbard('Ubr',params['Ubr'],atom=1,amplitude=right(pos),statistics='f')

# cluster
H2=Hexagon('H2')
H4=Hexagon('H4')

import itertools as it
import scipy.linalg as sl
def effectiveH(fbfm,k,scalefree=1.0,scaleint=1.0,schmidt=False):
    k=fbfm.basis.BZ.type(k)
    assert fbfm.basis.nsp==1
    permutation=np.argsort((fbfm.basis.BZ-k).sorted(history=True)[1])
    eups,uups=fbfm.basis.E2[:,0],fbfm.basis.U2[:,:,0]
    edws,udws=fbfm.basis.E1[:,0],fbfm.basis.U1[:,:,0]
    Us=np.array([opt.value for opt in fbfm.igenerator.operators.values()])
    assert len(Us)==uups.shape[0]==udws.shape[0]
    vs=[]
    es=edws[permutation]-eups
    sums=np.zeros(fbfm.basis.nk,dtype=fbfm.dtype)
    for i,(uup,udw) in enumerate(zip(uups,udws)):
        udw=udw[permutation]
        vs.append(np.sqrt(Us[i]/fbfm.basis.nk)*uup.conjugate()*udw)
        for j in range(len(udw)):
            sums[j]+=np.vdot(uup,uup)*udw[j]*udw[j].conjugate()*Us[i]/fbfm.basis.nk
    V=np.zeros((len(vs),len(vs)),dtype=fbfm.dtype)
    E=np.zeros((len(vs),len(vs)),dtype=fbfm.dtype)
    U=np.zeros((len(vs),len(vs)),dtype=fbfm.dtype)
    # if schmidt:
    #     S=np.zeros((len(vs),len(vs)),dtype=fbfm.dtype)
    #     S[0,0]=1/sl.norm(vs[0])
    #     S[1,0]=0
    #     inner=np.vdot(vs[0],vs[1])/np.vdot(vs[0],vs[0])
    #     norm=sl.norm(vs[1]-inner*vs[0])
    #     S[0,1]=-inner/norm
    #     S[1,1]=1/norm
    #     vs=np.array(vs)
    #     vs=S.T.dot(vs)
    #     for (i,j) in it.product(range(len(vs)),range(len(vs))):
    #         V[i,j]=np.vdot(vs[i],vs[j])
    #         E[i,j]=np.vdot(vs[i],es*vs[j])
    #         U[i,j]=np.vdot(vs[i],sums*vs[j])
    #     assert np.allclose(V,np.identity(len(vs)))
    #     E*=scalefree
    #     U*=scaleint
    #     M=E+U-sl.inv(S).dot(sl.inv(S).T.conjugate())
    for (i,j) in it.product(range(len(vs)),range(len(vs))):
        V[i,j]=np.vdot(vs[i],vs[j])
        E[i,j]=np.vdot(vs[i],es*vs[j])
        U[i,j]=np.vdot(vs[i],sums*vs[j])
    E*=scalefree
    U*=scaleint
    M=np.dot(sl.inv(V),E+U)-V
    if schmidt:
        S=np.zeros((len(vs),len(vs)),dtype=fbfm.dtype)
        S[0,0]=1/sl.norm(vs[0])
        S[1,0]=0
        inner=np.vdot(vs[0],vs[1])/np.vdot(vs[0],vs[0])
        norm=sl.norm(vs[1]-inner*vs[0])
        S[0,1]=-inner/norm
        S[1,1]=1/norm
        M=sl.inv(S).dot(M).dot(S)
        assert np.allclose(sl.inv(S).dot(sl.inv(S).T.conjugate()),sl.inv(S).dot(V).dot(S))
    return M

import HamiltonianPy.FBFM as FB

class EEB(FB.EB):
    def __init__(self,scalefree=1.0,scaleint=1.0,showms=(),schmidt=False,**karg):
        super(FB.EB,self).__init__(**karg)
        self.scalefree=scalefree
        self.scaleint=scaleint
        self.showms=showms
        self.schmidt=schmidt

def FBFMEEB(engine,app):
    path=app.path
    bz,reciprocals=engine.basis.BZ,engine.lattice.reciprocals
    if not isinstance(path,BaseSpace): path=bz.path(KMap(reciprocals,path) if isinstance(path,str) else path,mode='Q')
    result=np.zeros((path.rank(0),engine.basis.U1.shape[0]+1))
    result[:,0]=path.mesh(0) if path.mesh(0).ndim==1 else np.array(range(path.rank(0)))
    flag=True
    for i,paras in enumerate(path('+')):
        m=effectiveH(engine,schmidt=app.schmidt,scalefree=app.scalefree,scaleint=app.scaleint,**paras)
        if i in app.showms: engine.log<<'%s: \n%s\n'%(i,m)
        flag=flag and np.allclose(m,m.T.conjugate())
        evs=sl.eig(m)[0]
        assert np.allclose(evs.imag,0.0)
        result[i,1:]=sorted(evs.real)
    engine.log<<'Hermitian: %s\n'%flag
    name='%s_%s%s'%(engine.tostr(mask=path.tags),app.name,app.suffix)
    if app.savedata: np.savetxt('%s/%s.dat'%(engine.dout,name),result)
    if app.plot: app.figure('L',result,'%s/%s'%(engine.dout,name))
    if app.returndata: return result

def FBFMECN(engine,app):
    engine.log<<'%s\n'%engine
    engine.log<<'%s: '%app.BZ.rank('k')
    def matrix(i,j):
        engine.log<<'%s-%s%s'%(i,j,'..' if (i+1,j+1)!=app.BZ.type.periods else '')
        return effectiveH(engine,k=[i,j])
    phases=app.set(matrix)
    engine.log<<'\n'
    engine.log<<'Chern numbers: %s'%(", ".join("%s(%s)"%(phase,n) for n,phase in zip(app.ns,phases)))<<'\n'
    if app.returndata: return phases

from scipy.linalg import eigh
class TH(App):
    def __init__(self, BZ, ns, T=None, scalefree=1.0, scaleint=1.0, ncontinuum=None, **kwargs):
        self.BZ = BZ
        self.ns = ns
        self.T = T
        self.scalefree = scalefree
        self.scaleint = scaleint
        self.ncontinuum = ncontinuum
        self.emesh = {}
        self.vmesh = {}
        self.cmesh = {}
        self.jmesh = {}

    def set(self, H):
        assert len(self.BZ.type.periods) == 2
        N1, N2 = self.BZ.type.periods
        for (i, j) in it.product(range(N1), range(N2)):
            es, vs = eigh(H(i, j))
            if isinstance(self.ncontinuum, int):
                self.jmesh[(i, j)] = iscontinuum(es[0:self.ncontinuum+10], self.ncontinuum)
            else:
                self.jmesh[(i, j)] = False
            self.emesh[(i, j)] = es[self.ns]
            self.vmesh[(i, j)] = vs[:, self.ns]
        phases = np.zeros(len(self.ns), dtype=np.float64)
        for (i, j) in it.product(range(N1), range(N2)):
            i1, j1 = i, j
            i2, j2 = (i+1)%N1, (j+1)%N2
            ic = self.jmesh[(i, j)]
            vs1, vs2, vs3, vs4 = self.vmesh[(i1, j1)], self.vmesh[(i2, j1)], self.vmesh[(i2, j2)], self.vmesh[(i1, j2)]
            self.cmesh[(i, j)] = np.zeros(phases.shape, dtype=np.float64)
            for (k, n) in enumerate(self.ns):
                if n<self.ncontinuum or not ic:
                    p1, p2 = np.vdot(vs1[:, k], vs2[:, k]), np.vdot(vs2[:, k], vs3[:, k])
                    p3, p4 = np.vdot(vs3[:, k], vs4[:, k]), np.vdot(vs4[:, k], vs1[:, k])
                    phase = np.angle(p1*p2*p3*p4)
                    self.cmesh[(i, j)][k] = phase
                    phases[k] += phase
        print("phases: %s"%(phases/np.pi/2))
        return phases/np.pi/2

    def compute(self):
        result = 0.0
        N1, N2 = self.BZ.type.periods
        for (i, j) in it.product(range(N1), range(N2)):
            es, cs = self.emesh[(i, j)], self.cmesh[(i, j)]
            for k in range(len(es)):
                result += coeff_1(bose(self.T, es[k]))*cs[k]
        return result*self.T

class THT(App):
    def __init__(self, Ts, **kwargs):
        self.Ts = Ts

def FBFMTHT(engine, app):
    hall = engine.apps[app.dependences[0]]
    hall.set(lambda i, j: engine.matrix(k=(i, j),scalefree=hall.scalefree,scaleint=hall.scaleint))
    result = np.zeros((len(app.Ts), 2), dtype=np.float64)
    name='%s_%s(%s,%s)'%(engine.tostr(), app.name, hall.scalefree, hall.scaleint)
    engine.log<<('%s: \n'%name)
    for i, T in enumerate(app.Ts):
        engine.log<<('%s, '%T, )
        hall.T = T
        result[i, 0] = T
        result[i, 1] = hall.compute()
    engine.log<<('\n')
    if app.savedata: np.savetxt('%s/%s.dat'%(engine.dout, name), result)
    if app.plot: app.figure('L', result, '%s/%s'%(engine.dout, name))
    if app.returndata: return result

class THP(App):
    def __init__(self, params, **kwargs):
        self.params = params

def FBFMTHP(engine, app):
    hall = engine.apps[app.dependences[0]]
    result = np.zeros((app.params.rank(0), 2), dtype=np.float64)
    name='%s_%s(%s,%s)_%s'%(engine.tostr(mask=app.params.tags), app.name, hall.scalefree, hall.scaleint, hall.T)
    engine.log<<('%s: \n'%name)
    for i, param in enumerate(app.params('+')):
        engine.log<<("%s, "%param)
        engine.update(**param)
        engine.tostr(mask=app.params.tags)
        hall.set(lambda i, j: engine.matrix(k=(i, j),scalefree=hall.scalefree,scaleint=hall.scaleint))
        result[i, 0] = list(param.values())[0]
        result[i, 1] = hall.compute()
    engine.log<<'\n'
    if app.savedata: np.savetxt('%s/%s.dat'%(engine.dout, name), result)
    if app.plot: app.figure('L', result, '%s/%s'%(engine.dout, name))
    if app.returndata: return result    

from scipy.integrate import quad
bose = lambda t, e: 1/(np.exp(e/t)-1)
def ploylogarithm(x, n, tol=5*10**-10):
    sum = x
    order = 1
    delta = 1.0
    while np.abs(delta)>tol:
        order += 1
        delta = x**order/order**n
        sum += delta
    return sum

coeff_1 = lambda n: quad(lambda x: (np.log((1+x)/x))**2, 0, n)[0]# - np.pi**2/3
coeff_2 = lambda n, tol=5*10**-10: (1+n)*(np.log((1+n)/n))**2 - (np.log(n))**2 - 2*ploylogarithm(-n, 2, tol=tol)# - np.pi**2/3

def iscontinuum(es, region):
    dEs = (es[1:]-es[:-1])/(es.max()-es.min())
    return dEs[region-1] < np.sum(dEs[region:])

from HamiltonianPy import *
import numpy as np

__all__=['name1','name2','nnb','parametermap','idfmap','haldane','km','t','ci','ti','dm','Ua','Ub','t2l','t2r','t2a','dml','dmr','Ual','Uar','Ubl','Ubr','H2','H4']

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
ci=lambda **parameters: Hopping('t2',parameters['t2'],neighbour=2,indexpacks=lambda bond: haldane(bond,parameters['phi']),statistics='f')
ti=lambda **parameters: Hopping('t2',parameters['t2'],neighbour=2,indexpacks=lambda bond: km(bond,parameters['phi']),statistics='f')
dm=lambda **parameters: Onsite('dm',parameters['dm'],indexpacks=sigmaz('sl'),statistics='f')
Ua=lambda **parameters: Hubbard('Ua',parameters['Ua'],atom=0,statistics='f')
Ub=lambda **parameters: Hubbard('Ub',parameters['Ub'],atom=1,statistics='f')

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

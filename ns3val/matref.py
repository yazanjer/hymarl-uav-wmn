import scipy.io as sio, numpy as np
import os
RES=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results')+'/'
def matref():
    """per-seed MATLAB means: dict method -> array (8 x 11), metric names"""
    R=sio.loadmat(RES+'headline_A8log.mat',squeeze_me=True,struct_as_record=False)['R']
    mn=list(R.metricNames); out={}
    for j,m in enumerate(R.methods): out[str(m)]=np.asarray(R.Data[:,j,:])
    H=sio.loadmat(RES+'hybrid.mat',squeeze_me=True,struct_as_record=False)['hyb']
    out['Hybrid']=np.asarray(H.A10.Data)
    I=sio.loadmat(RES+'improve.mat',squeeze_me=True,struct_as_record=False)['imp']
    h=I.heur

    return out,mn,I
if __name__=='__main__':
    out,mn,I=matref(); print(mn)
    h=I.heur; 
    for f in h._fieldnames: 
        v=getattr(h,f); print(f, type(v), getattr(v,'_fieldnames',None) or np.shape(v))

def allref():
    out,mn,I=matref()
    a=I.heur.A10; meths=[str(x) for x in a.methods]; D=np.asarray(a.Data)

    out['HeurMC']=D[:,meths.index('Heur-MC-LB'),:] if D.ndim==3 else D
    return out,mn

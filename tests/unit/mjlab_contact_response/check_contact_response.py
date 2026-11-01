import json,sys,numpy as np,warp as wp
from pathlib import Path
from holosoma.simulator.mjlab import contact_response as c
wp.init();d='cpu'
arr=lambda x,dtype=wp.float32:wp.array(np.asarray(x),dtype=dtype,device=d)
n=arr([2],wp.int32);world=arr([0,0],wp.int32);dt=arr([.005]);ref=arr([[.02,.3],[.02,1.]],wp.vec2)
wp.launch(c._fixed_stiffness,dim=2,inputs=[n,world,dt,0,ref],device=d)
r=ref.numpy();np.testing.assert_allclose(r,[[-2500,-30],[-2500,-100]],rtol=1e-6)
imp=arr([[.9,.95,.001,.5,2]]*2,c.vec5);dims=arr([3,3],wp.int32);addr=arr([[0,1,2,3],[4,5,6,7]],wp.int32)
v=arr([[2,2,2,2,0,0,0,0]]);pos=arr([[-.03]*4+[.1]*4]);margin=arr([[0]*8]);a0=np.array([[900.]*8],np.float32);a=arr(a0)
wp.launch(c._cap_penetration_bias,dim=(2,4),inputs=[n,world,dims,addr,int(c.ConeType.PYRAMIDAL),ref,imp,v,pos,margin,1.,a],device=d)
np.testing.assert_allclose(a.numpy()[0,:4],-30/.95,rtol=1e-6);np.testing.assert_array_equal(a.numpy()[0,4:],a0[0,4:])
a=arr(a0);wp.launch(c._cap_penetration_bias,dim=(2,4),inputs=[n,world,dims,addr,int(c.ConeType.ELLIPTIC),ref,imp,v,pos,margin,1.,a],device=d)
np.testing.assert_allclose(a.numpy()[0,0],-30/.95,rtol=1e-6);np.testing.assert_array_equal(a.numpy()[0,1:],a0[0,1:])
# Explicit direct references must not be mapped a second time.
wp.launch(c._fixed_stiffness,dim=2,inputs=[n,world,dt,0,ref],device=d);np.testing.assert_array_equal(ref.numpy(),r)
# Both critical damping/e=0 and timestep safety preserve original baseline stiffness.
ref2=arr([[.001,1.],[.02,1.]],wp.vec2);wp.launch(c._fixed_stiffness,dim=2,inputs=[n,world,dt,0,ref2],device=d);np.testing.assert_allclose(ref2.numpy(),[[-10000,-200],[-2500,-100]],rtol=1e-6)
report={'passed':True,'fixed_stiffness_for_different_damping':r.tolist(),'pyramidal_bias_velocity_cap':1,'elliptic_tangential_rows_unchanged':True,'separated_rows_unchanged':True,'direct_reference_not_remapped':True,'refsafe_dt_floor_preserved':True,'no_gpu_used':True}
if len(sys.argv) > 1:
    output = Path(sys.argv[1]); output.mkdir(parents=True, exist_ok=True)
    (output / 'cpu_acceptance.json').write_text(json.dumps(report, indent=2))
print(report)

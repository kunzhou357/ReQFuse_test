"""Inspect VI restoration/state saturation on known training degradations."""
from pathlib import Path
import sys
import json

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model.fusion import RobustFusionNet
from utils.image import read_image
from utils.degradation import LocalDegrader
from utils.auxiliary_loss import AuxiliaryLoss


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    state=torch.load(ROOT/'ckpt/last.pth',map_location='cpu',weights_only=True)
    model=RobustFusionNet(state['config']['model']).eval()
    model.load_state_dict(state['model'],strict=True)
    auxiliary=AuxiliaryLoss(**state['config']['auxiliary_loss'])
    records=[]
    for name,(top,left) in [('00123D',(104,120)),('00024N',(128,400))]:
        sources=ROOT.parent/'datasets/test'
        vi=read_image(sources/'vis'/(name+'.png'),3)[:,top:top+128,left:left+128]
        ir=read_image(sources/'ir'/(name+'.png'),1)[:,top:top+128,left:left+128]
        for kind in ('noise','blur','contrast','exposure'):
            degrader=LocalDegrader({'pair_probabilities':[0,1,0,0],
                    'local_probability':0.,'visible_types':[kind]})
            degraded_vi,degraded_ir,_,_,_=degrader(vi,ir,torch.Generator().manual_seed(123))
            head_stats={}
            def hook(key):
                def record(module,args,out):
                    z=out.float()
                    head_stats[key]={'min':float(z.min()),'max':float(z.max()),
                                     'mean':float(z.mean()),'std':float(z.std())}
                return record
            handles=[module.register_forward_hook(hook(key)) for key,module in [
                ('vi_demand_logits',model.restorer_visible.demand_head),
                ('vi_error_logits',model.error_visible.head),
                ('vi_feature',model.restorer_visible.refinement),
                ('ir_demand_logits',model.restorer_infrared.demand_head)]]
            try:
                result=model(degraded_vi[None],degraded_ir[None],restoration_only=True)
            finally:
                for handle in handles:handle.remove()
            target=auxiliary.error_map(degraded_vi[None],vi[None])
            restored=result['restored']['visible']
            row={'sample':name,'type':kind,
                 'input_ref_mae':float((degraded_vi-vi).abs().mean()),
                 'restored_ref_mae':float((restored-vi).abs().mean()),
                 'restored_input_mae':float((restored-degraded_vi).abs().mean()),
                 'demand_mean':float(result['demand']['visible'].mean()),
                 'demand_target_mean':float((target/.1).clamp(0,1).mean()),
                 'quality_mean':float(result['quality']['visible'].mean()),
                 'error_target_mean':float(auxiliary.error_map(restored,vi[None]).mean()),
                 'pred_error_mean':float(result['error']['visible'].mean()),
                 'heads':head_stats}
            records.append(row)
            print(json.dumps(row),flush=True)
    output=ROOT/'tests/diagnosis_20261007';output.mkdir(parents=True,exist_ok=True)
    (output/'visible_branch_probes.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':main()

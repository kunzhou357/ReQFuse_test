"""Optimize pixels, not model weights, to inspect the four-loss objective.

These per-example fits are diagnostic illustrations, not test-set training or
results to report as a fusion method. Nothing in model/ or utils/ is changed.
"""
from pathlib import Path
import sys
import json

import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tests'))
from utils.image import read_image,save_image
from utils.loss import FusionLoss,to_luminance,SobelGradient
from diagnose_fusion import tile,terms,checker_phase


def main():
    torch.set_num_threads(4)
    out=ROOT/'tests/diagnosis_20261007';out.mkdir(parents=True,exist_ok=True)
    vi=read_image(ROOT.parent/'datasets/test/vis/00123D.png',3)[None,...,144:272,128:256]
    ir=read_image(ROOT.parent/'datasets/test/ir/00123D.png',1)[None,...,144:272,128:256]
    previous=read_image(ROOT/'results/test/rgb/00123D.png',3)[None,...,144:272,128:256]
    target=torch.maximum(to_luminance(vi),ir)
    initial=(vi+target-to_luminance(vi)).clamp(.0001,.9999)
    images={'VI':vi,'IR':ir,'max_Y_plus_VI_chroma':initial,'trained_network':previous}
    records={'note':'Per-image pixel fits for objective diagnosis only; no network training.'}
    modes={'current_10_10_1_1':(10.,10.,1.,1.),'intensity50_gradient10':(50.,10.,1.,1.),
           'intensity10_gradient2':(10.,2.,1.,1.)}
    for name,weights in modes.items():
        criterion=FusionLoss(*weights,require_color=True)
        p=torch.nn.Parameter(torch.logit(initial.clone()))
        optimizer=torch.optim.Adam([p],lr=.04)
        history=[]
        for step in range(300):
            optimizer.zero_grad(set_to_none=True)
            image=p.sigmoid();loss=criterion(image,vi,ir)
            loss.backward();optimizer.step()
            if step in (0,99,199,299):history.append({'step':step+1,'loss':float(loss.detach())})
        with torch.no_grad():
            final=p.sigmoid()
            images[name]=final
            row={'history':history,'initial_terms':terms(initial,vi,ir,criterion),
                 'final_terms':terms(final,vi,ir,criterion),
                 'trained_network_terms':terms(previous,vi,ir,criterion),
                 'phase':checker_phase(to_luminance(final)-target)}
            records[name]=row
            save_image(final,out/'objective_probe'/(name+'.png'))
        print(name,json.dumps(row),flush=True)
    canvas=Image.new('RGB',(4*262+6,2*286+6),(24,27,32))
    for idx,(label,img) in enumerate(images.items()):
        tile(canvas,6+(idx%4)*262,6+(idx//4)*286,img[...,20:96,28:96],label,256,280)
    canvas.save(out/'objective_probe_crops.png')
    (out/'objective_probe.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':main()

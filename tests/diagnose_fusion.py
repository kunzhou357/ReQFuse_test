"""Read-only checkpoint diagnosis. All generated evidence stays under tests.

Runs on the supplied four examples; this is not training or benchmark evaluation.
"""
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.fusion import RobustFusionNet
from utils.image import read_image, save_image
from utils.loss import FusionLoss, SobelGradient, to_luminance
from utils.degradation import LocalDegrader

OUT = ROOT / 'tests' / 'diagnosis_20261007'
SOURCES = ROOT.parent / 'datasets' / 'test'
RESULTS = ROOT / 'results' / 'test'
CASES = {
    '00004N': (496, 188, 576, 352),
    '00008N': (204, 184, 276, 256),
    '00024N': (432, 144, 504, 336),
    '00123D': (144, 100, 224, 292),
}


def phase_energy(x):
    """2x2 phase energy after removing each block's mean, in source units."""
    z = to_luminance(x).float()
    h, w = z.shape[-2:]
    z = z[..., :h//2*2, :w//2*2]
    blocks = F.pixel_unshuffle(z, 2)
    d = blocks - blocks.mean(1, keepdim=True)
    return float(d.abs().mean())


def checker_phase(x):
    z=to_luminance(x).float()
    smooth=F.avg_pool2d(F.pad(z,(2,2,2,2),mode='replicate'),5,1)
    z=z-smooth
    means=[float(z[...,i::2,j::2].mean()) for i in range(2) for j in range(2)]
    return {'phase_means':means,'phase_range':max(means)-min(means)}


def terms(image, vi, ir, criterion):
    total, parts = criterion(image, vi, ir, return_components=True)
    return dict(total=float(total), **{key:float(v) for key,v in parts.items()})


def crop(x, box):
    l,t,r,b = box
    return x[..., t:b, l:r]


def tile(canvas, x, y, image, label, width=200, height=220):
    draw = ImageDraw.Draw(canvas)
    draw.text((x+3,y+3),label,fill='white')
    a = image.detach().float().clamp(0,1)[0].permute(1,2,0).numpy()
    a = np.round(a*255).astype(np.uint8)
    if a.shape[-1] == 1:
        a = a[...,0]
    im = Image.fromarray(a).convert('RGB')
    scale = min(width/im.width, (height-25)/im.height)
    im = im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.NEAREST)
    canvas.paste(im,(x+(width-im.width)//2,y+25))


@torch.inference_mode()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    state = torch.load(ROOT/'ckpt'/'last.pth',map_location='cpu',weights_only=True)
    model = RobustFusionNet(state['config']['model']).eval()
    model.load_state_dict(state['model'],strict=True)
    criterion = FusionLoss(**state['config']['fusion_loss'])
    records = {'checkpoint_step':state['step'],'config':state['config'],'cases':{},'synthetic_probes':{}}
    montage = Image.new('RGB',(5*206+6,len(CASES)*226+6),(24,27,32))
    for row,(name,box) in enumerate(CASES.items()):
        vi = read_image(SOURCES/'vis'/(name+'.png'),3)[None]
        ir = read_image(SOURCES/'ir'/(name+'.png'),1)[None]
        t = time.perf_counter()
        result = model(vi,ir)
        fused = result['fused']
        previous = read_image(RESULTS/'rgb'/(name+'.png'),3)[None]
        target = torch.maximum(to_luminance(vi),ir)
        sample = {'seconds':time.perf_counter()-t,
                  'reproduction_png_mae':float((fused-previous).abs().mean()),
                  'loss_full':terms(fused,vi,ir,criterion),
                  'loss_roi':terms(crop(fused,box),crop(vi,box),crop(ir,box),criterion),
                  'state':{},'roi_phase_energy':{}}
        for mod,source in [('visible',vi),('infrared',ir)]:
            q=result['quality'][mod];a=result['demand'][mod];e=result['error'][mod]
            sample['state'][mod] = dict(demand_mean=float(a.mean()),demand_max=float(a.max()),
                quality_mean=float(q.mean()),error_mean=float(e.mean()),
                restored_source_mae=float((result['restored'][mod]-source).abs().mean()))
        for label,img in [('vi',vi),('ir',ir),('restored_ir',result['restored']['infrared']),
                          ('fused',fused),('max_intensity',target)]:
            sample['roi_phase_energy'][label]=phase_energy(crop(img,box))
        for col,(label,img) in enumerate([('VI',vi),('IR',ir),('Restored IR',result['restored']['infrared']),
                                          ('Fused',fused),('Max intensity target',target)]):
            tile(montage,6+col*206,6+row*226,crop(img,box),name+' '+label)
        # Positive source gradients do not constrain the sign or phase of a texture.
        y=crop(to_luminance(fused),box);v=crop(to_luminance(vi),box);i=crop(ir,box)
        strong=(i>.5)&(i>v+.2)
        sample['roi_ir_dominant_pixels']=int(strong.sum())
        if strong.any():
            sample['roi_ir_dominant']={'ir_mean':float(i[strong].mean()),'fused_mean':float(y[strong].mean()),
                 'fraction_fused_below_ir_by_0.2':float((y[strong]<i[strong]-.2).float().mean())}
        save_image(fused,OUT/'reproduced'/(name+'.png'))
        records['cases'][name]=sample
        print('CASE',name,json.dumps(sample),flush=True)
        (OUT/'measurements.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    montage.save(OUT/'person_crops.png')
    # Known synthetic degradation, using the training generator and one 128 crop.
    name='00024N';vi=read_image(SOURCES/'vis'/(name+'.png'),3)[:,128:256,400:528]
    ir=read_image(SOURCES/'ir'/(name+'.png'),1)[:,128:256,400:528]
    for kind in ('noise','blur','contrast'):
        d=LocalDegrader({'pair_probabilities':[0,0,0,1], 'local_probability':0.,
                        'visible_types':[kind],'infrared_types':[kind]})
        vd,id,_,_,_=d(vi,ir,torch.Generator().manual_seed(123))
        result=model(vd[None],id[None])
        row={}
        for mod,source,reference in [('visible',vd[None],vi[None]),('infrared',id[None],ir[None])]:
            restored=result['restored'][mod];q=result['quality'][mod];a=result['demand'][mod]
            row[mod]=dict(input_reference_mae=float((source-reference).abs().mean()),
                restored_reference_mae=float((restored-reference).abs().mean()),
                restored_input_mae=float((restored-source).abs().mean()),
                demand_mean=float(a.mean()),demand_max=float(a.max()),quality_mean=float(q.mean()),
                predicted_error_mean=float(result['error'][mod].mean()))
        records['synthetic_probes'][kind]=row
        print('SYNTHETIC',kind,json.dumps(row),flush=True)
        (OUT/'measurements.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    # Counterfactual inference of existing weights, not trained method variants.
    name='00123D';box=(156,164,224,240)
    vi=read_image(SOURCES/'vis'/(name+'.png'),3)[None]
    ir=read_image(SOURCES/'ir'/(name+'.png'),1)[None]
    # Use float output, so PNG quantization is not counted as a model change.
    base=model(vi,ir,return_aux=False)
    target=torch.maximum(to_luminance(vi),ir)
    variants={'baseline':base}
    variants['quality_strength_zero']=model(vi,ir,quality_strength=0.,return_aux=False)
    handles=[]
    for m in list(model.cross_i_to_v.values())+list(model.cross_v_to_i.values()):
        handles.append(m.register_forward_hook(lambda module,args,output:torch.zeros_like(output)))
    variants['zero_cross_messages']=model(vi,ir,return_aux=False)
    for h in handles:h.remove()
    for dy,dx in ((0,1),(1,0),(1,1),(0,8)):
        def shift(z):return F.pad(z,(dx,0,dy,0),mode='replicate')[...,:z.shape[-2],:z.shape[-1]]
        pred=model(shift(vi),shift(ir),return_aux=False)
        pred=F.pad(pred[...,dy:,dx:],(0,dx,0,dy),mode='replicate')
        variants[f'shift_back_{dy}_{dx}']=pred
    # Equalize the four subpixel channel groups in the last fusion upsampler.
    def equalize(module,args,output):
        b,c,h,w=output.shape
        return output.reshape(b,c//4,4,h,w).mean(2,keepdim=True).expand(-1,-1,4,-1,-1).reshape_as(output)
    handle=model.backbone.up[2][0].register_forward_hook(equalize)
    variants['last_upsample_phase_mean']=model(vi,ir,return_aux=False)
    handle.remove()
    canvas=Image.new('RGB',(4*246+6,2*266+6),(24,27,32))
    records['counterfactual_inference']={}
    for idx,(key,pred) in enumerate(variants.items()):
        row={'full_mae_vs_baseline':float((pred-base).abs().mean()),
             'roi_mae_vs_baseline':float((crop(pred,box)-crop(base,box)).abs().mean()),
             'roi_phase':checker_phase(crop(to_luminance(pred)-target,box)),
             'loss_roi':terms(crop(pred,box),crop(vi,box),crop(ir,box),criterion)}
        records['counterfactual_inference'][key]=row
        tile(canvas,6+(idx%4)*246,6+(idx//4)*266,crop(pred,box),key,240,260)
        save_image(pred,OUT/'counterfactual'/(key+'.png'))
        print('COUNTERFACTUAL',key,json.dumps(row),flush=True)
    canvas.save(OUT/'counterfactual_crops.png')
    yy,xx=torch.meshgrid(torch.arange(32),torch.arange(32),indexing='ij')
    checker=.5+.25*((yy+xx)%2*2-1).float()[None,None]
    sobel=SobelGradient()
    records['sobel_blind_spot']={'checker_pixel_range':[float(checker.min()),float(checker.max())],
        'checker_gradient_interior_max':float(sobel(checker)[...,1:-1,1:-1].max()),
        'horizontal_first_difference_mean':float((checker[...,1:]-checker[...,:-1]).abs().mean())}
    (OUT/'measurements.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('Saved',OUT,flush=True)


if __name__=='__main__':
    main()

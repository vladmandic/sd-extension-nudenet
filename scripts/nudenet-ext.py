import gradio as gr
from modules import scripts, scripts_postprocessing, processing, images # pylint: disable=import-error
import nudenet


def create_ui(accordion=True):
    with gr.Accordion('NudeNet', open = False, elem_id='nudenet') if accordion else gr.Group():
        with gr.Row():
            enabled = gr.Checkbox(label = 'Enabled', value = False)
            metadata = gr.Checkbox(label = 'Metadata', value = True)
            copy = gr.Checkbox(label = 'Save as copy', value = False)
        with gr.Row():
            score = gr.Slider(label = 'Sensitivity', value = 0.2, mininimum = 0, maximum = 1, step = 0.01, interactive=True)
            blocks = gr.Slider(label = 'Block size', value = 3, minimum = 1, maximum = 10, step = 1, interactive=True)
        with gr.Row():
            censor = gr.Dropdown(label = 'Censor', value = [], choices = sorted(nudenet.labels), multiselect=True, interactive=True)
        with gr.Row():
            method = gr.Dropdown(label = 'Method', value = 'pixelate', choices = ['none', 'pixelate', 'blur', 'image', 'block'], interactive=True)
        with gr.Row():
            overlay = gr.Textbox(label = 'Overlay', value = '', placeholder = 'Path to image or leave default', interactive=True)
    return [enabled, metadata, copy, score, blocks, censor, method, overlay]

def process(p: processing.StableDiffusionProcessing=None, pp: scripts.PostprocessImageArgs=None, enabled=True, metadata=True, copy=False, score=0.2, blocks=3, censor=[], method='pixelate', overlay=''):
    if not enabled:
        return
    if nudenet.detector is None:
        nudenet.detector = nudenet.NudeDetector() # loads and initializes model once
    nudes = nudenet.detector.censor(image=pp.image, method=method, min_score=score, censor=censor, blocks=blocks, overlay=overlay)
    if pp is None:
        nudenet.log.error('NudeNet: no image received')
    if len(censor) > 0: # replace image if anything is censored
        if not copy:
            pp.image = nudes.output
        else:
            info = processing.create_infotext(p)
            images.save_image(nudes.output, path=p.outpath_samples, seed=p.seed, prompt=p.prompt, info=info, p=p, suffix="-censored")
    if metadata:
        meta = '; '.join([f'{d["label"]}:{d["score"]}' for d in nudes.detections]) # add all metadata
        nsfw = any([d["label"] in nudenet.nsfw for d in nudes.detections])
        if p is not None:
            p.extra_generation_params["NudeNet"] = meta
            p.extra_generation_params["NSFW"] = nsfw
        if hasattr(pp, 'info'):
            pp.info['NudeNet'] = meta
            pp.info['NSFW'] = nsfw

class Script(scripts.Script):
    def title(self):
        return 'NudeNet'

    def show(self, _is_img2img):
        return scripts.AlwaysVisible

    def ui(self, _is_img2img):
        return create_ui(accordion=True)

    def postprocess_image(self, p: processing.StableDiffusionProcessing, pp: scripts.PostprocessImageArgs, enabled, metadata, copy, score, blocks, censor, method, overlay): # pylint: disable=arguments-differ
        process(p, pp, enabled, metadata, copy, score, blocks, censor, method, overlay)

class ScriptPostprocessing(scripts_postprocessing.ScriptPostprocessing):
    name = 'NudeNet'
    order = 10000

    def ui(self):
        enabled, metadata, copy, score, blocks, censor, method, overlay = create_ui(accordion=False)
        return { 'enabled': enabled, 'metadata': metadata, 'copy': copy, 'score': score, 'blocks': blocks, 'censor': censor, 'method': method, 'overlay': overlay }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enabled, metadata, copy, score, blocks, censor, method, overlay):
        process(None, pp, enabled, metadata, copy, score, blocks, censor, method, overlay)

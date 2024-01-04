# built-in imports and third party imports
import gradio as gr
# import required modules from sdnext
from modules import scripts, scripts_postprocessing, script_callbacks, processing, images # pylint: disable=import-error
# import actual nudenet module relative to extension root
import nudenet # pylint: disable=wrong-import-order


# main ui
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


# main processing used in both modes
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


# defines script for dual-mode usage
class Script(scripts.Script):
    # see below for all available options and callbacks
    # <https://github.com/vladmandic/automatic/blob/master/modules/scripts.py#L26>

    def title(self):
        return 'NudeNet'

    def show(self, _is_img2img):
        return scripts.AlwaysVisible

    # return signature is array of gradio components
    def ui(self, _is_img2img):
        return create_ui(accordion=True)

    # triggered by callback
    def postprocess_image(self, p: processing.StableDiffusionProcessing, pp: scripts.PostprocessImageArgs, enabled, metadata, copy, score, blocks, censor, method, overlay): # pylint: disable=arguments-differ
        process(p, pp, enabled, metadata, copy, score, blocks, censor, method, overlay)


# defines postprocessing script for dual-mode usage
class ScriptPostprocessing(scripts_postprocessing.ScriptPostprocessing):
    name = 'NudeNet'
    order = 10000

    # return signature is object with gradio components
    def ui(self):
        enabled, metadata, copy, score, blocks, censor, method, overlay = create_ui(accordion=False)
        return { 'enabled': enabled, 'metadata': metadata, 'copy': copy, 'score': score, 'blocks': blocks, 'censor': censor, 'method': method, 'overlay': overlay }

    # triggered by callback
    def process(self, pp: scripts_postprocessing.PostprocessedImage, enabled, metadata, copy, score, blocks, censor, method, overlay): # pylint: disable=arguments-differ
        process(None, pp, enabled, metadata, copy, score, blocks, censor, method, overlay)


# define api
def nudenet_api(_, app):
    from fastapi import Body
    from modules.api import api

    @app.post("/nudenet")
    async def nudenet_censor(
        image: str = Body("", title='nudenet input image'),
        score: float = Body(0.2, title='nudenet threshold score'),
        blocks: int = Body(3, title='nudenet pixelation blocks'),
        censor: list = Body([], title='nudenet censorship items'),
        method: str = Body('pixelate', title='nudenet censorship method'),
        overlay: str = Body('', title='nudenet overlay image path'),
    ):
        base64image = image
        image = api.decode_base64_to_image(image)
        if nudenet.detector is None:
            nudenet.detector = nudenet.NudeDetector() # loads and initializes model once
        nudes = nudenet.detector.censor(image=image, method=method, min_score=score, censor=censor, blocks=blocks, overlay=overlay)
        if len(censor) > 0: # replace image if anything is censored
            base64image = api.encode_pil_to_base64(nudes.output).decode("utf-8")
        detections_dict = { d["label"]: d["score"] for d in nudes.detections }
        return { "image": base64image, "detections": detections_dict }


script_callbacks.on_app_started(nudenet_api)

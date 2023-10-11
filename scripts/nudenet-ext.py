import gradio as gr
from modules import scripts, processing # pylint: disable=import-error
import nudenet


class Script(scripts.Script):
    def title(self):
        return 'NudeNet'

    def show(self, _is_img2img):
        return scripts.AlwaysVisible

    def ui(self, _is_img2img):
        with gr.Accordion('NudeNet', open = False, elem_id='nudenet'):
            with gr.Row():
                enabled = gr.Checkbox(label = 'Enabled', value = False)
                metadata = gr.Checkbox(label = 'Metadata', value = True)
            with gr.Row():
                score = gr.Slider(label = 'Sensitivity', value = 0.2, mininimum = 0, maximum = 1, step = 0.01, interactive=True)
                blocks = gr.Slider(label = 'Block size', value = 3, minimum = 1, maximum = 10, step = 1, interactive=True)
            with gr.Row():
                censor = gr.Dropdown(label = 'Censor', value = [], choices = sorted(nudenet.labels), multiselect=True, interactive=True)
            with gr.Row():
                method = gr.Dropdown(label = 'Method', value = 'pixelate', choices = ['none', 'pixelate', 'blur', 'image', 'block'], interactive=True)
            with gr.Row():
                overlay = gr.Textbox(label = 'Overlay', value = '', placeholder = 'Path to image or leave default', interactive=True)
        return [enabled, metadata, score, blocks, censor, method, overlay]

    def postprocess_image(self, p: processing.StableDiffusionProcessing, pp: scripts.PostprocessImageArgs, enabled, metadata, score, blocks, censor, method, overlay): # pylint: disable=arguments-differ
        if not enabled:
            return
        if nudenet.detector is None:
            nudenet.detector = nudenet.NudeDetector() # loads and initializes model once
        nudes = nudenet.detector.censor(image=pp.image, method=method, min_score=score, censor=censor, blocks=blocks, overlay=overlay)
        if len(censor) > 0: # replace image if anything is censored
            pp.image = nudes.output
        if metadata:
            p.extra_generation_params["NudeNet"] = '; '.join([f'{d["label"]}:{d["score"]}' for d in nudes.detections]) # add all metadata
